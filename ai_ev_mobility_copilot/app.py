import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

# -----------------------------
# Config + Path Helpers
# -----------------------------
st.set_page_config(page_title="Mercedes AI EV Mobility Copilot", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DB_PATH = Path(BASE_DIR) / "rag" / "chroma_db"


def p(*parts):
    return os.path.join(BASE_DIR, *parts)


# -----------------------------
# RAG + Claude Helpers
# -----------------------------
@st.cache_resource
def load_rag():
    client = chromadb.PersistentClient(path=str(KB_DB_PATH))
    col = client.get_or_create_collection("ev_kb")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return col, embed_model


def rag_retrieve(query: str, k: int = 3) -> str:
    col, embed_model = load_rag()
    q_emb = embed_model.encode([query]).tolist()[0]
    res = col.query(query_embeddings=[q_emb], n_results=k)
    docs = res.get("documents", [[]])[0]
    return "\n\n".join(docs) if docs else ""


def call_claude_for_params(user_prompt: str, kb_context: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "Missing ANTHROPIC_API_KEY. Claude is disabled."

    client = Anthropic(api_key=api_key)

    system = (
        "You are an EV trip planning assistant for a Mercedes EV Mobility Copilot prototype. "
        "Return STRICT JSON only with keys: safety_buffer_pct, max_charge_pct, objective "
        "(one of: 'min_time','min_cost','balanced','sustainability'), explanation."
    )

    msg = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=400,
        temperature=0.2,
        system=system,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Knowledge Base Context:\n{kb_context}\n\n"
                    f"User request:\n{user_prompt}\n\n"
                    "Return JSON only."
                ),
            }
        ],
    )

    text = msg.content[0].text.strip()
    return text, None


# -----------------------------
# Load Data
# -----------------------------
routes = pd.read_csv(p("data", "routes.csv"))
vehicles = pd.read_csv(p("data", "vehicle_profiles.csv"))
chargers = pd.read_csv(p("data", "chargers.csv"))

# -----------------------------
# Validation
# -----------------------------
required_route_cols = {
    "route_id",
    "route_key",
    "from_city",
    "to_city",
    "distance_miles",
    "typical_highway_speed_mph",
    "corridor",
}
required_vehicle_cols = {
    "vehicle_id",
    "name",
    "battery_kwh",
    "efficiency_wh_per_mile",
    "fast_charge_kw",
    "max_charge_pct",
}
required_charger_cols = {
    "charger_id",
    "name",
    "near_city",
    "network",
    "fast_charge_kw",
    "price_per_kwh",
    "corridor",
    "route_key",
    "mile_from_start",
}

missing_routes = required_route_cols - set(routes.columns)
missing_vehicles = required_vehicle_cols - set(vehicles.columns)
missing_chargers = required_charger_cols - set(chargers.columns)

if missing_routes or missing_vehicles or missing_chargers:
    st.error("CSV schema mismatch. Please fix missing columns.")
    if missing_routes:
        st.write(f"routes.csv missing: {sorted(list(missing_routes))}")
    if missing_vehicles:
        st.write(f"vehicle_profiles.csv missing: {sorted(list(missing_vehicles))}")
    if missing_chargers:
        st.write(f"chargers.csv missing: {sorted(list(missing_chargers))}")
    st.stop()


# -----------------------------
# Optimization Helpers
# -----------------------------
def wh_per_mile_with_speed(base_wh_mi: float, speed_mph: float) -> float:
    speed_penalty = max(0.0, (speed_mph - 60.0) * 0.01)
    return base_wh_mi * (1.0 + speed_penalty)


def drive_kwh(distance_mi: float, wh_mi: float) -> float:
    return (distance_mi * wh_mi) / 1000.0


def charge_minutes(kwh_to_add: float, vehicle_kw: float, station_kw: float) -> float:
    effective_kw = max(20.0, min(vehicle_kw, station_kw))
    return (kwh_to_add / effective_kw) * 60.0


def optimize_stops(
    route_distance_mi: float,
    speed_mph: float,
    start_soc_pct: float,
    battery_kwh: float,
    base_wh_mi: float,
    vehicle_fast_kw: float,
    max_charge_pct: float,
    chargers_df: pd.DataFrame,
    safety_buffer_pct: float = 10.0,
):
    """
    Finds a feasible charging plan that minimizes total trip time
    using simplified charging and driving assumptions.
    """
    wh_mi = wh_per_mile_with_speed(base_wh_mi, speed_mph)

    nodes = [{"type": "start", "mile": 0.0, "name": "Start"}]
    for _, r in chargers_df.sort_values("mile_from_start").iterrows():
        nodes.append(
            {
                "type": "charger",
                "mile": float(r["mile_from_start"]),
                "name": str(r["name"]),
                "near_city": str(r["near_city"]),
                "station_kw": float(r["fast_charge_kw"]),
                "price": float(r["price_per_kwh"]),
            }
        )
    nodes.append({"type": "dest", "mile": float(route_distance_mi), "name": "Destination"})

    max_soc_kwh = battery_kwh * (max_charge_pct / 100.0)
    safety_kwh = battery_kwh * (safety_buffer_pct / 100.0)

    step_kwh = 1.0
    energy_levels = np.arange(safety_kwh, max_soc_kwh + 0.001, step_kwh)

    start_kwh = battery_kwh * (start_soc_pct / 100.0)
    start_kwh = np.clip(start_kwh, safety_kwh, max_soc_kwh)

    INF = 1e18
    n = len(nodes)

    dp = np.full((n, len(energy_levels)), INF, dtype=float)
    prev = [[None for _ in range(len(energy_levels))] for _ in range(n)]

    e0_idx = int(np.argmin(np.abs(energy_levels - start_kwh)))
    dp[0, e0_idx] = 0.0

    for i in range(n - 1):
        for e_idx, e_kwh in enumerate(energy_levels):
            cur_time = dp[i, e_idx]
            if cur_time >= INF:
                continue

            charge_options = [e_kwh]
            if nodes[i]["type"] in ["start", "charger"]:
                options = list(
                    np.unique(
                        np.clip(
                            np.concatenate([np.arange(e_kwh, max_soc_kwh + 0.001, 5.0), [max_soc_kwh]]),
                            safety_kwh,
                            max_soc_kwh,
                        )
                    )
                )
                charge_options = options

            for charged_kwh in charge_options:
                extra_charge = max(0.0, charged_kwh - e_kwh)
                add_charge_time = 0.0

                if extra_charge > 0 and nodes[i]["type"] in ["start", "charger"]:
                    station_kw = 9999.0 if nodes[i]["type"] == "start" else nodes[i]["station_kw"]
                    add_charge_time = charge_minutes(extra_charge, vehicle_fast_kw, station_kw)

                for j in range(i + 1, n):
                    dist_ij = nodes[j]["mile"] - nodes[i]["mile"]
                    if dist_ij <= 0:
                        continue

                    need = drive_kwh(dist_ij, wh_mi)
                    remaining = charged_kwh - need

                    if remaining < safety_kwh:
                        break

                    drive_time = (dist_ij / max(1.0, speed_mph)) * 60.0
                    total_time = cur_time + add_charge_time + drive_time

                    rem_idx = int(np.argmin(np.abs(energy_levels - remaining)))
                    if total_time < dp[j, rem_idx]:
                        dp[j, rem_idx] = total_time
                        prev[j][rem_idx] = (i, e_idx, charged_kwh)

    dest_i = n - 1
    best_e = int(np.argmin(dp[dest_i]))
    best_time = float(dp[dest_i, best_e])

    if best_time >= INF:
        return None, None

    steps = []
    cur = (dest_i, best_e)

    while cur[0] != 0:
        i, e_idx = cur
        back = prev[i][e_idx]
        if back is None:
            break
        pi, pe_idx, charged_kwh = back
        steps.append((pi, pe_idx, i, e_idx, charged_kwh))
        cur = (pi, pe_idx)

    steps.reverse()

    plan = []
    for (pi, pe_idx, i, e_idx, charged_kwh) in steps:
        from_node = nodes[pi]
        to_node = nodes[i]
        from_energy = float(energy_levels[pe_idx])

        dist = to_node["mile"] - from_node["mile"]
        need = drive_kwh(dist, wh_mi)
        arrive = charged_kwh - need

        charge_added = max(0.0, charged_kwh - from_energy)
        charge_min = 0.0
        if charge_added > 0 and from_node["type"] in ["start", "charger"]:
            station_kw = 9999.0 if from_node["type"] == "start" else from_node["station_kw"]
            charge_min = charge_minutes(charge_added, vehicle_fast_kw, station_kw)

        plan.append(
            {
                "from": from_node["name"],
                "to": to_node["name"],
                "segment_miles": round(dist, 1),
                "charge_added_kwh": round(charge_added, 1),
                "charge_minutes": int(round(charge_min)),
                "arrive_kwh": round(arrive, 1),
                "arrive_soc_pct": int(round((arrive / battery_kwh) * 100)),
                "stop_city": from_node.get("near_city", ""),
            }
        )

    return plan, best_time


# -----------------------------
# UI Header
# -----------------------------
st.title("Mercedes AI EV Mobility Copilot")
st.caption("Road Trip EV charging planner with optimization, sustainability scoring, Claude reasoning, and RAG guidance.")

# -----------------------------
# Controls
# -----------------------------
colA, colB, colC = st.columns(3)

route_options = routes.apply(
    lambda r: f"{r['from_city']} → {r['to_city']} ({int(r['distance_miles'])} mi)",
    axis=1,
).tolist()

with colA:
    route_label = st.selectbox("Select Road Trip", route_options, index=0)
    route_idx = route_options.index(route_label)
    route = routes.iloc[route_idx]

with colB:
    vehicle_label = st.selectbox("Select Vehicle", vehicles["name"].tolist(), index=0)
    vehicle = vehicles[vehicles["name"] == vehicle_label].iloc[0]

with colC:
    start_soc = st.slider("Starting Battery (%)", 10, 100, 65, 5)

# -----------------------------
# Copilot Input
# -----------------------------
st.markdown("### Ask the Copilot")
user_request = st.text_input(
    "Trip preferences (natural language)",
    value="Minimize charging time, keep arrival SOC above 12%, sustainability focused.",
)

kb_context = rag_retrieve(user_request, k=3)

with st.expander("Retrieved EV guidance (RAG)"):
    st.write(kb_context if kb_context else "No context retrieved.")

# Defaults if Claude is off
def local_parse_trip_preferences(user_prompt: str, kb_context: str):
    text = (user_prompt or "").lower()

    safety_buffer_pct = 10.0
    max_charge_pct = 80.0
    objective = "min_time"

    if "12%" in text or "12 %" in text:
        safety_buffer_pct = 12.0
    elif "15%" in text or "15 %" in text:
        safety_buffer_pct = 15.0
    elif "20%" in text or "20 %" in text:
        safety_buffer_pct = 20.0

    if "sustainability" in text or "eco" in text or "efficient" in text:
        objective = "sustainability"
    elif "cheap" in text or "cheaper" in text or "cost" in text:
        objective = "min_cost"
    elif "balanced" in text:
        objective = "balanced"
    elif "minimize charging time" in text or "fastest" in text:
        objective = "min_time"

    explanation = (
        f"Using a {int(safety_buffer_pct)}% safety buffer and charging up to {int(max_charge_pct)}% "
        f"with a primary objective of {objective.replace('_',' ')}. "
        f"This recommendation incorporates EV best practices from the internal knowledge base."
    )

    return {
        "safety_buffer_pct": safety_buffer_pct,
        "max_charge_pct": max_charge_pct,
        "objective": objective,
        "explanation": explanation,
    }


# Defaults
safety_buffer_pct = 10.0
objective = "min_time"
ai_explanation = ""
max_charge_override = None

try:
    claude_json, claude_err = call_claude_for_params(user_request, kb_context)

    if claude_err:
        st.info(claude_err)
        parsed = local_parse_trip_preferences(user_request, kb_context)
    else:
        try:
            parsed = json.loads(claude_json)
        except Exception:
            st.warning("Claude response invalid. Using local AI parser.")
            parsed = local_parse_trip_preferences(user_request, kb_context)

except Exception as e:
    st.warning(f"Claude unavailable. Using local AI parser. Details: {e}")
    parsed = local_parse_trip_preferences(user_request, kb_context)

safety_buffer_pct = float(parsed.get("safety_buffer_pct", safety_buffer_pct))
max_charge_override = parsed.get("max_charge_pct", None)
objective = str(parsed.get("objective", objective))
ai_explanation = str(parsed.get("explanation", ""))


# -----------------------------
# Calculations
# -----------------------------
distance = float(route["distance_miles"])
speed = float(route["typical_highway_speed_mph"])

eff_wh_mi = float(vehicle["efficiency_wh_per_mile"])
battery_kwh = float(vehicle["battery_kwh"])
vehicle_fast_kw = float(vehicle["fast_charge_kw"])
vehicle_max_charge_pct = float(vehicle["max_charge_pct"])

effective_max_charge_pct = float(max_charge_override) if max_charge_override else vehicle_max_charge_pct

usable_kwh = battery_kwh * (start_soc / 100.0)

speed_penalty = max(0.0, (speed - 60.0) * 0.01)
adj_eff_wh_mi = eff_wh_mi * (1.0 + speed_penalty)
needed_kwh = (distance * adj_eff_wh_mi) / 1000.0

sust_score = int(np.clip(100 - ((adj_eff_wh_mi - 260) / 2.0), 0, 100))

route_corridor = str(route["corridor"]).strip()
route_key = str(route["route_key"]).strip()
route_chargers = chargers[chargers["route_key"].astype(str).str.strip() == route_key].copy()

# -----------------------------
# KPIs
# -----------------------------
st.markdown("---")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Trip Distance", f"{distance:.0f} mi")
k2.metric("Energy Needed (adj)", f"{needed_kwh:.1f} kWh")
k3.metric("Energy Available", f"{usable_kwh:.1f} kWh")
k4.metric("Feasible Without Charging", "Yes" if usable_kwh >= needed_kwh else "No")

st.markdown("### Sustainability")
st.write(
    "This is a prototype score based on efficiency and highway speed assumptions. "
    "Higher score indicates more energy-efficient trip planning assumptions."
)
st.metric("Sustainability Score", f"{sust_score}/100")

if ai_explanation:
    st.markdown("### Copilot Explanation")
    st.write(ai_explanation)

st.markdown("### AI Planning Parameters")
p1, p2, p3 = st.columns(3)
p1.metric("Safety Buffer", f"{safety_buffer_pct:.1f}%")
p2.metric("Max Charge Target", f"{effective_max_charge_pct:.0f}%")
p3.metric("Objective", objective)

# -----------------------------
# Charging Recommendation
# -----------------------------
st.markdown("### Charging Recommendation")
st.caption(f"Corridor: {route_corridor} | Route Key: {route_key}")

if usable_kwh >= needed_kwh:
    st.success("No charging stop required for this trip under current assumptions.")
else:
    st.warning("Charging stop plan optimized for total trip time.")
    if route_chargers.empty:
        st.info("No chargers found for this route. Add route_key and mile_from_start rows in chargers.csv.")
    else:
        plan, total_min = optimize_stops(
            route_distance_mi=distance,
            speed_mph=speed,
            start_soc_pct=start_soc,
            battery_kwh=battery_kwh,
            base_wh_mi=eff_wh_mi,
            vehicle_fast_kw=vehicle_fast_kw,
            max_charge_pct=effective_max_charge_pct,
            chargers_df=route_chargers,
            safety_buffer_pct=safety_buffer_pct,
        )

        if plan is None:
            st.error("No feasible plan found with current battery and charger set.")
        else:
            st.metric("Estimated Total Trip Time (drive + charging)", f"{int(round(total_min))} minutes")

            st.markdown("#### Optimized Charging Plan")
            st.dataframe(pd.DataFrame(plan), use_container_width=True)

            st.markdown("#### Chargers on This Route")
            st.dataframe(
                route_chargers.sort_values("mile_from_start")[
                    ["name", "near_city", "network", "fast_charge_kw", "price_per_kwh", "mile_from_start"]
                ],
                use_container_width=True,
            )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Prototype uses curated real road trips, corridor-tagged chargers, RAG-based EV guidance, "
    "and Claude for trip preference interpretation. Phase 2: live charger availability, real routing, and pricing optimization."
)