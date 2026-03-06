# AI EV Mobility Copilot

AI-powered EV road trip planning assistant that recommends optimized charging stops, incorporates sustainability-aware driving guidance, and supports natural language trip preferences using RAG and AI reasoning.

---

# Project Objective

Long-distance EV travel still creates friction for drivers due to:

• range anxiety  
• inefficient charging decisions  
• uncertainty about charging availability  
• lack of proactive trip planning tools  

This project demonstrates how AI can power a **Mobility Copilot** that proactively plans EV road trips using battery constraints, route-aware charging infrastructure, and sustainability guidance.

---

# Key Features

• Real road trip simulation across major US EV corridors  
• Battery-aware charging stop optimization  
• Sustainability scoring for route efficiency  
• RAG-based EV knowledge retrieval  
• Natural language trip preference parsing  
• AI-powered charging recommendation logic  
• Graceful fallback logic when LLM services are unavailable

---

# System Architecture

![Architecture](assets/architecture.png)

### High-Level Flow

1. Driver enters trip request and preferences  
2. System retrieves EV guidance using RAG knowledge base  
3. Trip optimizer evaluates battery range, charging speed, and route distance  
4. Charging stops are selected to minimize travel time while maintaining battery safety buffer  
5. Copilot presents optimized charging plan with sustainability score

---

# Prototype Screenshots

### Trip Planning Interface

![Hero](assets/mercedes_hero.png)

---

### EV Knowledge Retrieval (RAG)

![RAG](assets/mercedes_rag.png)

---

### Trip Metrics

![KPIs](assets/mercedes_kpis.png)

---

### Optimized Charging Plan

![Plan](assets/mercedes_plan.png)

---

# AI System Design

The prototype combines multiple AI and optimization components.

### RAG Knowledge Retrieval

A knowledge base of EV charging best practices is embedded using `sentence-transformers` and stored in a Chroma vector database. Relevant guidance is retrieved dynamically during trip planning.

### Preference Parsing

User trip preferences written in natural language are parsed to extract planning constraints such as:

• safety battery buffer  
• charging strategy  
• optimization objective  

Claude-based parsing is supported with a local fallback parser to ensure system resilience.

### Charging Optimization

The route optimizer evaluates:

• vehicle battery capacity  
• energy consumption per mile  
• charger power levels  
• route distance  

It selects charging stops that minimize total trip time while maintaining a safe arrival battery level.

---

# Example Trip Scenario

Route: Los Angeles → San Francisco  
Vehicle: Mercedes EQE SUV  
Starting Battery: 65%

Output:

• Optimized charging stop recommendation  
• Estimated charging time  
• Estimated trip duration  
• Sustainability score  

---

# Product Thinking

The goal of this project is to demonstrate how **AI can move in-car assistants from command-based systems to decision-support copilots**.

Instead of asking drivers to manually plan charging stops, the system:

• predicts charging needs  
• recommends efficient charging stations  
• integrates sustainability considerations  
• explains planning decisions using EV guidance

---

# Project Structure
ai-ev-mobility-copilot
│
├─ app.py
├─ data/
│ ├─ routes.csv
│ ├─ chargers.csv
│ └─ vehicle_profiles.csv
│
├─ rag/
│ ├─ knowledge.md
│ └─ chroma_db/
│
├─ scripts/
│ └─ embed_knowledge.py
│
├─ assets/
│ └─ architecture.png
│
└─ README.


---

# Running the Prototype

Clone the repository:
git clone https://github.com/yashdubey727/ai-ev-mobility-copilot.git


Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py


---

# Future Improvements

• Real-time charging station availability  
• Integration with real map routing APIs  
• pricing-aware charging optimization  
• vehicle-specific charging curves  
• personalized driver energy profiles

---

# Why This Project Matters

This prototype demonstrates how AI can power **intelligent EV mobility systems**, combining machine learning, optimization, and product design to improve long-distance electric vehicle travel.
