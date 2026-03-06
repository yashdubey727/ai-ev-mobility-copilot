# AI EV Mobility Copilot

AI-powered EV road trip planning assistant that recommends optimized charging stops, incorporates sustainability-aware driving guidance, and supports natural language trip preferences using Retrieval-Augmented Generation (RAG) and optimization logic.

This project demonstrates how AI systems can power intelligent mobility copilots for electric vehicles by combining route-aware charging optimization, EV knowledge retrieval, and explainable recommendations.

---

# Project Objective

Long-distance EV travel still introduces friction for drivers due to:

- range anxiety
- inefficient charging decisions
- uncertainty around charging availability
- lack of proactive trip planning tools

The goal of this project is to design an **AI-powered mobility copilot** that assists EV drivers in planning efficient long-distance trips by predicting charging needs, recommending optimal charging stops, and integrating sustainability-aware driving guidance.

---

# System Architecture

![Architecture](assets/architecture.png)

### High-Level Flow

1. Driver enters trip request and preferences
2. System retrieves EV guidance from a knowledge base using RAG
3. Trip optimizer evaluates battery range, route distance, and charger characteristics
4. Charging stops are selected to minimize travel time while maintaining a safe battery buffer
5. Copilot generates a recommended charging plan with sustainability insights

---

# Key Features

- EV road trip planning prototype
- Battery-aware charging stop optimization
- Route-specific charger recommendation
- Sustainability scoring for driving efficiency
- Natural language trip preference input
- RAG-based EV charging guidance retrieval
- AI-assisted planning with graceful fallback logic when LLM services are unavailable

---

# AI System Design

The prototype combines multiple AI and optimization components.

## Retrieval-Augmented Knowledge (RAG)

A knowledge base of EV charging best practices is embedded using sentence-transformer embeddings and stored in a Chroma vector database.

Relevant guidance is retrieved dynamically during trip planning to provide context-aware recommendations.

## Preference Parsing

Drivers can describe their trip preferences in natural language such as:

- maintain higher battery buffer
- prioritize sustainability
- minimize charging time

An LLM-based parser interprets these preferences and converts them into structured planning parameters.

A local fallback parser ensures the system continues operating even when the external LLM service is unavailable.

## Charging Optimization Engine

The trip planning engine evaluates:

- vehicle battery capacity
- energy consumption per mile
- route distance
- charger power levels
- safety battery buffers

The optimizer selects charging stops that minimize total travel time while maintaining a safe battery reserve.

---

# Example Trip Scenario

Route: Los Angeles → San Francisco  
Vehicle: Mercedes EQE SUV  
Starting Battery: 65%

System Output:

- optimized charging stop recommendation
- estimated charging time
- estimated total trip duration
- sustainability score

---

## Live Demo

https://ai-ev-mobility-copilot.streamlit.app

# Product Thinking

This project explores how automotive AI systems can evolve from simple command-based assistants to **context-aware mobility copilots**.

Instead of requiring drivers to manually estimate charging stops, the system:

- predicts charging needs
- recommends efficient charging locations
- integrates sustainability considerations
- explains decisions using EV knowledge retrieval

---

# Project Structure
ai-ev-mobility-copilot
│
├── app.py
│
├── data
│ ├── routes.csv
│ ├── chargers.csv
│ └── vehicle_profiles.csv
│
├── rag
│ ├── knowledge.md
│ └── chroma_db
│
├── scripts
│ └── embed_knowledge.py
│
├── assets
│ └── architecture.png
│
└── README.md


---

# Running the Prototype

Clone the repository:
https://github.com/yashdubey727/ai-ev-mobility-copilot


Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py


---

# Future Improvements

- real-time charging station availability
- integration with navigation APIs
- pricing-aware charging optimization
- vehicle-specific charging curves
- driver behavior learning for energy prediction

---

# Why This Project Matters

This prototype demonstrates how AI can power intelligent EV mobility systems by combining machine learning, retrieval-based knowledge systems, and optimization algorithms to improve long-distance electric vehicle travel.

The approach illustrates how future automotive AI assistants could proactively guide drivers through complex mobility decisions rather than simply responding to commands.

# Business Impact

Estimated Business Impact

If deployed across EV fleets:

• Reduce EV trip planning friction by ~40%  
• Improve charging efficiency by ~15–20%  
• Increase driver confidence in long-distance EV travel  
• Strengthen digital mobility service adoption

