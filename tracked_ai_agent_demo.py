# tracked_ai_agent_demo.py
import numpy as np
import time
import os
import random
from energy_tracker import EnergyTracker

# --- Configuration ---
NUM_AGENT_TURNS = 5 # Number of times the agent "thinks" and "acts"
LOCAL_PROCESSING_COMPLEXITY = 50000 # Controls how much local CPU work is done
LLM_MODEL_ENERGY_PER_TOKEN_MJ = { # Dummy values, research real estimates! (in milliJoules)
    "GPT-4o": 0.05,  # Hypothetical MJ per token
    "Mixtral-8x7B-Instruct-v0.1": 0.02, # Hypothetical MJ per token
    "Custom_Local_LLM": 0.001 # Much lower if run locally
}
# Convert MJ to Joules for add_estimated_external_cost
def mj_to_joules(mj): return mj * 1000

# --- Simulate AI Agent Components ---

def simulate_llm_call(model_name, prompt_tokens, completion_tokens):
    """Simulates an LLM API call and estimates its energy/carbon cost."""
    total_tokens = prompt_tokens + completion_tokens
    # Estimate energy based on tokens and a dummy model cost
    energy_mj_per_token = LLM_MODEL_ENERGY_PER_TOKEN_MJ.get(model_name, 0.03) # Default if not found
    estimated_joules = mj_to_joules(total_tokens * energy_mj_per_token)
    print(f"    - Simulated LLM Call ({model_name}): {total_tokens} tokens. Est. Energy: {estimated_joules:.2f}J")
    return estimated_joules

def simulate_local_processing(complexity):
    """Simulates agent's internal thought process or data manipulation."""
    _ = np.random.rand(complexity, 10).dot(np.random.rand(10, 1)) # Dummy heavy computation
    time.sleep(0.1) # Simulate some minor real time

def simulate_tool_use(tool_name):
    """Simulates calling an external tool."""
    print(f"    - Agent using tool: {tool_name}")
    time.sleep(0.5) # Simulate API latency

def run_ai_agent_simulation():
    print("Starting AI Agent carbon footprint simulation...")

    tracker = EnergyTracker(project_name="AI_Agent_Carbon_Footprint", region="US") # Set your region
    tracker.start()

    print("\n--- Agent Initialization ---")
    tracker.track_step("agent_startup")
    simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY // 2)
    time.sleep(0.5)
    tracker.end_step("agent_startup")

    for i in range(NUM_AGENT_TURNS):
        print(f"\n--- Agent Turn {i+1}/{NUM_AGENT_TURNS} ---")
        tracker.track_step(f"agent_turn_{i+1}")

        # 1. Agent plans/reasons (often involves LLM call)
        tracker.track_step(f"turn_{i+1}_planning_with_llm")
        prompt_t = random.randint(50, 200)
        completion_t = random.randint(20, 100)
        llm_energy = simulate_llm_call("GPT-4o", prompt_t, completion_t)
        tracker.add_estimated_external_cost(llm_energy, f"LLM Call (GPT-4o) - Turn {i+1}")
        simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY) # Some local parsing of LLM response
        tracker.end_step(f"turn_{i+1}_planning_with_llm")

        # 2. Agent decides to use a tool (could be local or external)
        if i % 2 == 0:
            tracker.track_step(f"turn_{i+1}_tool_search")
            simulate_tool_use("WebSearch")
            # For simplicity, no specific external cost added for generic tool use in MVP
            tracker.end_step(f"turn_{i+1}_tool_search")
        else:
            tracker.track_step(f"turn_{i+1}_local_tool_processing")
            simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY * 0.8) # Heavier local tool
            tracker.end_step(f"turn_{i+1}_local_tool_processing")

        # 3. Agent reflects (local processing)
        tracker.track_step(f"turn_{i+1}_reflection")
        simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY * 0.5)
        tracker.end_step(f"turn_{i+1}_reflection")

        tracker.end_step(f"agent_turn_{i+1}")

    print("\n--- Agent Shutdown ---")
    tracker.track_step("agent_shutdown")
    simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY // 4)
    time.sleep(0.2)
    tracker.end_step("agent_shutdown")

    tracker.stop()
    results = tracker.get_results()

    print(f"\n--- AI Agent Carbon Footprint Report ({NUM_AGENT_TURNS} Turns) ---")
    print(f"Total Duration: {results['total_duration_sec']:.2f} seconds")
    print(f"Total Energy: {results['total_energy_kwh']:.6f} kWh")
    print(f"Total Carbon Emissions: {results['total_carbon_gco2']:.4f} gCO2")
    print("\n--- Detailed Step Report ---")
    for step, data in results["steps"].items():
        print(f"\nStep: {step}")
        print(f"  Duration: {data['duration']:.2f} seconds")
        print(f"  CPU Energy (Measured): {data['cpu_energy_joules'] / 3.6e6:.6f} kWh")
        if data['gpu_energy_joules'] > 0:
            print(f"  GPU Energy (Measured): {data['gpu_energy_joules'] / 3.6e6:.6f} kWh")
        if data['estimated_external_joules'] > 0: # Only show if external cost was added
            print(f"  External Energy (Estimated): {data['estimated_external_joules'] / 3.6e6:.6f} kWh")
            print(f"  External Carbon (Estimated): {data['estimated_external_gco2']:.4f} gCO2")
        print(f"  Total Step Carbon Emissions: {data['carbon_emissions_gco2']:.4f} gCO2")

if __name__ == "__main__":
    run_ai_agent_simulation()