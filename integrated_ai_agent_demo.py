import numpy as np
import time
import os
import random
from energy_tracker import EnergyTracker
import openai # New import
from openai import OpenAI # New import for the client

# --- Configuration ---
NUM_AGENT_TURNS = 3 # Reduce turns for quicker demo with real API calls
LOCAL_PROCESSING_COMPLEXITY = 50000

# Estimated energy cost per 1000 tokens for different models (research these values, they vary!)
# These are rough estimates and can be highly debated. For MVP, use illustrative values.
# Values are in Joules per 1000 tokens.
# Example: 1 token ~ 0.00034Wh = 0.001224 Joules (if 1 query = 0.34Wh for 1000 tokens)
# So, 1000 tokens = 1.224 Joules
LLM_ENERGY_JOULES_PER_1000_TOKENS = {
    "gpt-4o-2024-05-13": 1.5, # Example: 1.5 Joules per 1000 tokens (adjust based on research)
    "gpt-3.5-turbo": 0.5,     # Example: 0.5 Joules per 1000 tokens
    "mixtral-8x7b-instruct-v0.1": 0.8, # Assuming hosted inference
    "llama-3-8b-instruct": 0.3 # Assuming hosted inference
}

# Initialize OpenAI client (API key read from OPENAI_API_KEY environment variable)
client = OpenAI()

# --- Simulate AI Agent Components ---

def make_llm_call_and_estimate_energy(model_name, prompt_content):
    """
    Makes an actual LLM API call and returns its estimated energy cost in Joules.
    """
    print(f"    - Making actual LLM Call ({model_name})...")
    try:
        start_llm_call_time = time.time()
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt_content}
            ]
        )
        end_llm_call_time = time.time()
        llm_duration = end_llm_call_time - start_llm_call_time

        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens

        energy_joules_per_1000 = LLM_ENERGY_JOULES_PER_1000_TOKENS.get(model_name, 1.0)
        estimated_joules = (total_tokens / 1000) * energy_joules_per_1000

        print(f"      Response received in {llm_duration:.2f}s. Tokens: {total_tokens}. Est. Energy: {estimated_joules:.4f}J")
        return estimated_joules, total_tokens, completion.choices[0].message.content
    except openai.APIError as e:
        print(f"      LLM API Error: {e}")
        return 0, 0, f"LLM API Error: {e}"
    except Exception as e:
        print(f"      An unexpected error occurred during LLM call: {e}")
        return 0, 0, f"Error: {e}"

def simulate_local_processing(complexity, description="processing"):
    """Simulates agent's internal thought process or data manipulation."""
    start_time = time.time()
    # FIX: Convert complexity to an integer
    effective_complexity = int(complexity) 
    _ = np.random.rand(effective_complexity, 10).dot(np.random.rand(10, 1)) # Dummy heavy computation
    time.sleep(0.05) # Simulate some minor real time
    end_time = time.time()
    print(f"    - Simulated local {description}: {end_time - start_time:.2f}s")

def simulate_tool_use(tool_name):
    """Simulates calling an external tool."""
    print(f"    - Agent using tool: {tool_name}")
    time.sleep(0.5) # Simulate API latency


def run_ai_agent_simulation_with_llm():
    print("Starting AI Agent carbon footprint simulation with real LLM calls...")

    tracker = EnergyTracker(project_name="Real_AI_Agent_Carbon_Footprint", region="US") # Set your region
    tracker.start()

    print("\n--- Agent Initialization ---")
    tracker.track_step("agent_startup")
    simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY // 2, "startup")
    time.sleep(0.5)
    tracker.end_step("agent_startup")

    llm_model_to_use = "gpt-4o-2024-05-13" # Or "gpt-3.5-turbo" if you want faster/cheaper API calls

    for i in range(NUM_AGENT_TURNS):
        print(f"\n--- Agent Turn {i+1}/{NUM_AGENT_TURNS} ---")
        tracker.track_step(f"agent_turn_{i+1}")

        # 1. Agent plans/reasons using an LLM
        tracker.track_step(f"turn_{i+1}_planning_with_llm")
        agent_query = f"Agent, for turn {i+1}, how would you respond to a user asking about sustainable software? Be concise."
        llm_energy, tokens, llm_response = make_llm_call_and_estimate_energy(llm_model_to_use, agent_query)
        tracker.add_estimated_external_cost(llm_energy, f"LLM Call ({llm_model_to_use}) - Turn {i+1} ({tokens} tokens)")
        print(f"      LLM Response (truncated): '{llm_response[:80]}...'")
        simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY, "LLM response parsing") # Some local parsing of LLM response
        tracker.end_step(f"turn_{i+1}_planning_with_llm")

        # 2. Agent decides to use a tool or local action
        if i % 2 == 0:
            tracker.track_step(f"turn_{i+1}_tool_execution")
            simulate_tool_use("WebSearch API")
            simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY * 0.5, "tool result processing")
            tracker.end_step(f"turn_{i+1}_tool_execution")
        else:
            tracker.track_step(f"turn_{i+1}_local_data_analysis")
            simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY * 0.8, "local data analysis")
            tracker.end_step(f"turn_{i+1}_local_data_analysis")

        # 3. Agent reflects (local processing)
        tracker.track_step(f"turn_{i+1}_reflection")
        simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY * 0.5, "reflection")
        tracker.end_step(f"turn_{i+1}_reflection")

        tracker.end_step(f"agent_turn_{i+1}")

    print("\n--- Agent Shutdown ---")
    tracker.track_step("agent_shutdown")
    simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY // 4, "shutdown")
    time.sleep(0.2)
    tracker.end_step("agent_shutdown")

    tracker.stop()
    results = tracker.get_results()

    print(f"\n--- AI Agent Carbon Footprint Report ({NUM_AGENT_TURNS} Turns) ---")
    print(f"Total Duration: {results['total_duration_sec']:.2f} seconds")
    print(f"Total Energy: {results['total_energy_kwh']:.6f} kWh")
    print(f"Total Carbon Emissions: {results['total_carbon_gco2']:.4f} gCO2")
    print("\n--- Detailed Step Report ---")
    # Pretty print the results for clarity
    for step, data in results["steps"].items():
        print(f"\nStep: {step}")
        print(f"  Duration: {data['duration']:.2f} s")
        print(f"  CPU Energy (Measured): {data['cpu_energy_joules'] / 3.6e6:.6f} kWh")
        if data['gpu_energy_joules'] > 0:
            print(f"  GPU Energy (Measured): {data['gpu_energy_joules'] / 3.6e6:.6f} kWh")
        if data.get('estimated_external_joules', 0) > 0: # Check using .get for robustness
            print(f"  External Energy (Estimated): {data['estimated_external_joules'] / 3.6e6:.6f} kWh")
            print(f"  External Carbon (Estimated): {data['estimated_external_gco2']:.4f} gCO2")
        print(f"  Total Step Carbon Emissions: {data['carbon_emissions_gco2']:.4f} gCO2")
    print("-" * 40)


    # --- New: LLM for Post-Analysis of the Report ---
    print("\n--- Asking LLM for Insights on the Energy Report ---")
    report_summary_for_llm = []
    report_summary_for_llm.append(f"Overall Carbon Footprint: {results['total_carbon_gco2']:.4f} gCO2 over {results['total_duration_sec']:.2f} seconds.")
    report_summary_for_llm.append("Breakdown by step:")
    for step, data in results["steps"].items():
        report_summary_for_llm.append(f"- {step}: Duration {data['duration']:.2f}s, CPU {data['cpu_energy_joules'] / 3.6e6:.6f} kWh, GPU {data['gpu_energy_joules'] / 3.6e6:.6f} kWh, External {data.get('estimated_external_joules', 0) / 3.6e6:.6f} kWh, Total Carbon {data['carbon_emissions_gco2']:.4f} gCO2.")

    llm_analysis_prompt = (
        "Analyze the following energy consumption report for an AI agent's execution. "
        "Identify the most carbon-intensive steps, explain why they might be consuming so much, "
        "and suggest 2-3 concrete ways a developer could reduce this agent's carbon footprint.\n\n"
        "Energy Report:\n" + "\n".join(report_summary_for_llm) +
        "\n\nProvide actionable advice in bullet points."
    )

    tracker.track_step("llm_report_analysis") # Track the cost of this analysis itself!
    analysis_llm_energy, analysis_tokens, llm_analysis_response = make_llm_call_and_estimate_energy(
        llm_model_to_use, llm_analysis_prompt
    )
    tracker.add_estimated_external_cost(
        analysis_llm_energy, f"LLM Report Analysis Call ({llm_model_to_use}) - {analysis_tokens} tokens"
    )
    tracker.end_step("llm_report_analysis")

    print("\n--- LLM's Insights & Suggestions ---")
    print(llm_analysis_response)

    # Re-run stop to capture the final LLM analysis step
    tracker.stop()
    final_results = tracker.get_results()
    print(f"\nFinal Total Carbon (including LLM Analysis): {final_results['total_carbon_gco2']:.4f} gCO2")

if __name__ == "__main__":
    run_ai_agent_simulation_with_llm()