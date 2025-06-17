import numpy as np
import time
import os
import random
from energy_tracker import EnergyTracker
import openai
from openai import OpenAI
import matplotlib.pyplot as plt # Import matplotlib
import pandas as pd # Optional, but good for data handling before plotting

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

    # --- New: LLM for Post-Analysis of the Report ---
    print("\n--- Asking LLM for Insights on the Energy Report ---")

    # Temporarily get results *before* the final LLM analysis
    # so we can feed them into the LLM prompt.
    # This doesn't stop the tracker, just gets the current state.
    intermediate_results = tracker.get_results()

    report_summary_for_llm = []
    report_summary_for_llm.append(f"Overall Carbon Footprint: {intermediate_results['total_carbon_gco2']:.4f} gCO2 over {intermediate_results['total_duration_sec']:.2f} seconds.")
    report_summary_for_llm.append("Breakdown by step:")
    for step, data in intermediate_results["steps"].items():
        if step == "overall_external_costs" and data['estimated_external_joules'] == 0:
            continue
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

    # --- Finalization and Reporting ---
    # NOW, stop the tracker and get the FINAL results that include the LLM analysis step.
    tracker.stop()
    final_results = tracker.get_results()

    print(f"\n--- AI Agent Carbon Footprint Report (FINAL - {NUM_AGENT_TURNS} Turns) ---")
    print(f"Total Duration: {final_results['total_duration_sec']:.2f} seconds")
    print(f"Total Energy: {final_results['total_energy_kwh']:.6f} kWh")
    print(f"Total Carbon Emissions: {final_results['total_carbon_gco2']:.4f} gCO2")
    print("\n--- Detailed Step Report (FINAL) ---")
    # Pretty print the final results for clarity
    for step, data in final_results["steps"].items():
        if step == "overall_external_costs" and data['estimated_external_joules'] == 0:
            continue
        print(f"\nStep: {step}")
        print(f"  Duration: {data['duration']:.2f} s")
        print(f"  CPU Energy (Measured): {data['cpu_energy_joules'] / 3.6e6:.6f} kWh")
        if data['gpu_energy_joules'] > 0:
            print(f"  GPU Energy (Measured): {data['gpu_energy_joules'] / 3.6e6:.6f} kWh")
        if data.get('estimated_external_joules', 0) > 0:
            print(f"  External Energy (Estimated): {data['estimated_external_joules'] / 3.6e6:.6f} kWh")
            print(f"  External Carbon (Estimated): {data['estimated_external_gco2']:.4f} gCO2")
        print(f"  Total Step Carbon Emissions: {data['carbon_emissions_gco2']:.4f} gCO2")
    print("-" * 40)

    # --- New: Data Visualization ---
    print("\n--- Generating Carbon Footprint Graphs ---")

    # Filter out empty steps or 'overall_external_costs' if it's just a bucket
    plottable_steps = {
        k: v for k, v in final_results["steps"].items()
        if not (k == "overall_external_costs" and v['estimated_external_joules'] == 0)
    }

    step_names = list(plottable_steps.keys())
    total_carbon_gco2 = [data['carbon_emissions_gco2'] for data in plottable_steps.values()]
    cpu_joules = [data['cpu_energy_joules'] for data in plottable_steps.values()]
    gpu_joules = [data['gpu_energy_joules'] for data in plottable_steps.values()]
    external_joules = [data['estimated_external_joules'] for data in plottable_steps.values()]

    # Convert Joules to kWh for better scale on plots if values are small
    # Or keep as Joules if the numbers are large enough. Let's stick with Joules for now,
    # as external estimates are in Joules and can be directly compared.
    # If the numbers are too small to be visible, consider multiplying by a factor
    # or converting to mJ for external if they are tiny compared to local.

    # Graph 1: Total Carbon Emissions per Step
    plt.figure(figsize=(12, 7))
    plt.bar(step_names, total_carbon_gco2, color='skyblue')
    plt.xlabel("Agent Step")
    plt.ylabel("Carbon Emissions (gCO2)")
    plt.title("Carbon Emissions per AI Agent Step")
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig("carbon_emissions_per_step.png")
    # plt.show() # Uncomment to display graph immediately

    # Graph 2: Energy Consumption Breakdown per Step (Stacked Bar Chart)
    # Create a DataFrame for easier plotting with stacked bars
    df_energy = pd.DataFrame({
        'CPU': cpu_joules,
        'GPU': gpu_joules,
        'External': external_joules
    }, index=step_names)

    # Convert Joules to mJ (millijoules) if values are very small for better visibility
    # Or to kJ if they are large
    # Let's check the maximum value to decide scale
    max_energy_joules = max(max(cpu_joules), max(gpu_joules), max(external_joules))
    energy_unit = "Joules (J)"
    scale_factor = 1.0
    if max_energy_joules < 1.0 and max_energy_joules > 0:
        energy_unit = "MilliJoules (mJ)"
        scale_factor = 1000.0
    elif max_energy_joules >= 1000.0:
        energy_unit = "KiloJoules (kJ)"
        scale_factor = 1/1000.0

    df_energy_scaled = df_energy * scale_factor

    ax = df_energy_scaled.plot(kind='bar', stacked=True, figsize=(14, 8), color=['lightcoral', 'lightgreen', 'lightskyblue'])
    plt.xlabel("Agent Step")
    plt.ylabel(f"Energy Consumption ({energy_unit})")
    plt.title("Energy Consumption Breakdown per AI Agent Step")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Energy Source")

    # Add total values on top of stacked bars (optional, but nice)
    for c in ax.containers:
        labels = [f'{v.get_height():.2f}' if v.get_height() > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=7, color='black') # center for stacked, edge for single

    plt.tight_layout()
    plt.savefig("energy_breakdown_per_step.png")
    plt.show() # Show both graphs

    print("\nGraphs saved as 'carbon_emissions_per_step.png' and 'energy_breakdown_per_step.png'")


if __name__ == "__main__":
    run_ai_agent_simulation_with_llm()