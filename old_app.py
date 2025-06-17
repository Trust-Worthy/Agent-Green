from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import time
import os
import random
from energy_tracker import EnergyTracker # Assuming energy_tracker.py is in the same directory
import openai
from openai import OpenAI

# --- Configuration (Keep your existing config) ---
NUM_AGENT_TURNS = 3
LOCAL_PROCESSING_COMPLEXITY = 50000
LLM_ENERGY_JOULES_PER_1000_TOKENS = {
    "gpt-4o-2024-05-13": 1.5,
    "gpt-3.5-turbo": 0.5,
    "mixtral-8x7b-instruct-v0.1": 0.8,
    "llama-3-8b-instruct": 0.3
}
client = OpenAI() # Ensure OPENAI_API_KEY is set in your environment

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Simulate AI Agent Components (Keep these functions as they are) ---
def make_llm_call_and_estimate_energy(model_name, prompt_content):
    # ... (Your existing function code) ...
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
    start_time = time.time()
    effective_complexity = int(complexity)
    _ = np.random.rand(effective_complexity, 10).dot(np.random.rand(10, 1))
    time.sleep(0.05)
    end_time = time.time()
    print(f"    - Simulated local {description}: {end_time - start_time:.2f}s")

def simulate_tool_use(tool_name):
    print(f"    - Agent using tool: {tool_name}")
    time.sleep(0.5)

# --- Flask Route for Simulation ---
@app.route('/run-simulation', methods=['GET'])
def run_simulation_api():
    print("Starting AI Agent carbon footprint simulation with real LLM calls (via API)...")

    tracker = EnergyTracker(project_name="Real_AI_Agent_Carbon_Footprint", region="US")
    tracker.start()

    # Agent Initialization
    tracker.track_step("agent_startup")
    simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY // 2, "startup")
    time.sleep(0.5)
    tracker.end_step("agent_startup")

    llm_model_to_use = "gpt-4o-2024-05-13"

    for i in range(NUM_AGENT_TURNS):
        tracker.track_step(f"agent_turn_{i+1}")

        # 1. Agent plans/reasons using an LLM
        tracker.track_step(f"turn_{i+1}_planning_with_llm")
        agent_query = f"Agent, for turn {i+1}, how would you respond to a user asking about sustainable software? Be concise."
        llm_energy, tokens, llm_response = make_llm_call_and_estimate_energy(llm_model_to_use, agent_query)
        tracker.add_estimated_external_cost(llm_energy, f"LLM Call ({llm_model_to_use}) - Turn {i+1} ({tokens} tokens)")
        simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY, "LLM response parsing")
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

    # Agent Shutdown
    tracker.track_step("agent_shutdown")
    simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY // 4, "shutdown")
    time.sleep(0.2)
    tracker.end_step("agent_shutdown")

    # LLM for Post-Analysis
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

    tracker.track_step("llm_report_analysis")
    analysis_llm_energy, analysis_tokens, llm_analysis_response = make_llm_call_and_estimate_energy(
        llm_model_to_use, llm_analysis_prompt
    )
    tracker.add_estimated_external_cost(
        analysis_llm_energy, f"LLM Report Analysis Call ({llm_model_to_use}) - {analysis_tokens} tokens"
    )
    tracker.end_step("llm_report_analysis")

    # Final results
    tracker.stop()
    final_results = tracker.get_results()

    # Prepare data for JSON response
    plottable_steps = {
        k: {
            "duration": v['duration'],
            "cpu_energy_kwh": v['cpu_energy_joules'] / 3.6e6,
            "gpu_energy_kwh": v['gpu_energy_joules'] / 3.6e6,
            "external_energy_kwh": v.get('estimated_external_joules', 0) / 3.6e6,
            "total_carbon_gco2": v['carbon_emissions_gco2']
        }
        for k, v in final_results["steps"].items()
        if not (k == "overall_external_costs" and v['estimated_external_joules'] == 0)
    }

    response_data = {
        "total_duration_sec": final_results['total_duration_sec'],
        "total_energy_kwh": final_results['total_energy_kwh'],
        "total_carbon_gco2": final_results['total_carbon_gco2'],
        "steps": plottable_steps,
        "llm_analysis": llm_analysis_response
    }

    return jsonify(response_data)

if __name__ == '__main__':
    # To run: python app.py
    # This will start the Flask development server on http://127.0.0.1:5000/
    app.run(debug=True) # debug=True allows auto-reloading and better error messages