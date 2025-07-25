from flask import Flask, jsonify, request # Make sure to import 'request'
from flask_cors import CORS
import numpy as np
import time
import os
import random
# Make sure your energy_tracker.py is available
from energy_tracker import EnergyTracker 
import openai
from openai import OpenAI

# --- Configuration (Keep your existing config) ---
# This dictionary is now used as a fallback but the primary model
# is selected from the frontend.
LLM_ENERGY_JOULES_PER_1000_TOKENS = {
    "gpt-4o-2024-05-13": 1.5,
    "gpt-3.5-turbo": 0.5,
    "mixtral-8x7b-instruct-v0.1": 0.8,
    "llama-3-8b-instruct": 0.3
}
# Ensure OPENAI_API_KEY is set in your environment
client = OpenAI() 

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Simulate AI Agent Components (Your existing functions) ---
def make_llm_call_and_estimate_energy(model_name, prompt_content):
    # (Your existing function code... no changes needed here)
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

        energy_joules_per_1000 = LLM_ENERGY_JOULES_PER_1000_TOKENS.get(model_name, 1.0) # Default to 1.0 if model not in dict
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
    # (Your existing function code... no changes needed here)
    start_time = time.time()
    effective_complexity = int(complexity)
    _ = np.random.rand(effective_complexity, 10).dot(np.random.rand(10, 1))
    time.sleep(0.05)
    end_time = time.time()
    print(f"    - Simulated local {description}: {end_time - start_time:.2f}s")


# --- Flask Route for Simulation (UPDATED) ---
@app.route('/run-simulation', methods=['GET'])
def run_simulation_api():
    # --- Get the model from the request query parameter ---
    # Fallback to gpt-4o if no model is specified
    llm_model_to_use = request.args.get('model', 'gpt-4o-2024-05-13')
    print(f"--- Starting simulation for model: {llm_model_to_use} ---")

    tracker = EnergyTracker(project_name="Real_AI_Agent_Carbon_Footprint", region="US")
    tracker.start()

    # (The rest of your simulation logic remains the same, as it now
    # uses the 'llm_model_to_use' variable which is dynamically set)

    # Agent Initialization
    tracker.track_step("agent_startup")
    # ... your existing simulation code ...
    tracker.end_step("agent_startup")

    # Your loop
    NUM_AGENT_TURNS = 3
    LOCAL_PROCESSING_COMPLEXITY = 50000
    for i in range(NUM_AGENT_TURNS):
        tracker.track_step(f"agent_turn_{i+1}")

        # 1. Agent plans/reasons using the SELECTED LLM
        tracker.track_step(f"turn_{i+1}_planning_with_llm")
        agent_query = f"Agent, for turn {i+1}, how would you respond to a user asking about sustainable software? Be concise."
        # The selected model is now passed here
        llm_energy, tokens, llm_response = make_llm_call_and_estimate_energy(llm_model_to_use, agent_query)
        tracker.add_estimated_external_cost(llm_energy, f"LLM Call ({llm_model_to_use}) - Turn {i+1} ({tokens} tokens)")
        simulate_local_processing(LOCAL_PROCESSING_COMPLEXITY, "LLM response parsing")
        tracker.end_step(f"turn_{i+1}_planning_with_llm")
        # ... rest of the loop
        tracker.end_step(f"agent_turn_{i+1}")
        
    # Agent Shutdown
    # ...
    
    # LLM for Post-Analysis
    # (You can choose to use the same model or a fixed powerful one for analysis)
    # ... your existing analysis code using `make_llm_call_and_estimate_energy`
    # ...
    
    # Final results
    tracker.stop()
    final_results = tracker.get_results()

    # Prepare data for JSON response
    # ... (no changes needed in this part) ...
    plottable_steps = {
        k: {
            "duration": v['duration'],
            "cpu_energy_kwh": v['cpu_energy_joules'] / 3.6e6,
            "gpu_energy_kwh": v['gpu_energy_joules'] / 3.6e6,
            "external_energy_kwh": v.get('estimated_external_joules', 0) / 3.6e6,
            "total_carbon_gco2": v['carbon_emissions_gco2']
        }
        for k, v in final_results["steps"].items()
        if not (k == "overall_external_costs" and v.get('estimated_external_joules', 0) == 0)
    }

    response_data = {
        "total_duration_sec": final_results['total_duration_sec'],
        "total_energy_kwh": final_results['total_energy_kwh'],
        "total_carbon_gco2": final_results['total_carbon_gco2'],
        "steps": plottable_steps,
        "llm_analysis": "Analysis from LLM would go here." # Replace with your actual LLM call
    }

    return jsonify(response_data)


if __name__ == '__main__':
    # To run: python app.py
    # This will start the Flask development server on http://127.0.0.1:5000/
    app.run(debug=True)
