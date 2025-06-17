# 🌍 FOSS Energy Tracker: Basic Usage 🌿

The `foss-energy-tracker` is a Free and Open Source Software (FOSS) tool designed to help developers quickly measure the energy consumption and estimated carbon emissions of their Python scripts. It's particularly useful for understanding the impact of computationally intensive tasks like those in AI/ML pipelines.

## ✨ Features

* **Tracks CPU Energy:** Measures the energy used by your Python process on the CPU.
* **Tracks NVIDIA GPU Energy:** Measures energy consumed by NVIDIA GPUs during script execution (requires `nvidia-smi`).
* **Estimates Carbon Emissions:** Converts energy consumption into estimated grams of CO₂ based on a configurable region's energy mix.
* **Step-by-Step Reporting:** Allows you to segment your code and get energy reports for specific parts of your script.

## 🚀 Quick Start: Installation & Usage

### 1. Installation

First, make sure you have a Conda environment set up and active. If not, create one:

```bash
conda create --name my_tracker_env python=3.9
conda activate my_tracker_env
```

Then, install the necessary dependencies and the `foss-energy-tracker` library in editable mode (so you can easily make changes):

```bash
# Navigate to the foss-energy-tracker directory where setup.py is located
cd /path/to/your/foss-energy-tracker

# Install core dependencies (if not already present)
conda install numpy scikit-learn psutil

# Install the tracker in editable mode
pip install -e .
```

### 2. Usage in Your Python Script

Import `EnergyTracker` and use `tracker.start()`, `tracker.track_step()`, `tracker.end_step()`, and `tracker.stop()` around the code you want to measure.

Here's an example:

```python
# your_script_to_measure.py
from energy_tracker import EnergyTracker

# Initialize the tracker (you can specify your region, e.g., "US", "EU", "DE")
tracker = EnergyTracker(project_name="My Measured Code", region="US")
tracker.start() # Begin overall tracking

# --- Your computationally heavy code starts here ---

tracker.track_step("computation_phase_1")
print("Running phase 1...")
import time; time.sleep(3) # Simulate some work
tracker.end_step("computation_phase_1")

tracker.track_step("computation_phase_2")
print("Running phase 2...")
time.sleep(5) # Simulate more work
tracker.end_step("computation_phase_2")

# --- Your computationally heavy code ends here ---

tracker.stop() # End overall tracking and gather results
results = tracker.get_results()

print("\n--- Energy Report ---")
for step, data in results["steps"].items():
    print(f"Step: {step}")
    print(f"  Duration: {data['duration']:.2f} seconds")
    # Convert Joules to kWh for easier reading
    print(f"  CPU Energy: {data['cpu_energy_joules'] / 3.6e6:.6f} kWh")
    if data['gpu_energy_joules'] > 0: # Only show GPU if it was active
        print(f"  GPU Energy: {data['gpu_energy_joules'] / 3.6e6:.6f} kWh")
    print(f"  Carbon Emissions: {data['carbon_emissions_gco2']:.4f} gCO2")

print(f"\nTotal Run Energy: {results['total_energy_kwh']:.6f} kWh")
print(f"Total Carbon Emissions: {results['total_carbon_gco2']:.4f} gCO2")
```

Save the above code as `your_script_to_measure.py` (or use your `tracked_heavy_computation.py` directly).

### 3. Run Your Script

Make sure your Conda environment is active, then execute your script:

```bash
# Navigate to the directory containing your_script_to_measure.py
cd /path/to/your/scripts

python your_script_to_measure.py
```

You'll see output showing the progress of your script and, at the end, a summary of its energy consumption and carbon footprint.

---