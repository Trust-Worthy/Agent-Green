import time
import threading
from .cpu_tracker import CPUTracker
from .gpu_tracker import GPUTracker
from .carbon_estimator import CarbonEstimator

class EnergyTracker:
    def __init__(self, project_name="Untitled", region="DEFAULT", poll_interval=1.0):
        self.project_name = project_name
        self.cpu_tracker = CPUTracker()
        self.gpu_tracker = GPUTracker()
        self.carbon_estimator = CarbonEstimator(region=region)
        self.poll_interval = poll_interval # Seconds between energy polls

        self._is_tracking = False
        self._start_time = None
        
        self._overall_measured_cpu_joules = 0.0
        self._overall_measured_gpu_joules = 0.0
        self._overall_estimated_external_joules = 0.0
        self._overall_estimated_external_gco2 = 0.0

        self._step_data = {} # Stores all details for each named step
        # --- FIX: Use a stack for managing nested steps ---
        self._step_stack = [] # Stores (step_name, start_time, cpu_start_energy, gpu_start_energy) tuples

        self._poll_thread = None
        self._thread_stop_event = threading.Event()

    def _energy_polling_loop(self):
        """Threaded function to continuously poll energy usage."""
        # --- FIX: Track energy consistently and apply to the currently active step ---
        self.cpu_tracker.start() # Start CPU tracker for continuous polling
        self.gpu_tracker.start() # Start GPU tracker for continuous polling

        while not self._thread_stop_event.is_set():
            if self._is_tracking:
                # Get polled energy for the interval
                interval_cpu_energy = self.cpu_tracker.poll_energy_since_last_call() # Assuming this method exists and returns interval data
                interval_gpu_energy = self.gpu_tracker.poll_energy_since_last_call() # Assuming this method exists and returns interval data

                # Accumulate polled energy to the active step's data
                if self._step_stack:
                    current_step_name = self._step_stack[-1][0] # Get name of top-most active step
                    if current_step_name in self._step_data:
                        self._step_data[current_step_name]['cpu_energy_joules'] += interval_cpu_energy
                        self._step_data[current_step_name]['gpu_energy_joules'] += interval_gpu_energy
                        self._overall_measured_cpu_joules += interval_cpu_energy
                        self._overall_measured_gpu_joules += interval_gpu_energy
                else:
                    # If no step is active but tracking is on, accumulate to overall totals (e.g., between steps)
                    self._overall_measured_cpu_joules += interval_cpu_energy
                    self._overall_measured_gpu_joules += interval_gpu_energy

            self._thread_stop_event.wait(self.poll_interval)


    def start(self):
        if self._is_tracking:
            print("Tracker already running.")
            return

        self._is_tracking = True
        self._start_time = time.time()
        
        # --- Reset all totals and step data on new start ---
        self._overall_measured_cpu_joules = 0.0
        self._overall_measured_gpu_joules = 0.0
        self._overall_estimated_external_joules = 0.0
        self._overall_estimated_external_gco2 = 0.0
        self._step_data = {}
        self._step_stack = [] # Clear step stack

        self._thread_stop_event.clear() # Clear event for new start
        self._poll_thread = threading.Thread(target=self._energy_polling_loop)
        self._poll_thread.daemon = True 
        self._poll_thread.start()
        print(f"Energy tracking started for project '{self.project_name}'.")

    def stop(self):
        if not self._is_tracking:
            print("Tracker not running.")
            return

        # --- FIX: End all active steps before stopping overall ---
        while self._step_stack:
            # Pop the current step and end it
            step_name, _, _, _ = self._step_stack[-1]
            print(f"Warning: Ending pending step '{step_name}' before overall stop.")
            self.end_step(step_name)

        self._is_tracking = False
        self._thread_stop_event.set() # Signal polling thread to stop
        if self._poll_thread and self._poll_thread.is_alive():
            # Give thread a bit more time to process final polls and exit
            self._poll_thread.join(timeout=self.poll_interval * 3 + 1) 
            if self._poll_thread.is_alive():
                print("Warning: Polling thread did not terminate gracefully.")
        
        # --- FIX: Stop CPU/GPU trackers to get any final accumulated energy from the very last moments ---
        # This is important if poll_energy_since_last_call accumulates between calls.
        final_cpu_energy_remaining = self.cpu_tracker.stop()
        final_gpu_energy_remaining = self.gpu_tracker.stop()
        self._overall_measured_cpu_joules += final_cpu_energy_remaining
        self._overall_measured_gpu_joules += final_gpu_energy_remaining

        print(f"Energy tracking stopped for project '{self.project_name}'.")

    def track_step(self, step_name):
        if not self._is_tracking:
            print("Tracker not running. Start it first.")
            return

        # If a step with this name already exists, it might be a re-start or error.
        # For nested tracking, a name on the stack means it's still active.
        if step_name in self._step_data and not any(s[0] == step_name for s in self._step_stack):
             print(f"Warning: Step '{step_name}' already exists in results but not on stack. Overwriting.")
        
        # Get current accumulated energy *before* this step starts, so we can attribute energy correctly
        # from the polling loop.
        current_cpu_total_at_start = self._overall_measured_cpu_joules
        current_gpu_total_at_start = self._overall_measured_gpu_joules

        # Push to stack
        self._step_stack.append((step_name, time.time(), current_cpu_total_at_start, current_gpu_total_at_start))

        # Initialize step data if it's the first time we see this step name
        if step_name not in self._step_data:
            self._step_data[step_name] = {
                "duration": 0.0,
                "cpu_energy_joules": 0.0,
                "gpu_energy_joules": 0.0,
                "estimated_external_joules": 0.0,
                "estimated_external_gco2": 0.0,
                "carbon_emissions_gco2": 0.0
            }
        print(f"Tracking step: {step_name}")

    def end_step(self, step_name):
        if not self._is_tracking:
            print(f"Tracker not running. Cannot end step '{step_name}'.")
            return
        
        if not self._step_stack:
            print(f"No active steps to end. Cannot end step '{step_name}'.")
            return

        if self._step_stack[-1][0] != step_name:
            # This indicates an attempt to end a step that's not the innermost one.
            # For strict nesting, this is an error. For a hackathon MVP, we can warn
            # or try to find it in the stack. For simplicity, let's assume strict LIFO.
            print(f"Error: Attempted to end '{step_name}', but '{self._step_stack[-1][0]}' is the current active step.")
            return

        # Pop the step from the stack
        popped_step_name, step_start_time, cpu_start_energy, gpu_start_energy = self._step_stack.pop()

        end_time = time.time()
        step_data = self._step_data[popped_step_name]
        step_data["duration"] += (end_time - step_start_time) # Use += for duration if steps can be restarted


        # The energy for this step is the difference between overall accumulated energy
        # at its end and when it started.
        # This handles energy accumulated during polling directly into the overall totals.
        step_data["cpu_energy_joules"] += (self._overall_measured_cpu_joules - cpu_start_energy)
        step_data["gpu_energy_joules"] += (self._overall_measured_gpu_joules - gpu_start_energy) # Accumulate GPU energy the same way


        # Estimate carbon for measured portion of this step
        measured_step_energy_kwh = (step_data["cpu_energy_joules"] + step_data["gpu_energy_joules"]) / 3.6e6
        measured_step_carbon_gco2 = self.carbon_estimator.estimate_carbon_emissions(measured_step_energy_kwh)
        
        # Total carbon for this step (measured + estimated external)
        step_data["carbon_emissions_gco2"] = measured_step_carbon_gco2 + step_data["estimated_external_gco2"]

        print(f"Finished tracking step: {popped_step_name}. Duration: {step_data['duration']:.2f}s")

    def add_estimated_external_cost(self, energy_joules: float, description: str, carbon_gco2: float = None):
        """
        Adds an estimated energy and carbon cost for an external operation (e.g., LLM API call).
        This does not involve live measurement but accounts for known/estimated costs.
        """
        if not self._is_tracking:
            print(f"Warning: Attempted to add external cost for '{description}' but tracker is not active.")
            return

        energy_kwh = energy_joules / 3.6e6

        if carbon_gco2 is None:
            carbon_gco2 = self._calculate_carbon(energy_kwh)

        # --- FIX: Add external cost to the currently active (top-most) step ---
        if self._step_stack:
            current_step_name = self._step_stack[-1][0] # Get name of top-most active step
            self._step_data[current_step_name]['estimated_external_joules'] += energy_joules
            self._step_data[current_step_name]['estimated_external_gco2'] += carbon_gco2
            print(f"  Added estimated external cost to '{current_step_name}': {energy_joules:.2f}J, {carbon_gco2:.4f}gCO2 for '{description}'")
        else:
            # If no step is active, add to a general 'overall_external_costs' bucket
            # This is a special case; ideally, external costs are tied to a step.
            # We ensure this bucket exists and is not treated as a regular tracked step.
            if 'overall_external_costs' not in self._step_data:
                self._step_data['overall_external_costs'] = {
                    'duration': 0.0, 'cpu_energy_joules': 0.0, 'gpu_energy_joules': 0.0,
                    'estimated_external_joules': 0.0, 'estimated_external_gco2': 0.0,
                    'carbon_emissions_gco2': 0.0
                }
            self._step_data['overall_external_costs']['estimated_external_joules'] += energy_joules
            self._step_data['overall_external_costs']['estimated_external_gco2'] += carbon_gco2
            print(f"  Added estimated external cost (no active step): {energy_joules:.2f}J, {carbon_gco2:.4f}gCO2 for '{description}'")

        # Always accumulate to overall estimated totals regardless of specific step
        self._overall_estimated_external_joules += energy_joules
        self._overall_estimated_external_gco2 += carbon_gco2


    def get_results(self):
        """
        Retrieves the aggregated energy and carbon results.
        Ensures all steps are finalized before returning.
        """
        # Ensure any currently active steps are ended before finalizing results
        while self._step_stack:
            step_name, _, _, _ = self._step_stack[-1]
            print(f"Warning: Ending pending step '{step_name}' before getting final results.")
            self.end_step(step_name)

        # Calculate overall totals for the final report
        final_total_measured_energy_kwh = (self._overall_measured_cpu_joules + self._overall_measured_gpu_joules) / 3.6e6
        final_total_measured_carbon_gco2 = self.carbon_estimator.estimate_carbon_emissions(final_total_measured_energy_kwh)
        
        final_total_energy_kwh = final_total_measured_energy_kwh + (self._overall_estimated_external_joules / 3.6e6)
        final_total_carbon_gco2 = final_total_measured_carbon_gco2 + self._overall_estimated_external_gco2

        results = {
            "project_name": self.project_name,
            "total_duration_sec": time.time() - self._start_time if self._start_time else 0,
            "total_energy_kwh": final_total_energy_kwh,
            "total_carbon_gco2": final_total_carbon_gco2,
            "steps": {}
        }

        # Process individual step metrics for the final report
        for step_name, metrics in self._step_data.items():
            # Skip the 'overall_external_costs' bucket if it's just for un-stepped external costs
            if step_name == "overall_external_costs" and metrics['cpu_energy_joules'] == 0 and metrics['gpu_energy_joules'] == 0 and metrics['duration'] == 0:
                if metrics['estimated_external_joules'] == 0: # Only skip if truly empty, otherwise might contain real data
                    continue 
                # If it has estimated_external_joules but no measured, include it as a special "unattributed external" step

            step_duration = metrics.get('duration', 0.0)
            step_cpu_energy_joules = metrics.get('cpu_energy_joules', 0.0)
            step_gpu_energy_joules = metrics.get('gpu_energy_joules', 0.0)
            step_estimated_external_joules = metrics.get('estimated_external_joules', 0.0)
            step_estimated_external_gco2 = metrics.get('estimated_external_gco2', 0.0)

            total_step_joules = step_cpu_energy_joules + step_gpu_energy_joules + step_estimated_external_joules
            total_step_kwh = total_step_joules / 3.6e6

            # Carbon for this step
            # Calculate carbon for measured portion (CPU/GPU)
            measured_carbon_for_this_step = self.carbon_estimator.estimate_carbon_emissions(
                (step_cpu_energy_joules + step_gpu_energy_joules) / 3.6e6
            )
            # Sum with already-estimated external carbon for this step
            step_total_carbon_gco2 = measured_carbon_for_this_step + step_estimated_external_gco2

            results["steps"][step_name] = {
                "duration": step_duration,
                "cpu_energy_joules": step_cpu_energy_joules,
                "gpu_energy_joules": step_gpu_energy_joules,
                "estimated_external_joules": step_estimated_external_joules,
                "estimated_external_gco2": step_estimated_external_gco2,
                "total_step_energy_kwh": total_step_kwh, 
                "carbon_emissions_gco2": step_total_carbon_gco2
            }
        
        return results
    
    def _calculate_carbon(self, energy_kwh: float) -> float:
        """Helper method to estimate carbon for internal use, especially for external costs."""
        return self.carbon_estimator.estimate_carbon_emissions(energy_kwh)


    def __del__(self):
        # Ensure the polling thread is stopped if the object is garbage collected
        if self._poll_thread and self._poll_thread.is_alive():
            self._thread_stop_event.set()
            self._poll_thread.join(timeout=1)