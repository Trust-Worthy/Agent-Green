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
        
        # --- NEW/FIXED: Initialize overall energy totals ---
        self._overall_measured_cpu_joules = 0.0
        self._overall_measured_gpu_joules = 0.0
        self._overall_estimated_external_joules = 0.0
        self._overall_estimated_external_gco2 = 0.0 # To track explicit external carbon

        # --- FIX: Use a single dictionary for step data ---
        self._step_data = {} # This will store all details for each named step
        self._current_step_name = None # Renamed for clarity to avoid conflict with `_current_step` in `add_estimated_external_cost` original placeholder

        self._poll_thread = None
        self._thread_stop_event = threading.Event()

    def _energy_polling_loop(self):
        """Threaded function to continuously poll energy usage."""
        last_poll_time = time.time()
        while not self._thread_stop_event.is_set():
            current_time = time.time()
            elapsed_time_since_last_poll = current_time - last_poll_time
            last_poll_time = current_time

            if self._is_tracking:
                # Get energy for this interval from CPU and GPU
                # Assumption: CPUTracker.get_interval_energy() and GPUTracker.get_interval_energy()
                # return energy consumed since their last call or start/poll.
                # If your CPU/GPU trackers work differently, adjust this.
                
                # For CPU, let's assume CPUTracker.poll_energy() returns energy for the interval
                # And that CPUTracker.start() effectively resets its internal counter for a new interval
                # Based on your existing code, CPUTracker's start/stop seems to manage accumulation.
                # So the polling loop will need to capture energy and assign it to the active step.
                
                # We need to stop the CPU tracker to get its accumulated energy, then restart it.
                # This seems to be the intended way based on your `stop` in `end_step`.
                interval_cpu_energy = self.cpu_tracker.stop() 
                self.cpu_tracker.start() # Restart for the next interval

                # For GPU, assume poll_energy() directly updates internal state which stop() then reads.
                self.gpu_tracker.poll_energy() 
                # GPU energy will be accumulated when gpu_tracker.stop() is called in end_step or stop.

                if self._current_step_name:
                    # Accumulate polled CPU energy to the active step's data
                    self._step_data[self._current_step_name]['cpu_energy_joules'] += interval_cpu_energy
                    self._overall_measured_cpu_joules += interval_cpu_energy
                    
                    # GPU energy is often tracked differently; it might be accumulated only on stop.
                    # If GPUTracker.poll_energy() also returns an interval value, accumulate it similarly.
                    # For now, relying on GPUTracker.stop() for the final sum in end_step/stop.

            # Wait for the next poll interval, or stop if event is set
            self._thread_stop_event.wait(self.poll_interval)


    def start(self):
        if self._is_tracking:
            print("Tracker already running.")
            return

        self._is_tracking = True
        self._start_time = time.time()
        self.cpu_tracker.start() # Start CPU tracking at overall start
        self.gpu_tracker.start() # Start GPU tracking at overall start

        # --- Reset totals on new start ---
        self._overall_measured_cpu_joules = 0.0
        self._overall_measured_gpu_joules = 0.0
        self._overall_estimated_external_joules = 0.0
        self._overall_estimated_external_gco2 = 0.0
        self._step_data = {} # Clear previous step data

        self._thread_stop_event.clear() # Clear event for new start
        self._poll_thread = threading.Thread(target=self._energy_polling_loop)
        self._poll_thread.daemon = True # Allow main program to exit even if thread is running
        self._poll_thread.start()
        print(f"Energy tracking started for project '{self.project_name}'.")

    def stop(self):
        if not self._is_tracking:
            print("Tracker not running.")
            return

        # End any pending step before stopping overall
        if self._current_step_name:
            print(f"Warning: Ending pending step '{self._current_step_name}' before overall stop.")
            self.end_step(self._current_step_name)

        self._is_tracking = False
        self._thread_stop_event.set() # Signal polling thread to stop
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=self.poll_interval * 2 + 1) # Give thread time to clean up

        # Get final accumulated energy from the trackers
        final_cpu_energy = self.cpu_tracker.stop() 
        final_gpu_energy = self.gpu_tracker.stop() 
        
        # Add any remaining energy from the very last interval after thread stopped
        # This prevents loss of energy if stop() is called immediately after a poll cycle ends
        self._overall_measured_cpu_joules += final_cpu_energy
        self._overall_measured_gpu_joules += final_gpu_energy

        print(f"Energy tracking stopped for project '{self.project_name}'.")

    def track_step(self, step_name):
        if not self._is_tracking:
            print("Tracker not running. Start it first.")
            return

        if self._current_step_name:
            self.end_step(self._current_step_name) # Automatically end previous step

        self._current_step_name = step_name
        self._step_data[step_name] = { # Use _step_data
            "start_time": time.time(),
            "cpu_energy_joules": 0.0,
            "gpu_energy_joules": 0.0,
            "duration": 0.0,
            "estimated_external_joules": 0.0, # Initialize here
            "estimated_external_gco2": 0.0,   # Initialize here
            "carbon_emissions_gco2": 0.0      # Initialize here
        }
        # Restart CPU/GPU trackers for this specific step's measurement
        self.cpu_tracker.start() 
        self.gpu_tracker.start() 
        print(f"Tracking step: {step_name}")

    def end_step(self, step_name):
        if not self._is_tracking or self._current_step_name != step_name:
            print(f"Not currently tracking step '{step_name}' or tracker not running. Current step: {self._current_step_name}")
            return

        end_time = time.time()
        step_data = self._step_data[step_name] # Use _step_data
        step_data["duration"] = end_time - step_data["start_time"]

        # Stop and accumulate energy for this step from trackers
        # These stop calls provide the energy accumulated since their last start/reset
        step_cpu_energy_measured = self.cpu_tracker.stop()
        step_gpu_energy_measured = self.gpu_tracker.stop()
        
        step_data["cpu_energy_joules"] += step_cpu_energy_measured
        step_data["gpu_energy_joules"] += step_gpu_energy_measured

        # Add measured energy to overall totals
        self._overall_measured_cpu_joules += step_cpu_energy_measured
        self._overall_measured_gpu_joules += step_gpu_energy_measured

        # Estimate carbon for measured portion of this step
        measured_step_energy_kwh = (step_data["cpu_energy_joules"] + step_data["gpu_energy_joules"]) / 3.6e6
        measured_step_carbon_gco2 = self.carbon_estimator.estimate_carbon_emissions(measured_step_energy_kwh)
        
        # Total carbon for this step (measured + estimated external)
        step_data["carbon_emissions_gco2"] = measured_step_carbon_gco2 + step_data["estimated_external_gco2"]

        self._current_step_name = None # Reset current step
        
        # Restart trackers immediately for the next possible step or overall tracking
        self.cpu_tracker.start()
        self.gpu_tracker.start()
        
        print(f"Finished tracking step: {step_name}. Duration: {step_data['duration']:.2f}s")

    def add_estimated_external_cost(self, energy_joules: float, description: str, carbon_gco2: float = None):
        """
        Adds an estimated energy and carbon cost for an external operation (e.g., LLM API call).
        This does not involve live measurement but accounts for known/estimated costs.
        """
        if not self._is_tracking: # Use _is_tracking for consistency
            print(f"Warning: Attempted to add external cost for '{description}' but tracker is not active.")
            return

        energy_kwh = energy_joules / 3.6e6

        # Estimate carbon if not provided, using the tracker's region
        if carbon_gco2 is None:
            carbon_gco2 = self.carbon_estimator.estimate_carbon_emissions(energy_kwh)

        # Add to the currently active step, or to a general 'external_costs' bucket
        current_step_to_add_to = self._current_step_name if self._current_step_name else "overall_external_costs"

        if current_step_to_add_to not in self._step_data:
            self._step_data[current_step_to_add_to] = {
                'start_time': time.time(), # Use current time for non-tracked "step"
                'duration': 0.0,
                'cpu_energy_joules': 0.0,
                'gpu_energy_joules': 0.0,
                'estimated_external_joules': 0.0,
                'estimated_external_gco2': 0.0,
                'carbon_emissions_gco2': 0.0 # Will be calculated in get_results or if it becomes an actual step
            }
            # Note: For 'overall_external_costs', duration, cpu/gpu energy will remain 0.

        self._step_data[current_step_to_add_to]['estimated_external_joules'] += energy_joules
        self._step_data[current_step_to_add_to]['estimated_external_gco2'] += carbon_gco2
        
        # Accumulate to overall estimated totals
        self._overall_estimated_external_joules += energy_joules
        self._overall_estimated_external_gco2 += carbon_gco2

        print(f"  Added estimated external cost to '{current_step_to_add_to}': {energy_joules:.2f}J, {carbon_gco2:.4f}gCO2 for '{description}'")


    def get_results(self):
        """
        Retrieves the aggregated energy and carbon results.
        Ensures all steps are finalized before returning.
        """
        # Ensure any currently active step is ended before finalizing results
        if self._current_step_name:
            print(f"Warning: Ending pending step '{self._current_step_name}' before getting results.")
            self.end_step(self._current_step_name)

        # Calculate overall totals for the final report
        # These overall totals should already be accumulated correctly by stop() and add_estimated_external_cost()
        final_total_energy_joules = self._overall_measured_cpu_joules + self._overall_measured_gpu_joules + self._overall_estimated_external_joules
        final_total_energy_kwh = final_total_energy_joules / 3.6e6
        
        # Overall carbon should be sum of measured carbon + estimated external carbon
        # Measured carbon: self.carbon_estimator.estimate_carbon_emissions((self._overall_measured_cpu_joules + self._overall_measured_gpu_joules) / 3.6e6)
        # Estimated external carbon: self._overall_estimated_external_gco2
        final_total_carbon_gco2 = self.carbon_estimator.estimate_carbon_emissions(
            (self._overall_measured_cpu_joules + self._overall_measured_gpu_joules) / 3.6e6
        ) + self._overall_estimated_external_gco2


        # Prepare the final results dictionary
        results = {
            "project_name": self.project_name,
            "total_duration_sec": time.time() - self._start_time if self._start_time else 0,
            "total_energy_kwh": final_total_energy_kwh,
            "total_carbon_gco2": final_total_carbon_gco2,
            "steps": {}
        }

        # Process individual step metrics for the final report
        for step_name, metrics in self._step_data.items(): # Correctly iterate over self._step_data
            step_duration = metrics.get('duration', 0.0)
            step_cpu_energy_joules = metrics.get('cpu_energy_joules', 0.0)
            step_gpu_energy_joules = metrics.get('gpu_energy_joules', 0.0)
            step_estimated_external_joules = metrics.get('estimated_external_joules', 0.0)
            step_estimated_external_gco2 = metrics.get('estimated_external_gco2', 0.0)

            # Total energy for this specific step (measured + estimated external)
            total_step_joules = step_cpu_energy_joules + step_gpu_energy_joules + step_estimated_external_joules
            total_step_kwh = total_step_joules / 3.6e6

            # Carbon for this step is sum of carbon from measured energy and explicitly added external carbon
            measured_carbon_for_this_step = self.carbon_estimator.estimate_carbon_emissions(
                (step_cpu_energy_joules + step_gpu_energy_joules) / 3.6e6
            )
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
    
    # Missing _calculate_carbon method. Add it if it's not inherited/defined elsewhere.
    def _calculate_carbon(self, energy_kwh: float) -> float:
        """Helper method to estimate carbon for internal use, especially for external costs."""
        return self.carbon_estimator.estimate_carbon_emissions(energy_kwh)


    def __del__(self):
        # Ensure the polling thread is stopped if the object is garbage collected
        if self._poll_thread and self._poll_thread.is_alive():
            self._thread_stop_event.set()
            self._poll_thread.join(timeout=1) # Give it a moment to stop