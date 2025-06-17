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
        self._total_cpu_energy_joules = 0
        self._total_gpu_energy_joules = 0
        self._results = {"project_name": project_name, "steps": {}}
        self._current_step = None
        self._poll_thread = None
        self._thread_stop_event = threading.Event()

    def _energy_polling_loop(self):
        """Threaded function to continuously poll energy usage."""
        while not self._thread_stop_event.is_set():
            if self._is_tracking:
                self.cpu_tracker.stop() # Stop and get energy for interval
                self.gpu_tracker.poll_energy() # Update GPU energy for interval

                # Restart for next interval
                self.cpu_tracker.start()

            self._thread_stop_event.wait(self.poll_interval)


    def start(self):
        if self._is_tracking:
            print("Tracker already running.")
            return

        self._is_tracking = True
        self._start_time = time.time()
        self.cpu_tracker.start()
        self.gpu_tracker.start()

        self._poll_thread = threading.Thread(target=self._energy_polling_loop)
        self._poll_thread.daemon = True # Allow main program to exit even if thread is running
        self._poll_thread.start()
        print(f"Energy tracking started for project '{self.project_name}'.")

    def stop(self):
        if not self._is_tracking:
            print("Tracker not running.")
            return

        self._is_tracking = False
        self._thread_stop_event.set()
        if self._poll_thread:
            self._poll_thread.join() # Wait for polling thread to finish

        final_cpu_energy = self.cpu_tracker.stop() # Get final CPU energy
        final_gpu_energy = self.gpu_tracker.stop() # Get final GPU energy

        self._total_cpu_energy_joules += final_cpu_energy
        self._total_gpu_energy_joules += final_gpu_energy

        print(f"Energy tracking stopped for project '{self.project_name}'.")

    def track_step(self, step_name):
        if not self._is_tracking:
            print("Tracker not running. Start it first.")
            return

        if self._current_step:
            self.end_step(self._current_step) # Automatically end previous step

        self._current_step = step_name
        self._results["steps"][step_name] = {
            "start_time": time.time(),
            "cpu_energy_joules": 0,
            "gpu_energy_joules": 0,
            "duration": 0,
            "carbon_emissions_gco2": 0
        }
        # Reset tracker for current step's precise start
        self.cpu_tracker.start()
        self.gpu_tracker.start()
        print(f"Tracking step: {step_name}")

    def end_step(self, step_name):
        if not self._is_tracking or self._current_step != step_name:
            print(f"Not currently tracking step '{step_name}' or tracker not running.")
            return

        end_time = time.time()
        step_data = self._results["steps"][step_name]
        step_data["duration"] = end_time - step_data["start_time"]

        # Stop and accumulate energy for this step
        step_cpu_energy = self.cpu_tracker.stop()
        step_gpu_energy = self.gpu_tracker.stop()
        step_data["cpu_energy_joules"] += step_cpu_energy
        step_data["gpu_energy_joules"] += step_gpu_energy

        # Add to total for overall summary
        self._total_cpu_energy_joules += step_cpu_energy
        self._total_gpu_energy_joules += step_gpu_energy

        # Estimate carbon for this step
        total_step_energy_kwh = (step_data["cpu_energy_joules"] + step_data["gpu_energy_joules"]) / 3.6e6
        step_data["carbon_emissions_gco2"] = self.carbon_estimator.estimate_carbon_emissions(total_step_energy_kwh)

        self._current_step = None
        print(f"Finished tracking step: {step_name}. Duration: {step_data['duration']:.2f}s")


    def get_results(self):
        # Ensure all steps are ended and final energy is tallied
        if self._current_step:
            self.end_step(self._current_step) # End any pending step

        total_energy_kwh = (self._total_cpu_energy_joules + self._total_gpu_energy_joules) / 3.6e6
        total_carbon_gco2 = self.carbon_estimator.estimate_carbon_emissions(total_energy_kwh)

        self._results["total_energy_kwh"] = total_energy_kwh
        self._results["total_carbon_gco2"] = total_carbon_gco2
        self._results["total_duration_sec"] = time.time() - self._start_time if self._start_time else 0

        return self._results

    def __del__(self):
        # Ensure the polling thread is stopped if the object is garbage collected
        if self._poll_thread and self._poll_thread.is_alive():
            self._thread_stop_event.set()
            self._poll_thread.join(timeout=1) # Give it a moment to stop