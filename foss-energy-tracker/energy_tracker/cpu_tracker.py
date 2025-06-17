# in cpu_tracker.py
import psutil
import time

class CPUTracker:
    def __init__(self):
        # We no longer accumulate total_joules here; main_tracker does that
        self._last_cpu_times = None
        self._last_poll_time = None
        self._is_polling_active = False # New flag

    def start_polling_interval(self):
        """Starts internal tracking for interval-based energy polling."""
        self._last_cpu_times = psutil.cpu_times()
        self._last_poll_time = time.time()
        self._is_polling_active = True
        # print("CPUTracker: Started polling interval.") # For debugging

    def poll_and_get_interval_energy(self) -> float:
        """
        Polls current CPU usage and returns energy consumed since the last call
        to this method (or since start_polling_interval was called).
        Returns energy in Joules.
        """
        if not self._is_polling_active or self._last_cpu_times is None:
            return 0.0

        current_cpu_times = psutil.cpu_times()
        current_time = time.time()

        cpu_time_user_diff = current_cpu_times.user - self._last_cpu_times.user
        cpu_time_system_diff = current_cpu_times.system - self._last_cpu_times.system
        # You can add other cpu_times components if you deem them "active"
        
        cpu_active_time_delta = cpu_time_user_diff + cpu_time_system_diff
        time_interval_duration = current_time - self._last_poll_time

        energy_for_interval_joules = 0.0
        if time_interval_duration > 0 and cpu_active_time_delta > 0:
            # IMPORTANT: This AVG_CPU_POWER_WATTS is a critical placeholder.
            # You should research a typical power consumption for your CPU, or better,
            # use a more sophisticated model or actual power sensor data if available.
            # For a laptop CPU, it might be 15W-45W. For a desktop, 65W-250W+.
            AVG_CPU_POWER_WATTS = 50.0 

            # Calculate utilization over the elapsed wall-clock time
            cpu_utilization_fraction = cpu_active_time_delta / time_interval_duration
            
            # Energy (Joules) = Power (Watts) * Time (Seconds)
            # We scale the average power by the CPU's utilization fraction over the interval.
            energy_for_interval_joules = AVG_CPU_POWER_WATTS * cpu_utilization_fraction * time_interval_duration
            
        # Update for the next poll
        self._last_cpu_times = current_cpu_times
        self._last_poll_time = current_time
        
        return energy_for_interval_joules

    def stop_polling_interval(self):
        """Stops internal tracking and resets."""
        self._last_cpu_times = None
        self._last_poll_time = None
        self._is_polling_active = False
        # print("CPUTracker: Stopped polling interval.") # For debugging

    # These original start/stop methods are now mostly no-ops.
    # The main EnergyTracker calls start_polling_interval and stop_polling_interval.
    def start(self):
        """Placeholder for compatibility; actual polling started by main_tracker's thread."""
        pass 
    def stop(self):
        """Placeholder for compatibility; energy is captured by poll_and_get_interval_energy."""
        return 0.0 # Always return 0 as energy is handled by intervals.