# in gpu_tracker.py
import subprocess
import json
import time

class GPUTracker:
    def __init__(self):
        self._last_poll_time = None
        self._last_power_draw_watts = 0.0 # Store last measured power draw
        self._is_polling_active = False # New flag

    def start_polling_interval(self):
        """Starts internal tracking for interval-based energy polling."""
        self._last_poll_time = time.time()
        self._is_polling_active = True
        self._last_power_draw_watts = self._get_current_gpu_power_draw() # Get initial power draw
        # print("GPUTracker: Started polling interval.") # For debugging

    def _get_current_gpu_power_draw(self) -> float:
        """
        Queries nvidia-smi for current power draw in Watts.
        Returns 0.0 if no GPU or error.
        """
        try:
            # -q query, -d display, -u unit, -x xml
            smi_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            power_draw = float(smi_output.strip())
            return power_draw
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            # print("Warning: NVIDIA GPU not found or nvidia-smi failed.")
            return 0.0 # No GPU or error

    def poll_and_get_interval_energy(self) -> float:
        """
        Polls current GPU power draw and estimates energy consumed since the last call
        to this method (or since start_polling_interval was called).
        Returns energy in Joules.
        """
        if not self._is_polling_active or self._last_poll_time is None:
            return 0.0

        current_time = time.time()
        time_interval_duration = current_time - self._last_poll_time
        
        current_power_draw_watts = self._get_current_gpu_power_draw()

        energy_for_interval_joules = 0.0
        if time_interval_duration > 0:
            # Use the average power during the interval for energy calculation
            avg_power_watts = (self._last_power_draw_watts + current_power_draw_watts) / 2
            energy_for_interval_joules = avg_power_watts * time_interval_duration

        # Update for the next poll
        self._last_poll_time = current_time
        self._last_power_draw_watts = current_power_draw_watts

        return energy_for_interval_joules

    def stop_polling_interval(self):
        """Stops internal tracking and resets."""
        self._last_poll_time = None
        self._last_power_draw_watts = 0.0
        self._is_polling_active = False
        # print("GPUTracker: Stopped polling interval.") # For debugging

    # These original start/stop methods are now mostly no-ops.
    def start(self):
        """Placeholder for compatibility; actual polling started by main_tracker's thread."""
        pass
    def stop(self):
        """Placeholder for compatibility; energy is captured by poll_and_get_interval_energy."""
        return 0.0 # Always return 0 as energy is handled by intervals.