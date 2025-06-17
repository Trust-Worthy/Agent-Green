import subprocess
import json
import time

class GPUTracker:
    def __init__(self):
        self._start_time = None
        self._gpu_energy_joules = 0
        self._is_nvidia = self._check_nvidia_smi()

    def _check_nvidia_smi(self):
        try:
            subprocess.run(["nvidia-smi", "-q"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def start(self):
        self._start_time = time.time()
        self._last_poll_time = self._start_time

    def _get_gpu_power(self):
        """Polls NVIDIA GPU power draw."""
        if not self._is_nvidia:
            return 0 # No NVIDIA GPU found

        try:
            # Query power.draw and units (mW)
            cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
            output = subprocess.check_output(cmd).decode('utf-8').strip()
            power_mw = float(output.split('\n')[0])
            return power_mw / 1000.0 # Convert mW to Watts
        except Exception as e:
            # print(f"Error getting GPU power: {e}")
            return 0

    def stop(self):
        if self._start_time is None:
            return 0

        current_time = time.time()
        if self._is_nvidia:
            # Approximate energy by integrating power over time
            # For more accuracy, you'd poll power continuously or at frequent intervals
            # and sum (power * delta_time)
            avg_power = self._get_gpu_power() # Get final power reading
            duration = current_time - self._last_poll_time # Duration since last poll/start
            self._gpu_energy_joules += avg_power * duration

        self._start_time = None
        self._last_poll_time = None
        return self._gpu_energy_joules

    def poll_energy(self):
        """Polls GPU power and adds to total energy."""
        if self._start_time is None or not self._is_nvidia:
            return

        current_time = time.time()
        power = self._get_gpu_power() # Watts
        duration_since_last_poll = current_time - self._last_poll_time
        energy_added = power * duration_since_last_poll
        self._gpu_energy_joules += energy_added
        self._last_poll_time = current_time

    def get_total_energy_joules(self):
        return self._gpu_energy_joules