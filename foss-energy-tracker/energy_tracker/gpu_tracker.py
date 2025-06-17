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
    def poll_energy_since_last_call(self):
        if self._last_cpu_times is None: # Not started or already stopped
            return 0.0

        current_cpu_times = psutil.cpu_times()
        current_time = time.time()

        # Calculate CPU time differences
        user_diff = current_cpu_times.user - self._last_cpu_times.user
        system_diff = current_cpu_times.system - self._last_cpu_times.system
        # Consider other fields like `idle` or `iowait` if relevant for more accuracy

        cpu_time_delta = user_diff + system_diff # Total active CPU time
        time_delta = current_time - self._last_poll_time

        # If time_delta is zero, or no active CPU time, return 0
        if time_delta == 0 or cpu_time_delta == 0:
            energy_for_interval = 0.0
        else:
            # Simple approximation: power = (CPU_utilization / 100) * TDP
            # More accurately, you'd need actual power sensors or better models.
            # Assuming average CPU power (e.g., 50W) for demonstration purposes.
            # This is a very rough estimate without actual TDP or power sensors.
            AVG_CPU_POWER_WATTS = 50.0 # This is a placeholder! Research your actual CPU's TDP or average power.
            
            # CPU utilization is total CPU time divided by total elapsed time
            # For this simplified model, we'll just use a direct scaling.
            # A more accurate model would involve actual utilization percentages.
            
            # Let's simplify: assume constant power when CPU is "active" for the delta.
            # A common (but crude) way is to consider average power * time delta
            # or try to estimate based on active time.
            # For a quick MVP, let's use percentage of time active over the interval.
            cpu_utilization_percentage = (cpu_time_delta / time_delta) * 100 if time_delta > 0 else 0
            
            # Energy (Joules) = Power (Watts) * Time (Seconds)
            # If we assume 50W is for 100% utilization, then scale it.
            # This is still a very coarse estimation for actual CPU energy.
            energy_for_interval = (cpu_utilization_percentage / 100.0) * AVG_CPU_POWER_WATTS * time_delta


        # Update for next poll
        self._last_cpu_times = current_cpu_times
        self._last_poll_time = current_time
        
        # self._total_joules += energy_for_interval # No, total is accumulated by main tracker
        return energy_for_interval
