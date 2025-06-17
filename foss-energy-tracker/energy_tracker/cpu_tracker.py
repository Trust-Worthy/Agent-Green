import psutil
import os
import time

class CPUTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self._start_time = None
        self._start_cpu_times = None
        self._cpu_energy_joules = 0

    def start(self):
        self._start_time = time.time()
        self._start_cpu_times = self.process.cpu_times()

    def stop(self):
        if self._start_time is None:
            return 0 # Or raise error

        end_time = time.time()
        end_cpu_times = self.process.cpu_times()

        duration = end_time - self._start_time
        # User CPU time in seconds
        cpu_time_diff = (end_cpu_times.user - self._start_cpu_times.user) + \
                        (end_cpu_times.system - self._start_cpu_times.system)

        # Estimate average CPU power in Watts
        # This is a very rough estimation. Real power draw depends on many factors.
        # A more accurate way would be to read from /sys/class/powercap/intel-rapl/
        # or use specialized tools like Intel Power Gadget or powertop (requires root/privileges)
        # For simplicity, we'll use a constant average power for now, or assume
        # CPU utilization correlates to power (which is a simplification).
        # A typical desktop CPU might consume 20W idle to 100W+ under load.
        # Let's assume an average of 50W for active CPU usage as a placeholder.
        # More sophisticated: try to infer average power from CPU % utilization
        # and a known min/max power for the CPU model (hard to get programmatically).

        # For a truly FOSS approach, you'd integrate with:
        # - Linux: /sys/class/powercap/intel-rapl/ for Intel CPUs
        # - Linux: `powerstat` command line tool (parse its output)
        # - macOS: `powermetrics` (parse output)
        # - Windows: `powercfg` or specific vendor SDKs (e.g., Intel Power Gadget SDK)

        # For now, let's use a very simplified model based on CPU time and an assumed average power.
        # This is a place for significant improvement in your FOSS library!
        # If CPU_POWER_PER_SECOND_JOULES represents average power (e.g., 50 Watts = 50 Joules/sec)
        # then energy = power * time. If we have CPU time in seconds, we can assume a CPU consumes
        # some average power when it's actively working on this process.
        # For a better estimate, you'd need the *actual* power draw of the CPU.
        # Let's use a placeholder average power consumption for the CPU when active.
        # e.g., 60 Watts average active power.
        AVG_CPU_ACTIVE_POWER_WATTS = 60 # Joules per second

        # Total energy consumed by the process's CPU activity
        # This is very approximate. It assumes the process uses 100% of a 60W core when active.
        # Better: (CPU_percent / 100) * total_CPU_power_of_machine_watts * duration
        # but getting total_CPU_power_of_machine_watts programmatically is hard.
        energy_joules = cpu_time_diff * AVG_CPU_ACTIVE_POWER_WATTS

        self._cpu_energy_joules += energy_joules
        self._start_time = None
        self._start_cpu_times = None
        return energy_joules

    def get_total_energy_joules(self):
        return self._cpu_energy_joules

    def get_current_cpu_percent(self):
        return self.process.cpu_percent(interval=None) # Non-blocking call
    
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
