import os

class CarbonEstimator:
    # Source for average grid intensity (gCO2/kWh):
    # This data would ideally come from a dynamic API or a well-maintained database
    # like electricitymap.org or regional government data.
    # For a FOSS project, you could provide a static mapping or allow users to input.
    # These are illustrative global averages; real-world values vary wildly by region/time.
    # A better approach would be to use data from: https://github.com/Green-Software-Foundation/sci-guidance
    # or a service like electricitymap.org (though direct API access might not be FOSS friendly).
    # For now, we'll use some placeholder values.

    # Example average grid intensities (gCO2/kWh)
    # Source: Varies, these are illustrative and need to be kept up-to-date
    # Consider using https://app.electricitymaps.com/ or similar for real-time data or better static data
    REGION_CARBON_INTENSITIES = {
        "US": 400, # gCO2 per kWh (approx. US average, varies wildly by state)
        "EU": 270, # gCO2 per kWh (approx. EU average)
        "IN": 700, # gCO2 per kWh (approx. India average)
        "CN": 600, # gCO2 per kWh (approx. China average)
        "CA": 120, # gCO2 per kWh (approx. Canada average)
        "FR": 50,  # gCO2 per kWh (high nuclear, low carbon)
        "DE": 350, # gCO2 per kWh (Germany)
        "DEFAULT": 450 # Global average placeholder
    }

    def __init__(self, region="DEFAULT"):
        self.region = region.upper()
        self.carbon_intensity_gco2_per_kwh = self.REGION_CARBON_INTENSITIES.get(
            self.region, self.REGION_CARBON_INTENSITIES["DEFAULT"]
        )

    def estimate_carbon_emissions(self, energy_kwh):
        """
        Estimates carbon emissions in grams of CO2 based on energy consumed (kWh).
        """
        return energy_kwh * self.carbon_intensity_gco2_per_kwh

    def set_region(self, region):
        self.region = region.upper()
        self.carbon_intensity_gco2_per_kwh = self.REGION_CARBON_INTENSITIES.get(
            self.region, self.REGION_CARBON_INTENSITIES["DEFAULT"]
        )
        print(f"Carbon intensity set to {self.carbon_intensity_gco2_per_kwh} gCO2/kWh for region {self.region}")