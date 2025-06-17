const runSimulationBtn = document.getElementById('runSimulationBtn');
const loader = document.getElementById('loader');
const totalCarbonSpan = document.getElementById('totalCarbon');
const totalEnergySpan = document.getElementById('totalEnergy');
const totalDurationSpan = document.getElementById('totalDuration');
const llmAnalysisContent = document.getElementById('llmAnalysisContent');
const detailedReportContent = document.getElementById('detailedReportContent');

let carbonChartInstance = null;
let energyChartInstance = null;

runSimulationBtn.addEventListener('click', async () => {
    runSimulationBtn.disabled = true;
    loader.style.display = 'block'; // Show loader

    // Clear previous results
    totalCarbonSpan.textContent = 'N/A';
    totalEnergySpan.textContent = 'N/A';
    totalDurationSpan.textContent = 'N/A';
    llmAnalysisContent.innerHTML = '<p>Running simulation and fetching insights...</p>';
    detailedReportContent.textContent = 'Fetching detailed report...';

    // Destroy existing chart instances if they exist
    if (carbonChartInstance) {
        carbonChartInstance.destroy();
    }
    if (energyChartInstance) {
        energyChartInstance.destroy();
    }

    try {
        const response = await fetch('http://127.0.0.1:5000/run-simulation');
        const data = await response.json();

        // Update summary cards
        totalCarbonSpan.textContent = data.total_carbon_gco2.toFixed(4);
        totalEnergySpan.textContent = data.total_energy_kwh.toFixed(6);
        totalDurationSpan.textContent = data.total_duration_sec.toFixed(2);

        // Update LLM analysis
        llmAnalysisContent.textContent = data.llm_analysis;

        // Update detailed report (simple stringification for now)
        let detailedReportString = `Total Duration: ${data.total_duration_sec.toFixed(2)} seconds\n`;
        detailedReportString += `Total Energy: ${data.total_energy_kwh.toFixed(6)} kWh\n`;
        detailedReportString += `Total Carbon Emissions: ${data.total_carbon_gco2.toFixed(4)} gCO2\n\n`;
        detailedReportString += "--- Detailed Step Report ---\n";

        for (const stepName in data.steps) {
            const stepData = data.steps[stepName];
            detailedReportString += `\nStep: ${stepName}\n`;
            detailedReportString += `  Duration: ${stepData.duration.toFixed(2)} s\n`;
            detailedReportString += `  CPU Energy (Measured): ${stepData.cpu_energy_kwh.toFixed(6)} kWh\n`;
            if (stepData.gpu_energy_kwh > 0) {
                detailedReportString += `  GPU Energy (Measured): ${stepData.gpu_energy_kwh.toFixed(6)} kWh\n`;
            }
            if (stepData.external_energy_kwh > 0) {
                detailedReportString += `  External Energy (Estimated): ${stepData.external_energy_kwh.toFixed(6)} kWh\n`;
                detailedReportString += `  External Carbon (Estimated): ${stepData.total_carbon_gco2_for_external_only ? stepData.total_carbon_gco2_for_external_only.toFixed(4) : 'N/A'} gCO2\n`;
            }
            detailedReportString += `  Total Step Carbon Emissions: ${stepData.total_carbon_gco2.toFixed(4)} gCO2\n`;
        }
        detailedReportContent.textContent = detailedReportString;


        // Prepare data for Chart.js
        const stepNames = Object.keys(data.steps);
        const carbonData = Object.values(data.steps).map(step => step.total_carbon_gco2);
        const cpuEnergyData = Object.values(data.steps).map(step => step.cpu_energy_kwh);
        const gpuEnergyData = Object.values(data.steps).map(step => step.gpu_energy_kwh);
        const externalEnergyData = Object.values(data.steps).map(step => step.external_energy_kwh);

        // Chart 1: Carbon Emissions per Step
        const carbonCtx = document.getElementById('carbonChart').getContext('2d');
        carbonChartInstance = new Chart(carbonCtx, {
            type: 'bar',
            data: {
                labels: stepNames,
                datasets: [{
                    label: 'Carbon Emissions (gCO2)',
                    data: carbonData,
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Carbon Emissions (gCO2)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Agent Step'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Carbon Emissions per AI Agent Step'
                    }
                }
            }
        });

        // Chart 2: Energy Breakdown per Step (Stacked Bar Chart)
        const energyCtx = document.getElementById('energyChart').getContext('2d');
        energyChartInstance = new Chart(energyCtx, {
            type: 'bar',
            data: {
                labels: stepNames,
                datasets: [
                    {
                        label: 'CPU Energy (kWh)',
                        data: cpuEnergyData,
                        backgroundColor: 'rgba(255, 99, 132, 0.6)'
                    },
                    {
                        label: 'GPU Energy (kWh)',
                        data: gpuEnergyData,
                        backgroundColor: 'rgba(54, 162, 235, 0.6)'
                    },
                    {
                        label: 'External Energy (kWh)',
                        data: externalEnergyData,
                        backgroundColor: 'rgba(255, 206, 86, 0.6)'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: true,
                        title: {
                            display: true,
                            text: 'Agent Step'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    },
                    y: {
                        stacked: true,
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Energy Consumption (kWh)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Energy Consumption Breakdown per AI Agent Step'
                    }
                }
            }
        });


    } catch (error) {
        console.error('Error running simulation:', error);
        llmAnalysisContent.innerHTML = `<p style="color: red;">Error: ${error.message}. Please check the backend server.</p>`;
        detailedReportContent.textContent = `Error: ${error.message}.`;
        totalCarbonSpan.textContent = 'Error';
        totalEnergySpan.textContent = 'Error';
        totalDurationSpan.textContent = 'Error';
    } finally {
        runSimulationBtn.disabled = false;
        loader.style.display = 'none'; // Hide loader
    }
});