// --- DOM Element References ---
const runSimulationBtn = document.getElementById('runSimulationBtn');
const loader = document.getElementById('loader');
const totalCarbonSpan = document.getElementById('totalCarbon');
const totalEnergySpan = document.getElementById('totalEnergy');
const totalDurationSpan = document.getElementById('totalDuration');
const llmAnalysisContent = document.getElementById('llmAnalysisContent');
const detailedReportContent = document.getElementById('detailedReportContent');

// Tab elements
const tabDashboardBtn = document.getElementById('tabDashboardBtn');
const tabEstimatorBtn = document.getElementById('tabEstimatorBtn');
const dashboardContent = document.getElementById('dashboardContent');
const estimatorContent = document.getElementById('estimatorContent');

// Cost Estimator elements
const calculateCostBtn = document.getElementById('calculateCostBtn');
const numAgentsInput = document.getElementById('numAgents');
const runsPerDayInput = document.getElementById('runsPerDay');
const costPerKwhInput = document.getElementById('costPerKwh');
const costResultsSection = document.getElementById('costResultsSection');
const dailyCostSpan = document.getElementById('dailyCost');
const monthlyCostSpan = document.getElementById('monthlyCost');
const yearlyCostSpan = document.getElementById('yearlyCost');
const costEstimatorInfo = document.getElementById('cost-estimator-info');

// --- Global State ---
let carbonChartInstance = null;
let energyChartInstance = null;
let lastRunEnergyKwh = null; // Store the last simulation's energy result

// --- Tab Navigation Logic ---
function showTab(tabToShow) {
    // Hide all content
    dashboardContent.classList.remove('active');
    estimatorContent.classList.remove('active');

    // Deactivate all buttons
    tabDashboardBtn.classList.remove('active');
    tabEstimatorBtn.classList.remove('active');
    
    // Activate the selected tab and content
    if (tabToShow === 'dashboard') {
        dashboardContent.classList.add('active');
        tabDashboardBtn.classList.add('active');
    } else if (tabToShow === 'estimator') {
        estimatorContent.classList.add('active');
        tabEstimatorBtn.classList.add('active');
    }
}

tabDashboardBtn.addEventListener('click', () => showTab('dashboard'));
tabEstimatorBtn.addEventListener('click', () => showTab('estimator'));

// --- Cost Calculation Logic ---
calculateCostBtn.addEventListener('click', () => {
    if (lastRunEnergyKwh === null || lastRunEnergyKwh === 0) {
        costEstimatorInfo.textContent = 'Error: Please run a successful carbon simulation first to provide the energy data for this calculation.';
        costEstimatorInfo.style.backgroundColor = '#ffebed';
        costEstimatorInfo.style.borderColor = '#ffcdd2';
        costEstimatorInfo.style.color = '#c62828';
        return;
    }

    const numAgents = parseFloat(numAgentsInput.value);
    const runsPerDay = parseFloat(runsPerDayInput.value);
    const costPerKwh = parseFloat(costPerKwhInput.value);

    if (isNaN(numAgents) || isNaN(runsPerDay) || isNaN(costPerKwh)) {
        alert("Please enter valid numbers in all fields.");
        return;
    }

    const dailyCost = numAgents * runsPerDay * lastRunEnergyKwh * costPerKwh;
    const monthlyCost = dailyCost * 30.44; // Average days in a month
    const yearlyCost = dailyCost * 365.25; // Account for leap years

    dailyCostSpan.textContent = dailyCost.toFixed(2);
    monthlyCostSpan.textContent = monthlyCost.toFixed(2);
    yearlyCostSpan.textContent = yearlyCost.toFixed(2);
    
    costResultsSection.style.display = 'block';
    costEstimatorInfo.textContent = `Calculations based on an energy consumption of ${lastRunEnergyKwh.toFixed(6)} kWh per run.`;
    costEstimatorInfo.style.backgroundColor = '#e8f5e9';
    costEstimatorInfo.style.borderColor = '#c8e6c9';
    costEstimatorInfo.style.color = '#2e7d32';
});

// --- Simulation Logic ---
runSimulationBtn.addEventListener('click', async () => {
    runSimulationBtn.disabled = true;
    loader.style.display = 'block';

    // Clear previous results
    totalCarbonSpan.textContent = 'N/A';
    totalEnergySpan.textContent = 'N/A';
    totalDurationSpan.textContent = 'N/A';
    llmAnalysisContent.innerHTML = '<p>Running simulation and fetching insights...</p>';
    detailedReportContent.textContent = 'Fetching detailed report...';
    lastRunEnergyKwh = null; // Reset energy state

    if (carbonChartInstance) carbonChartInstance.destroy();
    if (energyChartInstance) energyChartInstance.destroy();

    try {
        // IMPORTANT: Replace with your actual backend endpoint if deployed
        const response = await fetch('http://127.0.0.1:5000/run-simulation');
         if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        lastRunEnergyKwh = data.total_energy_kwh; // Save for cost estimator

        // Update summary cards
        totalCarbonSpan.textContent = data.total_carbon_gco2.toFixed(4);
        totalEnergySpan.textContent = data.total_energy_kwh.toFixed(6);
        totalDurationSpan.textContent = data.total_duration_sec.toFixed(2);

        // Update LLM analysis
        llmAnalysisContent.textContent = data.llm_analysis;

        // Update detailed report
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
            }
            detailedReportString += `  Total Step Carbon Emissions: ${stepData.total_carbon_gco2.toFixed(4)} gCO2\n`;
        }
        detailedReportContent.textContent = detailedReportString;

        // Prepare data for charts
        const stepNames = Object.keys(data.steps);
        const carbonData = Object.values(data.steps).map(step => step.total_carbon_gco2);
        const cpuEnergyData = Object.values(data.steps).map(step => step.cpu_energy_kwh);
        const gpuEnergyData = Object.values(data.steps).map(step => step.gpu_energy_kwh);
        const externalEnergyData = Object.values(data.steps).map(step => step.external_energy_kwh);

        // Render charts
        renderCarbonChart(stepNames, carbonData);
        renderEnergyChart(stepNames, cpuEnergyData, gpuEnergyData, externalEnergyData);

    } catch (error) {
        console.error('Error running simulation:', error);
        const errorMessage = `Error: ${error.message}. Please ensure the backend server is running and accessible.`;
        llmAnalysisContent.innerHTML = `<p style="color: red;">${errorMessage}</p>`;
        detailedReportContent.textContent = errorMessage;
        totalCarbonSpan.textContent = 'Error';
        totalEnergySpan.textContent = 'Error';
        totalDurationSpan.textContent = 'Error';
    } finally {
        runSimulationBtn.disabled = false;
        loader.style.display = 'none';
    }
});

// --- Chart Rendering Functions ---
function renderCarbonChart(labels, data) {
    const ctx = document.getElementById('carbonChart').getContext('2d');
    carbonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Carbon Emissions (gCO2)',
                data: data,
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true } },
            plugins: { legend: { display: false } }
        }
    });
}

function renderEnergyChart(labels, cpuData, gpuData, externalData) {
    const ctx = document.getElementById('energyChart').getContext('2d');
    energyChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'CPU Energy (kWh)', data: cpuData, backgroundColor: 'rgba(255, 99, 132, 0.6)' },
                { label: 'GPU Energy (kWh)', data: gpuData, backgroundColor: 'rgba(54, 162, 235, 0.6)' },
                { label: 'External Energy (kWh)', data: externalData, backgroundColor: 'rgba(255, 206, 86, 0.6)' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
            plugins: { legend: { position: 'top' } }
        }
    });
}
