/**
 * BioSenseGA – Main Frontend Controller
 * Complete Dashboard Logic for Artificial Nose Sensor Array Optimization
 */

// Centralized Application State
const AppState = {
    dataLoaded: false,
    isTraining: false,
    baselineTrained: false,
    gaCompleted: false,
    optimizedTrained: false,
    
    datasetInfo: null,
    trainingResults: null,
    selectedIndices: [],
    selectedFeatureNames: [],
    sensorGroups: [],
    
    currentTestSample: null,
    currentHistory: [],
    
    // Pagination & filter state for Dataset Explorer
    tablePage: 1,
    tablePageSize: 15,
    tableTotalPages: 1,
    
    gasNames: ['Ethanol', 'Ethylene', 'Ammonia', 'Acetaldehyde', 'Acetone', 'Toluene'],
    gasColors: {
        'Ethanol': '#4388FF',
        'Ethylene': '#10B981',
        'Ammonia': '#F59E0B',
        'Acetaldehyde': '#EC4899',
        'Acetone': '#8B5CF6',
        'Toluene': '#06B6D4'
    }
};

// Plotly Dark Theme Default Layout Configuration
const PLOTLY_DARK_LAYOUT = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
        family: 'Plus Jakarta Sans, sans-serif',
        color: '#94A3B8',
        size: 11
    },
    margin: { t: 30, r: 25, l: 50, b: 40 },
    xaxis: {
        gridcolor: 'rgba(255, 255, 255, 0.05)',
        zerolinecolor: 'rgba(255, 255, 255, 0.1)',
        tickfont: { color: '#94A3B8' }
    },
    yaxis: {
        gridcolor: 'rgba(255, 255, 255, 0.05)',
        zerolinecolor: 'rgba(255, 255, 255, 0.1)',
        tickfont: { color: '#94A3B8' }
    }
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: false
};

// ==========================================================================
// 1. INITIALIZATION & NAVIGATION
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initSidebarToggle();
    syncInitialState();
});

function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const sectionId = item.getAttribute('data-section');
            if (sectionId) {
                navigateToSection(sectionId);
            }
        });
    });
}

function navigateToSection(sectionId) {
    // Update active nav item
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.getAttribute('data-section') === sectionId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    // Update active view section
    document.querySelectorAll('.view-section').forEach(sec => {
        sec.classList.remove('active');
    });

    const targetSection = document.getElementById(`section-${sectionId}`);
    if (targetSection) {
        targetSection.classList.add('active');
    }

    // Update breadcrumb
    const breadcrumb = document.getElementById('breadcrumb-current');
    if (breadcrumb) {
        const titles = {
            'overview': 'Platform Overview',
            'dataset-explorer': 'Dataset Explorer',
            'model-training': 'Model Training Hub',
            'genetic-algorithm': 'Genetic Algorithm Optimization',
            'feature-selection': 'Feature Selection Analysis',
            'gas-prediction': 'Gas Prediction Workbench',
            'performance-analysis': 'Model Performance Analysis',
            'prediction-history': 'Prediction History',
            'about-biosense': 'About BioSenseGA'
        };
        breadcrumb.textContent = titles[sectionId] || 'Dashboard';
    }

    // Resize Plotly charts in active section
    window.dispatchEvent(new Event('resize'));

    // Trigger section-specific data loads
    if (sectionId === 'dataset-explorer' && AppState.dataLoaded) {
        fetchDatasetTable(AppState.tablePage);
    } else if (sectionId === 'prediction-history') {
        fetchPredictionHistory();
    } else if (sectionId === 'gas-prediction') {
        if (AppState.selectedIndices && AppState.selectedIndices.length > 0) {
            renderDynamicPredictionInputs(AppState.sensorGroups, AppState.selectedIndices);
        } else {
            fetchSelectedFeatures();
        }
    }
}

function initSidebarToggle() {
    const toggleBtn = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('sidebar');
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
        });
    }
}

// ==========================================================================
// 2. STATE SYNCHRONIZATION
// ==========================================================================
async function syncInitialState() {
    try {
        const res = await fetch('/api/model_status');
        const data = await res.json();
        if (data.success && data.status) {
            const st = data.status;
            AppState.dataLoaded = st.data_loaded;
            AppState.isTraining = st.is_training;
            AppState.baselineTrained = st.baseline_trained;
            AppState.gaCompleted = st.ga_completed;
            AppState.optimizedTrained = st.optimized_trained;

            updateReadinessIndicators();

            if (st.data_loaded) {
                // Fetch dataset info & populate overview
                document.getElementById('kpi-samples').textContent = st.n_samples.toLocaleString();
                document.getElementById('ds-samples-count').textContent = st.n_samples.toLocaleString();
                
                // If models trained, fetch selected features
                if (st.ga_completed) {
                    await fetchSelectedFeatures();
                }
            } else {
                // Auto-load dataset on initial launch for seamless UX
                triggerLoadDataset();
            }
        }
    } catch (e) {
        console.warn('Initial state check error:', e);
    }
}

function updateReadinessIndicators() {
    const dot = document.getElementById('system-status-dot');
    const txt = document.getElementById('system-status-text');
    const trainBadge = document.getElementById('nav-train-badge');

    if (AppState.isTraining) {
        if (dot) dot.className = 'status-pulse-dot busy';
        if (txt) txt.textContent = 'Optimizing...';
        if (trainBadge) {
            trainBadge.textContent = 'Running';
            trainBadge.style.backgroundColor = 'var(--primary)';
        }
    } else if (AppState.optimizedTrained) {
        if (dot) dot.className = 'status-pulse-dot';
        if (txt) txt.textContent = 'Models Ready';
        if (trainBadge) {
            trainBadge.textContent = 'Trained';
            trainBadge.style.backgroundColor = 'var(--success)';
        }
    } else if (AppState.dataLoaded) {
        if (dot) dot.className = 'status-pulse-dot standby';
        if (txt) txt.textContent = 'Data Loaded';
        if (trainBadge) {
            trainBadge.textContent = 'Ready';
            trainBadge.style.backgroundColor = 'var(--warning)';
        }
    } else {
        if (dot) dot.className = 'status-pulse-dot standby';
        if (txt) txt.textContent = 'Standby';
        if (trainBadge) {
            trainBadge.textContent = 'Init';
            trainBadge.style.backgroundColor = 'rgba(255,255,255,0.1)';
        }
    }

    // Update Overview Readiness Checklist
    updateChecklistStep('check-dataset', AppState.dataLoaded, 
        'UCI Gas Sensor Array Drift Dataset loaded.', 
        'Dataset not loaded into memory.');
        
    updateChecklistStep('check-baseline', AppState.baselineTrained, 
        'Baseline Random Forest trained on 128 features.', 
        'Baseline model not trained yet.');
        
    updateChecklistStep('check-ga', AppState.gaCompleted, 
        'Genetic Algorithm optimization finished.', 
        'GA feature search not completed.');
        
    updateChecklistStep('check-opt', AppState.optimizedTrained, 
        'GA-Optimized Random Forest ready for inference.', 
        'Optimized model not ready for inference.');

    const readyBadge = document.getElementById('overview-readiness-badge');
    if (readyBadge) {
        if (AppState.optimizedTrained) {
            readyBadge.textContent = 'Ready for Inference';
            readyBadge.style.color = 'var(--success)';
        } else {
            readyBadge.textContent = 'Training Required';
            readyBadge.style.color = 'var(--warning)';
        }
    }
}

function updateChecklistStep(prefix, isDone, doneText, notDoneText) {
    const step = document.getElementById(`${prefix}-step`);
    const icon = document.getElementById(`${prefix}-icon`);
    const text = document.getElementById(`${prefix}-text`);

    if (step && icon && text) {
        if (isDone) {
            step.className = 'stage-step done';
            icon.innerHTML = '<i class="fas fa-circle-check" style="color: var(--success);"></i>';
            text.textContent = doneText;
        } else {
            step.className = 'stage-step';
            icon.innerHTML = '<i class="fas fa-circle-xmark" style="color: var(--text-muted);"></i>';
            text.textContent = notDoneText;
        }
    }
}

// ==========================================================================
// 3. DATASET LOADING & DATASET EXPLORER
// ==========================================================================
async function triggerLoadDataset() {
    const btn = document.getElementById('btn-load-dataset');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading Batches 1-10...';
    }

    showToast('Loading UCI Gas Sensor Array Drift batches...', 'info');

    try {
        const response = await fetch('/api/load_dataset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data_dir: 'data' })
        });
        const data = await response.json();

        if (data.success) {
            AppState.dataLoaded = true;
            AppState.datasetInfo = data.dataset_info;

            // Update overview KPIs
            document.getElementById('kpi-samples').textContent = data.dataset_info.n_samples.toLocaleString();
            document.getElementById('ds-samples-count').textContent = data.dataset_info.n_samples.toLocaleString();

            // Plot Overview Charts
            plotGasDistribution('overview-gas-chart', data.dataset_info.class_distribution);
            plotBatchesDistribution('dataset-batches-chart', data.dataset_info.batch_distribution);
            plotGasDistribution('dataset-classes-chart', data.dataset_info.class_distribution);

            // Fetch first page of dataset table
            fetchDatasetTable(1);

            updateReadinessIndicators();
            showToast(`Loaded ${data.dataset_info.n_samples.toLocaleString()} samples across 10 batches.`, 'success');
        } else {
            showToast(`Dataset loading error: ${data.error}`, 'error');
        }
    } catch (e) {
        showToast(`Failed to connect to backend: ${e.message}`, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-cloud-arrow-down"></i> Load / Reload Dataset';
        }
    }
}

function plotGasDistribution(containerId, classDist) {
    const container = document.getElementById(containerId);
    if (!container || !classDist) return;

    const labels = Object.keys(classDist);
    const values = Object.values(classDist);
    const colors = labels.map(label => AppState.gasColors[label] || '#4388FF');

    const trace = {
        labels: labels,
        values: values,
        type: 'pie',
        hole: 0.55,
        marker: { colors: colors },
        textinfo: 'label+percent',
        hoverinfo: 'label+value+percent',
        textfont: { color: '#F8FAFC', size: 11 }
    };

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        showlegend: false,
        margin: { t: 15, r: 15, l: 15, b: 15 }
    };

    Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
}

function plotBatchesDistribution(containerId, batchDist) {
    const container = document.getElementById(containerId);
    if (!container || !batchDist) return;

    const batches = Object.keys(batchDist).map(b => `Batch ${b}`);
    const counts = Object.values(batchDist);

    const trace = {
        x: batches,
        y: counts,
        type: 'bar',
        marker: {
            color: '#4388FF',
            line: { color: 'rgba(255,255,255,0.1)', width: 1 }
        }
    };

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: 'Batch ID (Recording Time)' },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: 'Sample Count' },
        margin: { t: 20, r: 20, l: 50, b: 40 }
    };

    Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
}

async function fetchDatasetTable(page = 1) {
    if (!AppState.dataLoaded) return;

    const batchFilter = document.getElementById('table-filter-batch')?.value || '';
    const gasFilter = document.getElementById('table-filter-gas')?.value || '';
    const searchQuery = document.getElementById('table-search-input')?.value || '';

    const url = `/api/dataset_table?page=${page}&page_size=${AppState.tablePageSize}&batch=${batchFilter}&gas_class=${gasFilter}&search=${encodeURIComponent(searchQuery)}`;

    try {
        const res = await fetch(url);
        const data = await res.json();

        if (data.success) {
            AppState.tablePage = data.page;
            AppState.tableTotalPages = data.total_pages;

            renderDatasetTableRows(data.samples);

            // Update pagination text & buttons
            const startRecord = (data.page - 1) * data.page_size + 1;
            const endRecord = Math.min(startRecord + data.samples.length - 1, data.total_records);
            
            const info = document.getElementById('table-pagination-info');
            if (info) {
                info.textContent = `Showing ${data.samples.length > 0 ? startRecord : 0} to ${endRecord} of ${data.total_records.toLocaleString()} entries`;
            }

            const pageDisp = document.getElementById('table-page-display');
            if (pageDisp) {
                pageDisp.textContent = `Page ${data.page} of ${data.total_pages}`;
            }

            const btnPrev = document.getElementById('btn-table-prev');
            const btnNext = document.getElementById('btn-table-next');
            if (btnPrev) btnPrev.disabled = data.page <= 1;
            if (btnNext) btnNext.disabled = data.page >= data.total_pages;
        }
    } catch (e) {
        console.error('Failed to fetch dataset table:', e);
    }
}

function renderDatasetTableRows(samples) {
    const tbody = document.getElementById('dataset-table-body');
    if (!tbody) return;

    if (!samples || samples.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 25px;">No records found matching filters.</td></tr>`;
        return;
    }

    let html = '';
    samples.forEach(s => {
        const gasClassLower = s.gas_name.toLowerCase();
        html += `
            <tr>
                <td class="mono"><strong>#${s.sample_id}</strong></td>
                <td><span class="tech-tag" style="padding: 2px 7px; font-size: 11px;">Batch ${s.batch_id}</span></td>
                <td><span class="gas-badge ${gasClassLower}">${s.gas_name}</span></td>
                <td class="mono">${s.s1_dr}</td>
                <td class="mono">${s.s1_abs_dr}</td>
                <td class="mono">${s.s5_dr}</td>
                <td class="mono">${s.s9_dr}</td>
                <td class="mono">${s.s13_dr}</td>
                <td>
                    <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 11px;" onclick="inspectSampleDetail(${s.sample_id})">
                        <i class="fas fa-magnifying-glass"></i> Inspect
                    </button>
                </td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
}

function changeTablePage(delta) {
    const newPage = AppState.tablePage + delta;
    if (newPage >= 1 && newPage <= AppState.tableTotalPages) {
        fetchDatasetTable(newPage);
    }
}

async function inspectSampleDetail(sampleIndex) {
    try {
        const res = await fetch(`/api/dataset_sample_detail?sample_index=${sampleIndex}`);
        const data = await res.json();
        if (data.success) {
            document.getElementById('modal-sample-title').textContent = `Sample #${data.sample_index} Details (Batch ${data.batch_id} - ${data.gas_name})`;

            let bodyHtml = `
                <div style="display: flex; gap: 14px; margin-bottom: 20px; font-size: 13px;">
                    <span class="tech-tag">Batch: <strong>${data.batch_id}</strong></span>
                    <span class="tech-tag">Analyte: <strong>${data.gas_name}</strong></span>
                    <span class="tech-tag">Features: <strong>128 Signal Points</strong></span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; max-height: 480px; overflow-y: auto;">
            `;

            data.sensors.forEach(s => {
                bodyHtml += `
                    <div style="background-color: var(--bg-card-secondary); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong style="color: var(--primary); font-size: 12px;">${s.sensor_name}</strong>
                            <span class="tech-tag" style="padding: 1px 6px; font-size: 10px;">${s.sensor_type}</span>
                        </div>
                        <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 6px;">Mean Response: <strong class="mono" style="color: var(--text-main);">${s.mean_response}</strong></div>
                        <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                            ${s.feature_values.map((v, i) => `<span class="tech-tag mono" style="font-size: 10px; padding: 2px 5px;">F${i+1}: ${v}</span>`).join('')}
                        </div>
                    </div>
                `;
            });

            bodyHtml += `</div>`;
            document.getElementById('modal-sample-body').innerHTML = bodyHtml;
            openModal('modal-sample-detail');
        }
    } catch (e) {
        showToast('Failed to load sample details: ' + e.message, 'error');
    }
}

// ==========================================================================
// 4. MODEL TRAINING & PIPELINE EXECUTION
// ==========================================================================
async function executeModelTraining() {
    if (AppState.isTraining) {
        showToast('Training is already in progress. Please wait.', 'error');
        return;
    }

    AppState.isTraining = true;
    updateReadinessIndicators();

    const btnTrain = document.getElementById('btn-train-pipeline');
    const btnGa = document.getElementById('btn-run-ga-only');
    if (btnTrain) btnTrain.disabled = true;
    if (btnGa) btnGa.disabled = true;

    // Retrieve parameters (optimized defaults: 20 pop, 8 gens for fast 5-second convergence)
    const popSize = parseInt(document.getElementById('param-pop-size')?.value) || 20;
    const nGens = parseInt(document.getElementById('param-n-gens')?.value) || 8;
    const mutRate = parseFloat(document.getElementById('param-mut-rate')?.value) || 0.05;
    const crossRate = parseFloat(document.getElementById('param-cross-rate')?.value) || 0.80;
    const penalty = parseFloat(document.getElementById('param-penalty')?.value) || 0.010;
    const seed = parseInt(document.getElementById('param-seed')?.value) || 42;

    // Reset pipeline visualizer
    resetPipelineVisualizer();
    setPipelineStage(1, 15, 'Stage 1: Preprocessing 13,910 sensor rows & scaling...');

    // Start timer & dynamic progress updater
    let secondsElapsed = 0;
    const timerInterval = setInterval(() => {
        secondsElapsed++;
        const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
        const secs = String(secondsElapsed % 60).padStart(2, '0');
        const badge = document.getElementById('training-timer-badge');
        if (badge) badge.textContent = `${mins}:${secs}`;
    }, 1000);

    // Dynamic progress stages (fast transition)
    const stageTimeout1 = setTimeout(() => setPipelineStage(2, 25, 'Stage 2: Stratified 80/20 train/test partition...'), 250);
    const stageTimeout2 = setTimeout(() => setPipelineStage(3, 45, 'Stage 3: Training Baseline Random Forest (128 features)...'), 500);
    const stageTimeout3 = setTimeout(() => setPipelineStage(4, 65, 'Stage 4: Running Genetic Algorithm feature optimization...'), 800);
    const stageTimeout4 = setTimeout(() => setPipelineStage(5, 80, 'Stage 5: Extracting optimal feature chromosome mask...'), 1100);
    const stageTimeout5 = setTimeout(() => setPipelineStage(6, 95, 'Stage 6: Training GA-Optimized Random Forest on selected features...'), 1400);

    try {
        let response = await fetch('/api/train_models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                population_size: popSize,
                n_generations: nGens,
                mutation_rate: mutRate,
                crossover_rate: crossRate,
                feature_penalty: penalty,
                random_seed: seed
            })
        });

        if (response.status === 404) {
            // Fallback to /api/run_ga if server is running older process
            response = await fetch('/api/run_ga', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    population_size: popSize,
                    n_generations: nGens,
                    mutation_rate: mutRate,
                    crossover_rate: crossRate,
                    feature_penalty: penalty,
                    random_seed: seed
                })
            });
        }

        clearInterval(timerInterval);
        clearTimeout(stageTimeout1);
        clearTimeout(stageTimeout2);
        clearTimeout(stageTimeout3);
        clearTimeout(stageTimeout4);
        clearTimeout(stageTimeout5);

        let data;
        try {
            data = await response.json();
        } catch (parseErr) {
            throw new Error(`Server returned non-JSON response (${response.status} ${response.statusText}). Please restart 'python app.py' in your terminal and reload.`);
        }

        if (data.success) {
            setPipelineStage(7, 100, 'Pipeline execution complete! Both models evaluated on held-out test data.');

            AppState.isTraining = false;
            AppState.dataLoaded = true;
            AppState.baselineTrained = true;
            AppState.gaCompleted = true;
            AppState.optimizedTrained = true;
            AppState.trainingResults = data;
            AppState.selectedIndices = data.optimized.selected_features;
            AppState.selectedFeatureNames = data.optimized.selected_feature_names;
            AppState.sensorGroups = data.sensor_groups;

            // Update UI across all sections
            displayTrainingResults(data);
            plotConvergenceChart('ga-convergence-chart', data.convergence_history);
            renderChromosomeViewer(data.optimized.selected_features);
            renderSensorArchitectureGrid(data.sensor_groups);
            plotSensorRetainedChart('sensor-retained-chart', data.sensor_groups);
            displayPerformanceAnalysis(data);

            // Dynamically generate the inputs for Gas Prediction
            renderDynamicPredictionInputs(data.sensor_groups, data.optimized.selected_features);

            updateReadinessIndicators();
            showToast(`Optimization complete! ${data.optimized.n_features} features selected (${data.optimized.feature_reduction_pct}% reduction).`, 'success');
        } else {
            AppState.isTraining = false;
            const descEl = document.getElementById('pipeline-stage-description');
            if (descEl) descEl.innerHTML = `<span style="color: var(--danger); font-weight: 600;"><i class="fas fa-triangle-exclamation"></i> Training error: ${data.error}</span>`;
            showToast(`Training error: ${data.error}`, 'error');
            updateReadinessIndicators();
        }
    } catch (e) {
        clearInterval(timerInterval);
        AppState.isTraining = false;
        const descEl = document.getElementById('pipeline-stage-description');
        if (descEl) descEl.innerHTML = `<span style="color: var(--danger); font-weight: 600;"><i class="fas fa-triangle-exclamation"></i> Pipeline failed: ${e.message}</span>`;
        showToast(`Request failed: ${e.message}`, 'error');
        updateReadinessIndicators();
    } finally {
        if (btnTrain) btnTrain.disabled = false;
        if (btnGa) btnGa.disabled = false;
    }
}

function resetPipelineVisualizer() {
    for (let i = 1; i <= 7; i++) {
        const step = document.getElementById(`pipe-stage-${i}`);
        if (step) step.className = 'stage-step';
    }
    const bar = document.getElementById('training-progress-bar');
    if (bar) bar.style.width = '0%';
}

function setPipelineStage(stageNum, pct, text) {
    for (let i = 1; i <= 7; i++) {
        const step = document.getElementById(`pipe-stage-${i}`);
        if (step) {
            if (i < stageNum) step.className = 'stage-step done';
            else if (i === stageNum) step.className = 'stage-step active';
            else step.className = 'stage-step';
        }
    }
    const bar = document.getElementById('training-progress-bar');
    if (bar) {
        bar.style.width = `${pct}%`;
        bar.className = stageNum < 7 ? 'progress-fill animated-stripes' : 'progress-fill';
    }
    const txt = document.getElementById('training-status-text');
    if (txt) txt.textContent = text;
    const pctTxt = document.getElementById('training-pct-text');
    if (pctTxt) pctTxt.textContent = `${pct}%`;
}

function displayTrainingResults(data) {
    const panel = document.getElementById('training-results-panel');
    if (panel) panel.style.display = 'block';

    const baseAcc = (data.baseline.accuracy * 100).toFixed(2);
    const optAcc = (data.optimized.accuracy * 100).toFixed(2);
    const delta = (optAcc - baseAcc).toFixed(2);
    const redPct = data.optimized.feature_reduction_pct.toFixed(1);

    document.getElementById('train-res-baseline-acc').textContent = `${baseAcc}%`;
    document.getElementById('train-res-opt-acc').textContent = `${optAcc}%`;
    document.getElementById('train-res-acc-delta').textContent = `Delta: ${delta >= 0 ? '+' : ''}${delta}%`;
    document.getElementById('train-res-reduction').textContent = `${redPct}%`;
    document.getElementById('train-res-feat-counts').textContent = `128 → ${data.optimized.n_features} Features`;
    document.getElementById('train-res-duration').textContent = `${data.pipeline_time}s`;
    document.getElementById('train-conclusion-text').textContent = data.conclusion;

    // Update Overview Cards
    document.getElementById('kpi-selected-features').textContent = `${data.optimized.n_features}`;
    document.getElementById('kpi-reduction-pct').textContent = `${redPct}% Dimensionality Reduction`;
    document.getElementById('kpi-opt-accuracy').textContent = `${optAcc}%`;
    document.getElementById('kpi-acc-sub').textContent = `Baseline: ${baseAcc}%`;
    document.getElementById('kpi-training-status').textContent = 'Optimized';
    document.getElementById('kpi-last-trained').textContent = `Fitted in ${data.pipeline_time}s`;
}

// ==========================================================================
// 5. GENETIC ALGORITHM VISUALIZATION
// ==========================================================================
function updateParamDisplay(tagId, val) {
    const el = document.getElementById(tagId);
    if (el) el.textContent = val;
}

function setPresetParams(type) {
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');

    if (type === 'fast') {
        applyParamValues(15, 6, 0.05, 0.80, 0.010);
    } else if (type === 'balanced') {
        applyParamValues(30, 15, 0.05, 0.80, 0.010);
    } else if (type === 'deep') {
        applyParamValues(50, 25, 0.08, 0.85, 0.015);
    }
}

function applyParamValues(pop, gens, mut, cross, penalty) {
    document.getElementById('param-pop-size').value = pop;
    updateParamDisplay('val-pop-size', pop);

    document.getElementById('param-n-gens').value = gens;
    updateParamDisplay('val-n-gens', gens);

    document.getElementById('param-mut-rate').value = mut;
    updateParamDisplay('val-mut-rate', mut.toFixed(2));

    document.getElementById('param-cross-rate').value = cross;
    updateParamDisplay('val-cross-rate', cross.toFixed(2));

    document.getElementById('param-penalty').value = penalty;
    updateParamDisplay('val-penalty', penalty.toFixed(3));
}

function plotConvergenceChart(containerId, history) {
    const container = document.getElementById(containerId);
    if (!container || !history || history.length === 0) return;

    const gens = history.map(h => `Gen ${h.generation}`);
    const bestFit = history.map(h => h.best_fitness);
    const avgFit = history.map(h => h.avg_fitness);

    const traceBest = {
        x: gens,
        y: bestFit,
        name: 'Best Fitness',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#4388FF', width: 2.5 },
        marker: { size: 6, color: '#4388FF' }
    };

    const traceAvg = {
        x: gens,
        y: avgFit,
        name: 'Mean Fitness',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#8B5CF6', width: 1.5, dash: 'dash' }
    };

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        showlegend: true,
        legend: { x: 0.02, y: 0.98, font: { color: '#94A3B8', size: 10 } },
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: 'Evolutionary Generation' },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: 'Fitness Score' }
    };

    Plotly.newPlot(containerId, [traceBest, traceAvg], layout, PLOTLY_CONFIG);
}

function renderChromosomeViewer(selectedFeatures) {
    const container = document.getElementById('chromosome-viewer-container');
    if (!container) return;

    const selectedSet = new Set(selectedFeatures || []);
    let bitsHtml = '';

    for (let i = 0; i < 128; i++) {
        const isSelected = selectedSet.has(i);
        const sensorId = Math.floor(i / 8) + 1;
        const ftIdx = i % 8;
        const ftTypes = ['DR', 'abs_DR', 'EMAi0.001', 'EMAi0.01', 'EMAi0.1', 'EMAd0.001', 'EMAd0.01', 'EMAd0.1'];
        const name = `S${sensorId}_${ftTypes[ftIdx]}`;

        bitsHtml += `
            <div class="chrom-bit ${isSelected ? 'active' : 'inactive'}" 
                 title="Index ${i} (${name}): ${isSelected ? 'Selected (1)' : 'Eliminated (0)'}">
                ${isSelected ? '1' : '0'}
            </div>
        `;
    }

    container.innerHTML = bitsHtml;
    const tag = document.getElementById('chromosome-selected-tag');
    if (tag) tag.textContent = `${selectedFeatures.length} / 128 Bits Active (1)`;
}

// ==========================================================================
// 6. FEATURE SELECTION MATRIX & SENSOR ARCHITECTURE
// ==========================================================================
function plotSensorRetainedChart(containerId, sensorGroups) {
    const container = document.getElementById(containerId);
    if (!container || !sensorGroups) return;

    const sensorNames = sensorGroups.map(s => s.sensor_name);
    const selectedCounts = sensorGroups.map(s => s.selected_features_count);
    const eliminatedCounts = sensorGroups.map(s => s.eliminated_features_count);

    const traceSelected = {
        x: sensorNames,
        y: selectedCounts,
        name: 'Selected Features',
        type: 'bar',
        marker: { color: '#10B981' }
    };

    const traceEliminated = {
        x: sensorNames,
        y: eliminatedCounts,
        name: 'Eliminated Features',
        type: 'bar',
        marker: { color: '#334155' }
    };

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        barmode: 'stack',
        showlegend: true,
        legend: { orientation: 'h', y: 1.15, x: 0.3 },
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: 'Features (Max 8)', range: [0, 8.5] }
    };

    Plotly.newPlot(containerId, [traceSelected, traceEliminated], layout, PLOTLY_CONFIG);
}

function renderSensorArchitectureGrid(sensorGroups) {
    const container = document.getElementById('sensor-matrix-grid-container');
    if (!container || !sensorGroups) return;

    let html = '';
    sensorGroups.forEach(s => {
        const miniBars = s.features.map(f => `
            <div class="mini-feature-bit ${f.selected ? 'selected' : ''}" title="${f.feature_name}: ${f.selected ? 'Selected' : 'Eliminated'}"></div>
        `).join('');

        html += `
            <div class="sensor-box ${s.active ? 'active-sensor' : 'inactive-sensor'}" onclick="inspectSensorDetail(${s.sensor_id})">
                <div class="sensor-box-top">
                    <span class="sensor-badge-name">
                        <i class="fas fa-microchip" style="color: var(--primary);"></i> ${s.sensor_name}
                    </span>
                    <span class="sensor-type-tag">${s.sensor_type}</span>
                </div>
                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 6px;">
                    ${s.category}
                </div>
                <div class="sensor-mini-bars">
                    ${miniBars}
                </div>
                <div class="sensor-stat-row">
                    <span>Selected: <strong style="color: var(--success);">${s.selected_features_count} / 8</strong></span>
                    <span>Retained: <strong>${s.selection_percentage}%</strong></span>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function inspectSensorDetail(sensorId) {
    if (!AppState.sensorGroups) return;
    const s = AppState.sensorGroups.find(grp => grp.sensor_id === sensorId);
    if (!s) return;

    document.getElementById('modal-sensor-title').textContent = `${s.sensor_name} (${s.sensor_type}) – Feature Breakdown`;

    let html = `
        <div style="margin-bottom: 16px;">
            <div style="font-size: 13px; font-weight: 700; color: var(--text-main);">${s.category}</div>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">Target Analytes: ${s.target_gases}</div>
            <div style="margin-top: 8px;">
                <span class="tech-tag">Selected: <strong>${s.selected_features_count} / 8</strong></span>
                <span class="tech-tag">Selection Rate: <strong>${s.selection_percentage}%</strong></span>
            </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: 8px;">
    `;

    s.features.forEach(f => {
        html += `
            <div style="background-color: var(--bg-card-secondary); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 10px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="mono" style="font-weight: 700; font-size: 12px; color: ${f.selected ? 'var(--primary)' : 'var(--text-muted)'};">${f.feature_name}</span>
                        <span style="font-size: 11px; color: var(--text-secondary);">${f.feature_title} (${f.symbol})</span>
                    </div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">${f.description}</div>
                </div>
                <span class="gas-badge" style="background-color: ${f.selected ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255,255,255,0.05)'}; color: ${f.selected ? 'var(--success)' : 'var(--text-muted)'};">
                    ${f.selected ? 'SELECTED' : 'ELIMINATED'}
                </span>
            </div>
        `;
    });

    html += `</div>`;
    document.getElementById('modal-sensor-body').innerHTML = html;
    openModal('modal-sensor-detail');
}

async function fetchSelectedFeatures() {
    try {
        const res = await fetch('/api/selected_features');
        const data = await res.json();
        if (data.success) {
            AppState.selectedIndices = data.selected_indices;
            AppState.selectedFeatureNames = data.selected_feature_names;
            AppState.sensorGroups = data.sensor_groups;

            renderChromosomeViewer(data.selected_indices);
            renderSensorArchitectureGrid(data.sensor_groups);
            plotSensorRetainedChart('sensor-retained-chart', data.sensor_groups);
            renderDynamicPredictionInputs(data.sensor_groups, data.selected_indices);
        }
    } catch (e) {
        console.warn('Could not fetch selected features:', e);
    }
}

// ==========================================================================
// 7. DYNAMIC SENSOR INPUT GENERATION & GAS PREDICTION (MOST IMPORTANT)
// ==========================================================================
function renderDynamicPredictionInputs(sensorGroups, selectedIndices) {
    const container = document.getElementById('dynamic-inputs-container');
    if (!container) return;

    if (!selectedIndices && AppState.selectedIndices) selectedIndices = AppState.selectedIndices;
    if (!sensorGroups && AppState.sensorGroups) sensorGroups = AppState.sensorGroups;

    if (!selectedIndices || selectedIndices.length === 0) {
        container.innerHTML = `
            <div class="card" style="text-align: center; padding: 40px; color: var(--text-muted);">
                <i class="fas fa-sliders" style="font-size: 32px; margin-bottom: 12px; color: #334155;"></i>
                <p>No features currently selected. Please train the Genetic Algorithm models first.</p>
            </div>
        `;
        return;
    }

    const nSelected = selectedIndices.length;
    const badge = document.getElementById('dynamic-input-count-badge');
    if (badge) badge.textContent = `${nSelected} Inputs Required (Reduced from 128)`;

    const instruction = document.getElementById('prediction-instruction-text');
    if (instruction) {
        instruction.innerHTML = `The Genetic Algorithm has selected <strong>${nSelected}</strong> important features from the original 128 features. Enter the corresponding sensor-derived feature values below to classify the gas analyte.`;
    }

    const sensorTypes = [
        'TGS2600', 'TGS2602', 'TGS2610', 'TGS2620',
        'TGS2600', 'TGS2602', 'TGS2610', 'TGS2620',
        'TGS2600', 'TGS2602', 'TGS2610', 'TGS2620',
        'TGS2600', 'TGS2602', 'TGS2610', 'TGS2620'
    ];
    const ftTypes = ['DR', 'abs_DR', 'EMAi0.001', 'EMAi0.01', 'EMAi0.1', 'EMAd0.001', 'EMAd0.01', 'EMAd0.1'];
    const ftSymbols = {'DR': 'ΔR', 'abs_DR': '|ΔR|', 'EMAi0.001': 'EMAi_0.001', 'EMAi0.01': 'EMAi_0.01', 'EMAi0.1': 'EMAi_0.1', 'EMAd0.001': 'EMAd_0.001', 'EMAd0.01': 'EMAd_0.01', 'EMAd0.1': 'EMAd_0.1'};

    // Build or normalize sensorGroups if undefined
    const selSet = new Set(selectedIndices.map(Number));
    if (!sensorGroups || !Array.isArray(sensorGroups) || sensorGroups.length === 0) {
        sensorGroups = [];
        for (let sIdx = 0; sIdx < 16; sIdx++) {
            const sId = sIdx + 1;
            const features = [];
            for (let f = 0; f < 8; f++) {
                const fIdx = sIdx * 8 + f;
                const ft = ftTypes[f];
                features.push({
                    feature_index: fIdx,
                    feature_name: `S${sId}_${ft}`,
                    symbol: ftSymbols[ft] || ft,
                    feature_title: ft,
                    description: `Sensor ${sId} ${ft} signal response`,
                    selected: selSet.has(fIdx)
                });
            }
            sensorGroups.push({
                sensor_id: sId,
                sensor_name: `Sensor ${sId}`,
                sensor_type: sensorTypes[sIdx],
                features: features
            });
        }
    }

    let formHtml = '';

    // Iterate through sensors that have at least 1 selected feature
    sensorGroups.forEach(s => {
        const activeFeatures = (s.features || []).filter(f => f.selected || selSet.has(f.feature_index));
        if (activeFeatures.length === 0) return;

        formHtml += `
            <div class="sensor-input-cluster">
                <div class="sensor-cluster-header">
                    <span class="cluster-title">
                        <i class="fas fa-microchip" style="color: var(--primary);"></i>
                        ${s.sensor_name} – ${s.sensor_type}
                    </span>
                    <span class="tech-tag" style="font-size: 11px;">${activeFeatures.length} / 8 Features Selected</span>
                </div>
                <div class="cluster-grid">
        `;

        activeFeatures.forEach(f => {
            const sym = f.symbol || ftSymbols[f.feature_code] || 'Val';
            formHtml += `
                <div class="feature-input-card">
                    <div class="feature-card-header">
                        <span class="feature-name-tag">${f.feature_name}</span>
                        <span class="feature-symbol-tag">${sym}</span>
                    </div>
                    <input type="number" step="any" class="feature-number-input" 
                           id="input-feat-${f.feature_index}" 
                           data-feature-name="${f.feature_name}"
                           data-feature-index="${f.feature_index}"
                           placeholder="Enter ${sym} value..." 
                           required>
                    <div class="feature-desc-hint" title="${f.description || ''}">${f.feature_title || f.feature_name}</div>
                </div>
            `;
        });

        formHtml += `
                </div>
            </div>
        `;
    });

    container.innerHTML = formHtml;
}

// LOAD SAMPLE VALUES from held-out test dataset
async function loadHeldOutSampleValues() {
    const btn = document.getElementById('btn-load-sample');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Retrieving Sample...';
    }

    const gasSelect = document.getElementById('select-sample-gas');
    const selectedGas = gasSelect ? gasSelect.value : '';

    try {
        const query = selectedGas ? `?gas=${encodeURIComponent(selectedGas)}` : '';
        let res = await fetch(`/api/get_test_sample${query}`);
        if (res.status === 404) {
            res = await fetch(`/api/get_sample${query}`);
        }

        let data;
        try {
            data = await res.json();
        } catch (parseErr) {
            throw new Error(`Server returned status ${res.status}. Please restart 'python app.py' in your terminal.`);
        }

        if (data.success) {
            AppState.currentTestSample = data;

            // Map selected values if provided or reconstruct from sample_values
            let selValues = data.selected_values || {};
            if ((!data.selected_values || Object.keys(data.selected_values).length === 0) && data.sample_values) {
                const ftTypes = ['DR', 'abs_DR', 'EMAi0.001', 'EMAi0.01', 'EMAi0.1', 'EMAd0.001', 'EMAd0.01', 'EMAd0.1'];
                (AppState.selectedIndices || []).forEach(idx => {
                    const sId = Math.floor(idx / 8) + 1;
                    const ft = ftTypes[idx % 8];
                    const name = `S${sId}_${ft}`;
                    const val = data.sample_values[idx];
                    selValues[name] = val;
                    selValues[String(idx)] = val;
                });
            }

            // Populate all active input fields with genuine unscaled sample values
            let populatedCount = 0;
            Object.keys(selValues).forEach(key => {
                const inputByName = document.querySelector(`input[data-feature-name="${key}"]`);
                const inputByIndex = document.querySelector(`input[data-feature-index="${key}"]`);
                
                if (inputByName) {
                    inputByName.value = selValues[key];
                    inputByName.style.borderColor = 'var(--border)';
                    populatedCount++;
                } else if (inputByIndex) {
                    inputByIndex.value = selValues[key];
                    inputByIndex.style.borderColor = 'var(--border)';
                    populatedCount++;
                }
            });

            // Update sample metadata tag
            const metaTag = document.getElementById('loaded-sample-meta-tag');
            if (metaTag) metaTag.style.display = 'flex';

            const idTag = document.getElementById('sample-meta-id');
            const sampleIdx = data.test_sample_index !== undefined ? data.test_sample_index : (data.sample_index !== undefined ? data.sample_index : '--');
            if (idTag) idTag.innerHTML = `<i class="fas fa-database"></i> Test Sample #${sampleIdx}`;

            const batchTag = document.getElementById('sample-meta-batch');
            if (batchTag) batchTag.innerHTML = `<i class="fas fa-layer-group"></i> Batch ${data.batch_id || 1}`;

            const truthTag = document.getElementById('sample-meta-truth');
            if (truthTag && data.true_gas) {
                truthTag.style.display = 'inline-flex';
                truthTag.innerHTML = `<i class="fas fa-flask"></i> Target Gas: <strong>${data.true_gas}</strong>`;
            }

            const gasName = data.true_gas || (selectedGas ? selectedGas : 'Test Sample');
            showToast(`Loaded ${populatedCount} sensor features for ${gasName}. Click 'Predict Gas Class' to classify!`, 'success');
        } else {
            showToast('Failed to retrieve test sample: ' + (data.error || 'Unknown error'), 'error');
        }
    } catch (e) {
        showToast('Error retrieving test sample: ' + e.message, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-shuffle"></i> Load Sample Values';
        }
    }
}

function clearPredictionInputs() {
    const inputs = document.querySelectorAll('.feature-number-input');
    inputs.forEach(inp => {
        inp.value = '';
        inp.style.borderColor = 'var(--border)';
    });

    const metaTag = document.getElementById('loaded-sample-meta-tag');
    if (metaTag) metaTag.style.display = 'none';

    AppState.currentTestSample = null;
    showToast('Input fields cleared.', 'info');
}

function fillBaselineDefaults() {
    const inputs = document.querySelectorAll('.feature-number-input');
    let filled = 0;
    inputs.forEach(inp => {
        if (inp.value.trim() === '') {
            inp.value = '0.0';
            inp.style.borderColor = 'var(--border)';
            filled++;
        }
    });
    showToast(`Filled ${filled} empty input(s) with baseline signal 0.0.`, 'info');
}

// PREDICT GAS
async function executeGasPrediction() {
    if (!AppState.selectedIndices || AppState.selectedIndices.length === 0) {
        showToast('GA-Optimized model is not trained yet. Train models first.', 'error');
        return;
    }

    const inputs = document.querySelectorAll('.feature-number-input');
    const inputValues = {};
    let missingCount = 0;
    let invalidCount = 0;

    inputs.forEach(inp => {
        const ftName = inp.getAttribute('data-feature-name');
        const ftIdx = inp.getAttribute('data-feature-index');
        const valStr = inp.value.trim();

        if (valStr === '') {
            missingCount++;
            // Auto-fill empty input with 0.0 (neutral baseline sensor response)
            inp.value = '0.0';
            inp.style.borderColor = 'var(--border)';
            inputValues[ftName] = 0.0;
            if (ftIdx) inputValues[ftIdx] = 0.0;
        } else {
            const num = parseFloat(valStr);
            if (isNaN(num)) {
                invalidCount++;
                inp.style.borderColor = 'var(--danger)';
            } else {
                inp.style.borderColor = 'var(--border)';
                inputValues[ftName] = num;
                if (ftIdx) inputValues[ftIdx] = num;
            }
        }
    });

    if (invalidCount > 0) {
        showToast(`Found ${invalidCount} invalid numeric inputs. Please verify format.`, 'error');
        return;
    }

    if (missingCount > 0) {
        showToast(`Auto-populated ${missingCount} unfilled inputs with baseline 0.0 and proceeding to classify.`, 'info');
    }

    // Construct full 128-element sensor vector as universal fallback for legacy backend routes
    const ftTypes = ['DR', 'abs_DR', 'EMAi0.001', 'EMAi0.01', 'EMAi0.1', 'EMAd0.001', 'EMAd0.01', 'EMAd0.1'];
    const fullSensorVector = new Array(128).fill(0.0);
    for (let i = 0; i < 128; i++) {
        const sId = Math.floor(i / 8) + 1;
        const ft = ftTypes[i % 8];
        const name = `S${sId}_${ft}`;
        if (inputValues[name] !== undefined) {
            fullSensorVector[i] = inputValues[name];
        } else if (inputValues[String(i)] !== undefined) {
            fullSensorVector[i] = inputValues[String(i)];
        } else if (AppState.currentTestSample && AppState.currentTestSample.sample_values && AppState.currentTestSample.sample_values[i] !== undefined) {
            fullSensorVector[i] = AppState.currentTestSample.sample_values[i];
        } else {
            fullSensorVector[i] = 0.0;
        }
    }

    const btn = document.getElementById('btn-predict-gas');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Classifying...';
    }

    const payload = {
        features: inputValues,
        sensor_values: fullSensorVector,
        input_source: AppState.currentTestSample ? 'Held-out Dataset Sample' : 'Manual Sensor Input',
        sample_id: AppState.currentTestSample ? (AppState.currentTestSample.test_sample_index ?? AppState.currentTestSample.sample_index) : null,
        ground_truth: AppState.currentTestSample ? AppState.currentTestSample.true_gas : null
    };

    try {
        let res = await fetch('/api/predict_selected', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (res.status === 404 || res.status === 400) {
            // Fallback for full vector prediction endpoint if older process
            const fallbackRes = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (fallbackRes.ok) {
                res = fallbackRes;
            }
        }

        let data;
        try {
            data = await res.json();
        } catch (parseErr) {
            throw new Error(`Server returned non-JSON (${res.status}). Please restart 'python app.py' in your terminal.`);
        }

        if (data.success) {
            displayPredictionResult(data);
            showToast(`Predicted: ${data.predicted_gas} (${((data.confidence || 0) * 100).toFixed(1)}% confidence)`, 'success');
            // Refresh prediction count badge
            syncPredictionCount();
        } else {
            showToast(`Prediction rejected: ${data.error}`, 'error');
        }
    } catch (e) {
        showToast(`Prediction failed: ${e.message}`, 'error');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-wand-magic-sparkles"></i> Predict Gas Class';
        }
    }
}

function displayPredictionResult(data) {
    const card = document.getElementById('prediction-result-card');
    if (card) {
        card.style.borderColor = AppState.gasColors[data.predicted_gas] || 'var(--primary)';
    }

    const nameEl = document.getElementById('res-gas-name');
    if (nameEl) {
        nameEl.textContent = data.predicted_gas || 'Unknown Gas';
        nameEl.style.color = AppState.gasColors[data.predicted_gas] || '#fff';
    }

    const formulaEl = document.getElementById('res-gas-formula');
    if (formulaEl) {
        if (data.gas_metadata && (data.gas_metadata.formula || data.gas_metadata.type)) {
            formulaEl.textContent = `Formula: ${data.gas_metadata.formula || '--'} | Category: ${data.gas_metadata.type || '--'}`;
        } else {
            const defaultMetas = {
                'Ethanol': 'Formula: C2H5OH | Category: Primary Alcohol',
                'Ethylene': 'Formula: C2H4 | Category: Alkene Hydrocarbon',
                'Ammonia': 'Formula: NH3 | Category: Inorganic Base',
                'Acetaldehyde': 'Formula: CH3CHO | Category: Aldehyde',
                'Acetone': 'Formula: C3H6O | Category: Ketone',
                'Toluene': 'Formula: C7H8 | Category: Aromatic Hydrocarbon'
            };
            formulaEl.textContent = defaultMetas[data.predicted_gas] || 'AI Chemical Classification Profile';
        }
    }

    const confPct = ((data.confidence || 0) * 100).toFixed(1);
    const confText = document.getElementById('res-confidence-pct');
    if (confText) confText.textContent = `${confPct}%`;

    const confBar = document.getElementById('res-confidence-bar');
    if (confBar) {
        confBar.style.width = `${confPct}%`;
        confBar.style.backgroundColor = (data.confidence || 0) > 0.8 ? 'var(--success)' : ((data.confidence || 0) > 0.5 ? 'var(--primary)' : 'var(--warning)');
    }

    const modelEl = document.getElementById('res-model-used');
    if (modelEl) modelEl.textContent = data.model_type || 'GA-Optimized Random Forest';

    const featEl = document.getElementById('res-features-used');
    if (featEl) featEl.textContent = `${data.features_used || (AppState.selectedIndices ? AppState.selectedIndices.length : 63)} Features`;

    const timeEl = document.getElementById('res-timestamp');
    if (timeEl) timeEl.textContent = data.timestamp || new Date().toLocaleTimeString();

    // Ground truth verification
    const gtRow = document.getElementById('res-ground-truth-row');
    const gtMatch = document.getElementById('res-truth-match');
    if (gtRow && gtMatch) {
        const gt = data.ground_truth || (AppState.currentTestSample ? AppState.currentTestSample.true_gas : null);
        if (gt) {
            gtRow.style.display = 'flex';
            const isMatch = gt.toLowerCase() === (data.predicted_gas || '').toLowerCase();
            gtMatch.textContent = isMatch ? `Verified Match (${gt})` : `Mismatch (Truth: ${gt})`;
            gtMatch.style.color = isMatch ? 'var(--success)' : 'var(--danger)';
        } else {
            gtRow.style.display = 'none';
        }
    }

    // Plot horizontal probability chart
    plotProbabilitiesChart('prediction-prob-chart', data.probabilities);
}

function plotProbabilitiesChart(containerId, probabilities) {
    const container = document.getElementById(containerId);
    if (!container || !probabilities) return;

    const gases = Object.keys(probabilities);
    const probs = gases.map(g => (probabilities[g] * 100).toFixed(1));
    const colors = gases.map(g => AppState.gasColors[g] || '#4388FF');

    const trace = {
        x: probs,
        y: gases,
        type: 'bar',
        orientation: 'h',
        marker: { color: colors },
        text: probs.map(p => `${p}%`),
        textposition: 'auto',
        textfont: { color: '#ffffff', size: 10, family: 'JetBrains Mono' }
    };

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        margin: { t: 5, r: 25, l: 85, b: 25 },
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: 'Probability (%)', range: [0, 105] },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, automargin: true }
    };

    Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
}

// ==========================================================================
// 8. PERFORMANCE ANALYSIS & BENCHMARKS
// ==========================================================================
function displayPerformanceAnalysis(data) {
    const base = data.baseline;
    const opt = data.optimized;

    const baseAcc = (base.accuracy * 100).toFixed(2);
    const optAcc = (opt.accuracy * 100).toFixed(2);
    const delta = (optAcc - baseAcc).toFixed(2);
    const redPct = opt.feature_reduction_pct.toFixed(1);
    const speedup = (base.training_time / (opt.training_time || 0.01)).toFixed(1);

    document.getElementById('perf-base-acc').textContent = `${baseAcc}%`;
    document.getElementById('perf-opt-acc').textContent = `${optAcc}%`;
    document.getElementById('perf-acc-delta').textContent = `Delta: ${delta >= 0 ? '+' : ''}${delta}%`;
    document.getElementById('perf-red-pct').textContent = `${redPct}%`;
    document.getElementById('perf-feat-sub').textContent = `128 → ${opt.n_features} Features`;
    document.getElementById('perf-speedup').textContent = `${speedup}x`;

    // Metrics Comparison Chart
    plotMetricsComparison('perf-metrics-chart', base, opt);

    // Feature Reduction Donut Chart
    plotFeatureReductionDonut('perf-reduction-chart', opt.n_features, 128 - opt.n_features);

    // Confusion Matrices
    plotConfusionMatrixHeatmap('cm-chart-optimized', opt.confusion_matrix, 'Optimized Random Forest Confusion Matrix');
    plotConfusionMatrixHeatmap('cm-chart-baseline', base.confusion_matrix, 'Baseline Random Forest Confusion Matrix');

    // Classification Report Table
    renderClassificationReportTable(opt.per_class_report);

    // Academic Conclusion Paragraph
    const conclusionEl = document.getElementById('academic-conclusion-paragraph');
    if (conclusionEl) conclusionEl.textContent = data.conclusion;
}

function plotMetricsComparison(containerId, base, opt) {
    const metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score'];
    const baseVals = [base.accuracy * 100, base.precision * 100, base.recall * 100, base.f1_score * 100];
    const optVals = [opt.accuracy * 100, opt.precision * 100, opt.recall * 100, opt.f1_score * 100];

    const traceBase = {
        x: metrics,
        y: baseVals,
        name: 'Baseline (128 Feat)',
        type: 'bar',
        marker: { color: '#64748B' }
    };

    const traceOpt = {
        x: metrics,
        y: optVals,
        name: 'GA-Optimized',
        type: 'bar',
        marker: { color: '#4388FF' }
    };

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        barmode: 'group',
        showlegend: true,
        legend: { orientation: 'h', y: 1.15, x: 0.2 },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: 'Percentage (%)', range: [80, 102] }
    };

    Plotly.newPlot(containerId, [traceBase, traceOpt], layout, PLOTLY_CONFIG);
}

function plotFeatureReductionDonut(containerId, selected, eliminated) {
    const trace = {
        labels: ['Selected Features', 'Eliminated Features'],
        values: [selected, eliminated],
        type: 'pie',
        hole: 0.6,
        marker: { colors: ['#10B981', '#1E293B'] },
        textinfo: 'label+value+percent',
        hoverinfo: 'label+value+percent'
    };

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        showlegend: false,
        margin: { t: 15, r: 15, l: 15, b: 15 }
    };

    Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
}

function plotConfusionMatrixHeatmap(containerId, cmData, title) {
    const container = document.getElementById(containerId);
    if (!container || !cmData) return;

    const labels = AppState.gasNames;

    const trace = {
        z: cmData,
        x: labels,
        y: labels,
        type: 'heatmap',
        colorscale: [
            [0, '#0D1424'],
            [0.2, '#17243A'],
            [0.5, '#4388FF'],
            [1, '#10B981']
        ],
        hoverongaps: false
    };

    const annotations = [];
    for (let i = 0; i < labels.length; i++) {
        for (let j = 0; j < labels.length; j++) {
            annotations.push({
                x: labels[j],
                y: labels[i],
                text: String(cmData[i][j]),
                font: { family: 'JetBrains Mono', color: '#ffffff', size: 12 },
                showarrow: false
            });
        }
    }

    const layout = {
        ...PLOTLY_DARK_LAYOUT,
        title: { text: title, font: { color: '#F8FAFC', size: 13 } },
        xaxis: { ...PLOTLY_DARK_LAYOUT.xaxis, title: 'Predicted Gas Analyte' },
        yaxis: { ...PLOTLY_DARK_LAYOUT.yaxis, title: 'Actual Ground Truth' },
        annotations: annotations
    };

    Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
}

function toggleCmView(type) {
    const optChart = document.getElementById('cm-chart-optimized');
    const baseChart = document.getElementById('cm-chart-baseline');
    const btnOpt = document.getElementById('btn-cm-opt');
    const btnBase = document.getElementById('btn-cm-base');

    if (type === 'optimized') {
        if (optChart) optChart.style.display = 'block';
        if (baseChart) baseChart.style.display = 'none';
        if (btnOpt) btnOpt.classList.add('active');
        if (btnBase) btnBase.classList.remove('active');
    } else {
        if (optChart) optChart.style.display = 'none';
        if (baseChart) baseChart.style.display = 'block';
        if (btnOpt) btnOpt.classList.remove('active');
        if (btnBase) btnBase.classList.add('active');
    }
}

function renderClassificationReportTable(report) {
    const tbody = document.getElementById('classification-report-body');
    if (!tbody || !report) return;

    let html = '';
    AppState.gasNames.forEach(gas => {
        const row = report[gas];
        if (row) {
            const prec = (row.precision * 100).toFixed(2);
            const rec = (row.recall * 100).toFixed(2);
            const f1 = (row['f1-score'] * 100).toFixed(2);
            const supp = row.support;

            html += `
                <tr>
                    <td><span class="gas-badge ${gas.toLowerCase()}">${gas}</span></td>
                    <td class="mono">${prec}%</td>
                    <td class="mono">${rec}%</td>
                    <td class="mono">${f1}%</td>
                    <td class="mono">${supp}</td>
                </tr>
            `;
        }
    });

    tbody.innerHTML = html;
}

// ==========================================================================
// 9. PREDICTION HISTORY
// ==========================================================================
async function fetchPredictionHistory() {
    try {
        const res = await fetch('/api/prediction_history');
        const data = await res.json();
        if (data.success) {
            AppState.currentHistory = data.history;
            renderPredictionHistoryTable(data.history);
            syncPredictionCount();
        }
    } catch (e) {
        console.error('Failed to load prediction history:', e);
    }
}

function renderPredictionHistoryTable(history) {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;

    if (!history || history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 30px;">No predictions recorded in this session.</td></tr>`;
        return;
    }

    let html = '';
    history.forEach(item => {
        const gasLower = item.predicted_gas.toLowerCase();
        const confPct = (item.confidence * 100).toFixed(1);
        const truthTag = item.ground_truth 
            ? `<span class="gas-badge ${item.ground_truth.toLowerCase()}">${item.ground_truth}</span>`
            : '<span class="text-muted">N/A</span>';

        html += `
            <tr>
                <td class="mono"><strong>${item.id}</strong></td>
                <td><small class="text-muted">${item.timestamp}</small></td>
                <td><span class="gas-badge ${gasLower}">${item.predicted_gas}</span></td>
                <td class="mono">${confPct}%</td>
                <td><small>${item.model_used}</small></td>
                <td class="mono">${item.n_features_used}</td>
                <td><span class="tech-tag" style="font-size: 11px;">${item.input_source}</span></td>
                <td>${truthTag}</td>
                <td>
                    <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="viewHistoryDetail('${item.id}')">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
}

function viewHistoryDetail(predId) {
    const item = AppState.currentHistory.find(h => h.id === predId);
    if (!item) return;

    document.getElementById('modal-history-title').textContent = `Inference Audit: ${item.id} (${item.predicted_gas})`;

    let html = `
        <div style="display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;">
            <span class="tech-tag">Timestamp: <strong>${item.timestamp}</strong></span>
            <span class="tech-tag">Confidence: <strong>${(item.confidence * 100).toFixed(1)}%</strong></span>
            <span class="tech-tag">Source: <strong>${item.input_source}</strong></span>
        </div>

        <h4 style="font-size: 13px; font-weight: 700; color: var(--text-main); margin-bottom: 10px;">Class Probabilities:</h4>
        <div style="display: flex; flex-direction: column; gap: 8px;">
    `;

    Object.keys(item.probabilities).forEach(gas => {
        const p = (item.probabilities[gas] * 100).toFixed(1);
        html += `
            <div style="background-color: var(--bg-card-secondary); border-radius: var(--radius-sm); padding: 8px 12px; display: flex; justify-content: space-between; align-items: center;">
                <span class="gas-badge ${gas.toLowerCase()}">${gas}</span>
                <span class="mono" style="font-weight: 700; color: var(--text-main);">${p}%</span>
            </div>
        `;
    });

    html += `</div>`;
    document.getElementById('modal-history-body').innerHTML = html;
    openModal('modal-history-detail');
}

async function confirmClearHistory() {
    if (confirm('Are you sure you want to clear the prediction history for this session?')) {
        try {
            const res = await fetch('/api/clear_history', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                fetchPredictionHistory();
                showToast('Prediction history cleared.', 'info');
            }
        } catch (e) {
            showToast('Failed to clear history: ' + e.message, 'error');
        }
    }
}

function syncPredictionCount() {
    const badge = document.getElementById('nav-history-count');
    if (badge) {
        fetch('/api/prediction_history')
            .then(r => r.json())
            .then(d => {
                if (d.success) badge.textContent = d.count;
            })
            .catch(() => {});
    }
}

// ==========================================================================
// 10. MODAL & TOAST UTILITIES
// ==========================================================================
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.add('active');
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.remove('active');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const icons = {
        'success': 'fas fa-circle-check',
        'error': 'fas fa-circle-exclamation',
        'info': 'fas fa-circle-info'
    };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="${icons[type] || icons.info}"></i>
        <span style="font-size: 13px;">${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4500);
}
