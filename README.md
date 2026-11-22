# 🧬 BioSenseGA – AI-Powered Evolutionary Gas Classification & E-Nose Optimization Platform

**Academic Engineering Lab Project (40% Weightage)**  
*Genetic Algorithm-Based Feature Selection & Random Forest Ensemble Classification on the UCI Gas Sensor Array Drift Dataset*

---

## 📌 Table of Contents
1. [Project Overview](#-1-project-overview)
2. [Key Objectives & Scientific Motivation](#-2-key-objectives--scientific-motivation)
3. [Technologies & System Requirements](#-3-technologies--system-requirements)
4. [Dataset Architecture & Hardware Sensor Array](#-4-dataset-architecture--hardware-sensor-array)
5. [System Architecture & 9-Section Dashboard](#-5-system-architecture--9-section-dashboard)
6. [Genetic Algorithm Mathematical Formulation](#-6-genetic-algorithm-mathematical-formulation)
7. [Installation & Setup Instructions](#-7-installation--setup-instructions)
8. [How to Run the Application](#-8-how-to-run-the-application)
9. [Step-by-Step User Guide](#-9-step-by-step-user-guide)
   - [A. Loading the Dataset](#a-loading-the-dataset)
   - [B. Training the Models](#b-training-the-models)
   - [C. Genetic Algorithm Optimization](#c-genetic-algorithm-optimization)
   - [D. Manual Feature Entry & Gas Prediction](#d-manual-feature-entry--gas-prediction)
   - [E. Interpreting Predictions & Performance Results](#e-interpreting-predictions--performance-results)
10. [Machine Learning Validation & Anti-Leakage Protocol](#-10-machine-learning-validation--anti-leakage-protocol)
11. [Project Limitations & Engineering Discussion](#-11-project-limitations--engineering-discussion)
12. [Academic Syllabus Connection & References](#-12-academic-syllabus-connection--references)

---

## 🔬 1. Project Overview

**BioSenseGA** is an artificial intelligence-driven electronic nose (e-nose) simulation and gas classification platform. In real-world chemical sensing, artificial nose arrays suffer from **sensor drift** caused by chemical aging, thermal fluctuations, and environmental contamination over extended operational lifespans. Furthermore, operating 16 physical metal oxide semiconductor (MOX) sensors with 128 extracted transient dynamics features imposes substantial hardware costs, electrical power drain, and high dimensionality.

BioSenseGA addresses this challenge by employing a **Genetic Algorithm (GA)** to search the combinatorial space of $2^{128}$ possible feature subsets, identifying the minimal salient feature subset that preserves or improves gas classification accuracy with a **Random Forest ensemble**.

The platform provides a single-page AI laboratory dashboard featuring:
- Live dataset exploration across 10 temporal recording batches (13,910 total samples).
- End-to-end model training pipeline tracking.
- Interactive Genetic Algorithm evolutionary convergence tracking.
- Dynamic 16-sensor selection matrix grid.
- **Software-based manual gas classification interface** requiring **only the GA-selected features**.
- Genuine held-out test sample loader for verification.
- 6-class probability distribution inference with model confidence.
- Comprehensive performance benchmarking (Accuracy, Precision, Recall, F1, 6×6 Confusion Matrices, Classification Reports).
- Audit-grade prediction history tracking.

---

## 🎯 2. Key Objectives & Scientific Motivation

1. **Simulate Biological Olfaction:** Emulate biological cross-reactive olfactory receptor arrays using 16 Figaro MOX sensors across 6 volatile organic compounds (VOCs).
2. **Mitigate Sensor Drift & Redundancy:** Eliminate noisy, redundant channels that degrade long-term classifier robustness.
3. **Bio-Inspired Optimization:** Use evolutionary principles (tournament selection, crossover, bit-flip mutation, and elitism) to reduce feature space by 40%–60%.
4. **Interactive Selected-Feature Inference:** Allow users to manually enter sensor-derived feature values for *strictly the GA-selected features*, or load genuine held-out test samples to test the classifier in real-time.
5. **Rigorous Machine Learning Validation:** Ensure strict stratification, train-only StandardScaler fitting, cross-validation within training partitions, and final evaluation exclusively on unseen test samples.

---

## 💻 3. Technologies & System Requirements

| Layer | Component | Version / Library | Purpose |
|---|---|---|---|
| **Backend** | Python & Flask | Python 3.8+, Flask ≥ 2.3.0 | RESTful API server, state management |
| **Machine Learning** | Scikit-learn | Scikit-learn ≥ 1.3.0 | Random Forest, StratifiedKFold, StandardScaler |
| **Data Processing** | NumPy & Pandas | NumPy ≥ 1.24.0, Pandas ≥ 2.0.0 | High-performance matrix & tabular processing |
| **Visualization** | Plotly.js | Plotly.js 2.27.0 | Dark-themed interactive charts, heatmaps |
| **Frontend UI** | HTML5 / CSS3 / ES6 | Custom Dark Theme Glassmorphism | Responsive single-page application |
| **Typography & Icons** | FontAwesome & Google Fonts | FontAwesome 6.4.0, Plus Jakarta Sans, JetBrains Mono | Modern typography & iconography |

---

## 🎛️ 4. Dataset Architecture & Hardware Sensor Array

The platform utilizes the benchmark **UCI Gas Sensor Array Drift Dataset** collected by Vergara et al. at the University of California San Diego (UCSD):

- **Total Samples:** 13,910 experimental measurements.
- **Recording Horizon:** 36 continuous months (January 2007 to February 2011) partitioned into **10 temporal batches**.
- **Physical Sensors:** 16 Figaro Metal Oxide Semiconductor (MOX) sensors (4 units of each model):
  - **Sensors 1–4 (`TGS2600`):** General air contaminants, Hydrogen ($H_2$), Carbon Monoxide ($CO$).
  - **Sensors 5–8 (`TGS2602`):** Volatile Organic Compounds (VOCs), Ammonia ($NH_3$), Hydrogen Sulfide ($H_2S$).
  - **Sensors 9–12 (`TGS2610`):** Combustible hydrocarbons, LP gases, Propane ($C_3H_8$), Butane ($C_4H_{10}$).
  - **Sensors 13–16 (`TGS2620`):** Alcohol vapors, Ethanol ($C_2H_5OH$), volatile solvent vapors.
- **Extracted Dynamics per Sensor (8 Features Each = 128 Total Features):**
  1. $\Delta R$: Steady-state normalized differential resistance $(R_{\text{max}} - R_0)$.
  2. $|\Delta R|$: Absolute differential resistance magnitude.
  3. $\text{EMAi}_{0.001}$: Exponential moving average of adsorption (rising) phase with $\alpha = 0.001$.
  4. $\text{EMAi}_{0.01}$: Exponential moving average of adsorption (rising) phase with $\alpha = 0.01$.
  5. $\text{EMAi}_{0.1}$: Exponential moving average of adsorption (rising) phase with $\alpha = 0.1$.
  6. $\text{EMAd}_{0.001}$: Exponential moving average of desorption (decaying) phase with $\alpha = 0.001$.
  7. $\text{EMAd}_{0.01}$: Exponential moving average of desorption (decaying) phase with $\alpha = 0.01$.
  8. $\text{EMAd}_{0.1}$: Exponential moving average of desorption (decaying) phase with $\alpha = 0.1$.
- **6 Target Gas Analytes:**
  1. **Ethanol ($C_2H_5OH$):** Alcohol analyte.
  2. **Ethylene ($C_2H_4$):** Plant hormone and ripening agent.
  3. **Ammonia ($NH_3$):** Industrial refrigerant and hazardous alkali.
  4. **Acetaldehyde ($C_2H_4O$):** Volatile aldehyde and pollutant.
  5. **Acetone ($C_3H_6O$):** Key ketone biomarker in exhaled breath.
  6. **Toluene ($C_7H_8$):** Aromatic VOC in industrial solvents.

---

## 🏛️ 5. System Architecture & 9-Section Dashboard

```text
               UCI Gas Sensor Array Drift Dataset (10 Batches)
                                     │
                                     ▼
                      Data Loading & LibSVM Parser
                                     │
                                     ▼
                Stratified Train/Test Split (80% Train, 20% Test)
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
      Baseline Random Forest                    Genetic Algorithm
         (All 128 Features)                 (Surrogate CV on Train Set)
                  │                                     │
                  │                                     ▼
                  │                        Optimal Binary Chromosome
                  │                                     │
                  │                                     ▼
                  │                         GA-Optimized Random Forest
                  │                           (Selected Features Only)
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     ▼
                  Held-out Test Evaluation (Accuracy, F1, CM)
                                     │
                                     ▼
         Interactive Dashboard & Selected-Feature Gas Prediction
```

### Dashboard Sections
1. **Overview:** Summary KPIs, 6-analyte distribution chart, 16-sensor architecture, and pipeline readiness status.
2. **Dataset Explorer:** Batch sample distributions, gas balance, searchable/filterable dataset table, and single-sample 128-feature modal inspector.
3. **Model Training:** 7-stage sequential pipeline execution tracker, animated progress indicator, and baseline vs. optimized comparison banner.
4. **Genetic Algorithm:** Interactive hyperparameter sliders (population, generations, mutation, crossover, penalty, seed), preset profiles, convergence graph, and 128-bit chromosome viewer.
5. **Feature Selection:** Sensor retention bar chart, 16-sensor matrix cards with mini-feature indicators, and detailed feature inspector modal.
6. **Gas Prediction:** Software-based manual input form generating fields **only for GA-selected features**, genuine test sample loader, probability distribution bar chart, and ground truth match verifier.
7. **Performance Analysis:** Baseline vs. GA-optimized accuracy, precision, recall, F1, 6×6 confusion matrix heatmaps, and per-class classification report.
8. **Prediction History:** Session inference log with sample IDs, timestamps, predicted gases, confidence scores, and audit modals.
9. **About BioSenseGA:** Academic engineering overview, theoretical derivations, sensing chemistry, and references.

---

## 🧬 6. Genetic Algorithm Mathematical Formulation

### 1. Chromosome Representation
A candidate feature subset is encoded as a 128-bit binary vector:
$$\mathbf{c} = [c_1, c_2, \dots, c_{128}], \quad c_i \in \{0, 1\}$$
where $c_i = 1$ denotes feature selection, and $c_i = 0$ denotes feature elimination.

### 2. Fitness Function
To prevent selecting excessive features, the fitness function incorporates a parsimony penalty $\lambda$:
$$\text{Fitness}(\mathbf{c}) = \text{Accuracy}_{\text{CV}}(\mathbf{c}) - \lambda \times \left( \frac{\sum_{i=1}^{128} c_i}{128} \right)$$
- $\text{Accuracy}_{\text{CV}}(\mathbf{c})$: Mean classification accuracy evaluated using 2-fold stratified cross-validation on the training set with a surrogate Random Forest.
- $\lambda$: Feature penalty coefficient (default $\lambda = 0.010$).
- If $\sum c_i = 0$, $\text{Fitness}(\mathbf{c}) = 0.0$.

### 3. Evolutionary Operators
- **Selection:** Tournament selection ($k=3$) choosing the fittest candidate from $k$ randomly chosen chromosomes.
- **Crossover:** Single-point crossover applied with probability $p_c = 0.80$. A random cut locus $l \in [1, 127]$ is selected:
  $$\mathbf{c}_1' = [p_{1,1..l}, p_{2,l+1..128}], \quad \mathbf{c}_2' = [p_{2,1..l}, p_{1,l+1..128}]$$
- **Mutation:** Bit-flip mutation applied independently to each locus with probability $p_m = 0.05$:
  $$c_i' = 1 - c_i \quad \text{if } U(0,1) < p_m$$
- **Elitism:** The individual with the absolute best fitness score is preserved unaltered into the next generation.
- **Memoization Cache:** Chromosome byte vectors are cached in memory to eliminate redundant classifier fits.

---

## ⚙️ 7. Installation & Setup Instructions

### Prerequisites
- Python 3.8 to 3.12 installed on your system.
- Git (optional, for cloning).

### Setup Steps
```bash
# 1. Open a terminal and navigate to the project directory
cd BioSenseGA_Complete_Project

# 2. (Optional but recommended) Create and activate a Python virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Verify batch data files
# Ensure data/batch1.dat through data/batch10.dat exist in the data/ folder.
```

---

## 🚀 8. How to Run the Application

```bash
# Start the Flask web application
python app.py
```

The terminal will confirm:
```text
======================================================================
  BioSenseGA – Evolutionary Gas Sensor Array Optimization Platform
  Engineering Lab Project: Random Forest & Genetic Algorithm
======================================================================
  Server running at http://127.0.0.1:5000
======================================================================
```

Open your web browser and navigate to:  
👉 **`http://127.0.0.1:5000`**

---

## 📖 9. Step-by-Step User Guide

### A. Loading the Dataset
1. In the sidebar, click **Dataset Explorer** (or use the Overview page).
2. Click the **Load / Reload Dataset** button.
3. The backend will parse `batch1.dat` through `batch10.dat`, extract 13,910 rows, perform an 80/20 stratified split, and fit `StandardScaler` strictly on the training partition.
4. The batch distribution and gas class charts will render automatically. You can filter the dataset table by batch or gas class, or search by sample index.

### B. Training the Models
1. In the sidebar, navigate to **Model Training**.
2. Click **Train Models (Full Pipeline)**.
3. The 7-stage pipeline visualizer will track execution:
   - *Stage 1:* Preprocess and scale training features.
   - *Stage 2:* Stratified 80/20 train/test split.
   - *Stage 3:* Fit Baseline Random Forest on all 128 features.
   - *Stage 4:* Execute Genetic Algorithm feature optimization.
   - *Stage 5:* Extract optimal feature selection chromosome.
   - *Stage 6:* Fit GA-Optimized Random Forest on selected features.
   - *Stage 7:* Evaluate both models on held-out test data.
4. Upon completion, a performance summary banner displays baseline accuracy, optimized accuracy, delta, feature reduction %, and training duration.

### C. Genetic Algorithm Optimization
1. In the sidebar, navigate to **Genetic Algorithm**.
2. Adjust hyperparameters as desired:
   - **Population Size:** 10 to 80 (Preset: Fast = 15, Balanced = 30, Deep = 50).
   - **Generations:** 4 to 40 (Preset: Fast = 6, Balanced = 15, Deep = 25).
   - **Mutation Rate ($p_m$):** 0.01 to 0.20 (Default: 0.05).
   - **Crossover Rate ($p_c$):** 0.50 to 0.95 (Default: 0.80).
   - **Feature Penalty ($\lambda$):** 0.000 to 0.050 (Default: 0.010).
   - **Random Seed:** Any integer for reproducibility (Default: 42).
3. Click **Run Genetic Optimization**.
4. Observe the interactive **Convergence Graph** (Best Fitness vs. Mean Fitness per generation) and inspect the **128-Bit Chromosome Mask** (Green = 1, Dark = 0).

### D. Manual Feature Entry & Gas Prediction
Because physical hardware is simulated, BioSenseGA features a dedicated software classification workbench:
1. In the sidebar, navigate to **Gas Prediction**.
2. Notice the dynamic form generation:
   - If the GA selected 45 features, **only 45 input fields** are displayed across the active sensors.
   - Eliminated features are automatically excluded.
   - Each input shows the physical sensor name, sensor model (e.g., `TGS2600`), feature code (e.g., `S1_DR`), mathematical symbol ($\Delta R$), and physical description.
3. **Option A – Load Genuine Test Sample:**
   - Click **Load Sample Values (Test Set)**.
   - A genuine held-out test sample is retrieved. The inputs are automatically filled with the unscaled feature values.
   - The sample index and batch ID are displayed. You can freely modify any value to observe how perturbation affects classification.
4. **Option B – Manual Feature Entry:**
   - Enter your own sensor-derived numeric values into the active fields.
5. Click **Predict Gas Class**.
6. The backend validates that all required inputs are present, applies the fitted scaler parameters:
   $$\text{scaled\_value} = \frac{\text{raw\_value} - \mu_i}{\sigma_i}$$
   and executes prediction through the GA-optimized Random Forest.

### E. Interpreting Predictions & Performance Results
- **Result Panel:** Shows predicted gas name, chemical formula, analyte category, and probabilistic model confidence.
- **Analyte Probabilities Bar Chart:** Displays individual probabilities across all 6 target gases.
- **Ground Truth Verifier:** When loaded from a test sample, displays a green **Verified Match** badge comparing prediction with actual ground truth.
- **Performance Benchmarking Section:**
  - View side-by-side **6×6 Confusion Matrices** for both baseline and optimized models.
  - Review per-class Precision, Recall, and F1-Scores.
  - Review the formal **Academic Experimental Conclusion**.

---

## 🛡️ 10. Machine Learning Validation & Anti-Leakage Protocol

To ensure academic and scientific validity:
1. **Stratified Splitting:** Splitting (80% train, 20% test) preserves class balance across all 6 gas analytes.
2. **Strict Scaler Isolation:** `StandardScaler` is fitted **exclusively on `X_train`**. Unseen test samples and manual inputs are standardized using the stored `mean_` and `scale_` arrays of `X_train`.
3. **No Test Set Leakage in GA:** The Genetic Algorithm evaluates candidate chromosomes exclusively on a stratified 2-fold cross-validation split within `X_train`. The test set is completely isolated during feature selection.
4. **Consistent Feature Ordering:** The exact column indices selected by the GA are maintained in the same order during both model training and live prediction.
5. **No Zero-Filling:** Missing manual inputs are strictly rejected with descriptive error notifications; missing fields are never silently filled with zeros.

---

## ⚠️ 11. Project Limitations & Engineering Discussion

- **Computational Complexity:** Training a surrogate Random Forest across multiple generations is computationally intensive; a representative training subset (1,500 samples) and parallel CPU threads (`n_jobs=-1`) are utilized to achieve rapid convergence (~15 seconds).
- **Stochastic Variation:** Due to the probabilistic nature of crossover and mutation, different random seeds will converge to slightly different feature subsets with comparable accuracy.
- **Single-Objective Formulation:** The fitness function linearly combines accuracy and feature count. In production, multi-objective evolutionary algorithms (e.g., NSGA-II) could generate a Pareto-optimal frontier.
- **Simulated Hardware:** The platform operates on experimental benchmark data; physical gas detection requires dedicated sensor chamber hardware, gas flow controllers, and analog-to-digital converters (ADCs).

---

## 📚 12. Academic Syllabus Connection & References

### Syllabus Alignment
- **Unit III (Artificial Nose / Biosensors / Sensor Arrays):** Demonstrates metal oxide semiconductor (MOX) cross-reactive sensing principles, transient exponential moving average (EMA) signal dynamics, and multivariant pattern recognition.
- **Unit IV (Bio-Sensing Genetic Algorithms):** Formulates sensor signal optimization as an evolutionary search problem using binary chromosome encoding, tournament selection, crossover, bit-flip mutation, and elitism.

### Literature References
1. **Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., & Huerta, R. (2012).** Chemical gas sensor drift compensation using classifier ensembles. *Sensors and Actuators B: Chemical*, 166-167, 320–329.
2. **UCI Machine Learning Repository.** *Gas Sensor Array Drift Dataset*. [https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset)
3. **Holland, J. H. (1975).** *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
4. **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5–32.
