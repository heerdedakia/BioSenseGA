"""
BioSenseGA – Evolutionary Gas Classification & E-Nose Optimization Platform
Academic Engineering Lab Project
Dedicated to Random Forest Classifier Optimization using Genetic Algorithms.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
import sys
import time
import random
import datetime
import threading
import traceback

# Append current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    GAS_NAMES, GAS_METADATA, SENSOR_TYPES, SENSOR_METADATA,
    FEATURE_TYPES, FEATURE_METADATA, FEATURE_NAMES,
    SENSOR_FEATURE_MAPPING, DEFAULT_GA_PARAMS, N_FEATURES, TEST_SIZE
)
from backend.data_loader import load_all_batches, get_dataset_info
from backend.preprocessing import preprocess_data
from backend.genetic_algorithm import GeneticAlgorithm
from backend.classifier import train_and_evaluate, get_classifier
from backend.utils import (
    generate_conclusion, get_sensor_selection_summary,
    get_sensor_level_selection, get_detailed_sensor_groups, format_percentage
)

app = Flask(__name__)

# State lock to ensure thread safety during model training
state_lock = threading.Lock()

# Centralized application state
app_state = {
    'data_loaded': False,
    'is_training': False,
    'df': None,
    'processed_data': None,
    'dataset_info': None,
    'baseline_results': None,
    'ga_results': None,
    'best_chromosome': None,
    'ga_history': None,
    'scaler': None,
    'label_encoder': None,
    'baseline_clf': None,
    'optimized_clf': None,
    'selected_indices': None,
    'selected_feature_names': None,
    'sensor_groups': None,
    'training_metadata': None,
    'prediction_history': [],
    'prediction_counter': 0
}


@app.route('/')
def index():
    """Serve the single-page dashboard application."""
    return render_template('index.html')


@app.route('/api/model_status', methods=['GET'])
def get_model_status():
    """Return the current system and model readiness status."""
    try:
        data_loaded = app_state['data_loaded']
        is_training = app_state['is_training']
        baseline_trained = app_state['baseline_clf'] is not None
        ga_completed = app_state['best_chromosome'] is not None
        optimized_trained = app_state['optimized_clf'] is not None

        n_samples = app_state['dataset_info']['n_samples'] if data_loaded else 0
        n_train = app_state['processed_data']['n_train'] if data_loaded else 0
        n_test = app_state['processed_data']['n_test'] if data_loaded else 0

        selected_count = len(app_state['selected_indices']) if ga_completed else None
        reduction_pct = (
            ((N_FEATURES - selected_count) / N_FEATURES * 100.0)
            if (ga_completed and selected_count is not None)
            else None
        )

        baseline_acc = (
            app_state['baseline_results']['accuracy']
            if (baseline_trained and app_state['baseline_results'])
            else None
        )
        opt_acc = (
            app_state['ga_results']['accuracy']
            if (optimized_trained and app_state['ga_results'])
            else None
        )

        return jsonify({
            "success": True,
            "status": {
                "data_loaded": data_loaded,
                "is_training": is_training,
                "baseline_trained": baseline_trained,
                "ga_completed": ga_completed,
                "optimized_trained": optimized_trained,
                "n_samples": n_samples,
                "n_train": n_train,
                "n_test": n_test,
                "n_sensors": 16,
                "total_features": N_FEATURES,
                "n_classes": 6,
                "selected_features_count": selected_count,
                "feature_reduction_pct": round(reduction_pct, 2) if reduction_pct is not None else None,
                "baseline_accuracy": round(baseline_acc, 4) if baseline_acc is not None else None,
                "optimized_accuracy": round(opt_acc, 4) if opt_acc is not None else None,
                "training_metadata": app_state['training_metadata'],
                "prediction_count": len(app_state['prediction_history'])
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/load_dataset', methods=['POST'])
def load_dataset():
    """Load, inspect, and preprocess the 10 UCI Gas Sensor Array batches."""
    req_data = request.json or {}
    data_dir = req_data.get('data_dir', 'data')

    try:
        # Load raw batch files
        df = load_all_batches(data_dir)
        dataset_info = get_dataset_info(df)

        # Preprocess: stratified split and train-only StandardScaler fitting
        processed_data = preprocess_data(df, test_size=TEST_SIZE, random_state=42)

        with state_lock:
            app_state['df'] = df
            app_state['dataset_info'] = dataset_info
            app_state['processed_data'] = processed_data
            app_state['scaler'] = processed_data['scaler']
            app_state['label_encoder'] = processed_data['label_encoder']
            app_state['data_loaded'] = True
            # Reset dependent model state if reloaded
            app_state['baseline_clf'] = None
            app_state['optimized_clf'] = None
            app_state['best_chromosome'] = None
            app_state['ga_results'] = None
            app_state['baseline_results'] = None
            app_state['selected_indices'] = None
            app_state['selected_feature_names'] = None
            app_state['sensor_groups'] = None
            app_state['training_metadata'] = None

        return jsonify({
            "success": True,
            "dataset_info": dataset_info,
            "preprocessing": {
                "n_train": processed_data['n_train'],
                "n_test": processed_data['n_test'],
                "test_ratio": TEST_SIZE
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to load dataset: {str(e)}"}), 500


@app.route('/api/dataset_table', methods=['GET'])
def get_dataset_table():
    """Provide paginated, filtered dataset samples for the Dataset Explorer table."""
    if not app_state['data_loaded']:
        return jsonify({"success": False, "error": "Dataset not loaded yet."}), 400

    try:
        page = max(1, int(request.args.get('page', 1)))
        page_size = min(100, max(5, int(request.args.get('page_size', 15))))
        batch_filter = request.args.get('batch')
        gas_filter = request.args.get('gas_class')
        search_query = request.args.get('search', '').strip()

        df = app_state['df']
        filtered_df = df

        if batch_filter and batch_filter.isdigit():
            filtered_df = filtered_df[filtered_df['batch_id'] == int(batch_filter)]

        if gas_filter and gas_filter.isdigit():
            filtered_df = filtered_df[filtered_df['class_label'] == int(gas_filter)]

        if search_query:
            if search_query.isdigit():
                # Search by exact sample index
                target_idx = int(search_query)
                if 0 <= target_idx < len(df):
                    filtered_df = df.iloc[[target_idx]]
                else:
                    filtered_df = df.iloc[0:0]

        total_records = len(filtered_df)
        total_pages = max(1, (total_records + page_size - 1) // page_size)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_records)

        records_slice = filtered_df.iloc[start_idx:end_idx]

        samples = []
        for idx, row in records_slice.iterrows():
            class_id = int(row['class_label'])
            gas_name = GAS_NAMES.get(class_id, f"Class {class_id}")
            samples.append({
                "sample_id": int(idx),
                "batch_id": int(row['batch_id']),
                "class_label": class_id,
                "gas_name": gas_name,
                "s1_dr": round(float(row.get('feature_1', 0.0)), 4),
                "s1_abs_dr": round(float(row.get('feature_2', 0.0)), 4),
                "s5_dr": round(float(row.get('feature_33', 0.0)), 4),
                "s9_dr": round(float(row.get('feature_65', 0.0)), 4),
                "s13_dr": round(float(row.get('feature_97', 0.0)), 4),
            })

        return jsonify({
            "success": True,
            "samples": samples,
            "total_records": total_records,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/dataset_sample_detail', methods=['GET'])
def get_dataset_sample_detail():
    """Inspect full 128 features and sensor breakdown of a single sample."""
    if not app_state['data_loaded']:
        return jsonify({"success": False, "error": "Dataset not loaded"}), 400

    try:
        sample_index = int(request.args.get('sample_index', 0))
        df = app_state['df']
        if sample_index < 0 or sample_index >= len(df):
            return jsonify({"success": False, "error": "Invalid sample index"}), 400

        row = df.iloc[sample_index]
        feature_cols = [f'feature_{i}' for i in range(1, 129)]
        feature_values = [float(row[col]) for col in feature_cols]
        class_label = int(row['class_label'])

        # Group by sensor
        sensors_breakdown = []
        for s_idx in range(16):
            s_id = s_idx + 1
            s_type = SENSOR_TYPES[s_idx]
            s_vals = feature_values[s_idx * 8:(s_idx + 1) * 8]
            sensors_breakdown.append({
                "sensor_id": s_id,
                "sensor_name": f"Sensor {s_id}",
                "sensor_type": s_type,
                "mean_response": round(float(np.mean(s_vals)), 4),
                "feature_values": [round(v, 4) for v in s_vals]
            })

        return jsonify({
            "success": True,
            "sample_index": sample_index,
            "batch_id": int(row['batch_id']),
            "class_label": class_label,
            "gas_name": GAS_NAMES.get(class_label, f"Class {class_label}"),
            "sensors": sensors_breakdown
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/train_models', methods=['POST'])
@app.route('/api/run_ga', methods=['POST'])
def train_models_and_run_ga():
    """
    Executes the complete machine learning and feature selection pipeline:
    1. Verify dataset loaded
    2. Train baseline Random Forest with all 128 features
    3. Execute Genetic Algorithm on training cross-validation
    4. Extract optimal chromosome and map selected sensor features
    5. Train GA-optimized Random Forest on selected features
    6. Evaluate both models on held-out test data
    """
    # STEP 1: Ensure dataset is loaded and preprocessed
    if not app_state['data_loaded']:
        try:
            df = load_all_batches('data')
            dataset_info = get_dataset_info(df)
            processed_data = preprocess_data(df, test_size=TEST_SIZE, random_state=42)
            app_state['df'] = df
            app_state['dataset_info'] = dataset_info
            app_state['processed_data'] = processed_data
            app_state['scaler'] = processed_data['scaler']
            app_state['label_encoder'] = processed_data['label_encoder']
            app_state['data_loaded'] = True
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to load dataset: {str(e)}"}), 500

    # Prevent concurrent training requests
    with state_lock:
        if app_state['is_training']:
            return jsonify({
                "success": False,
                "error": "Model training is already in progress. Please wait for completion."
            }), 429
        app_state['is_training'] = True

    try:
        start_pipeline_time = time.time()
        req_data = request.json or {}

        population_size = int(req_data.get('population_size', 20))
        n_generations = int(req_data.get('n_generations', 8))
        mutation_rate = float(req_data.get('mutation_rate', 0.05))
        crossover_rate = float(req_data.get('crossover_rate', 0.8))
        feature_penalty = float(req_data.get('feature_penalty', 0.01))
        random_seed = int(req_data.get('random_seed', 42))

        X_train = app_state['processed_data']['X_train']
        X_test = app_state['processed_data']['X_test']
        y_train = app_state['processed_data']['y_train']
        y_test = app_state['processed_data']['y_test']
        class_names = [GAS_NAMES[i] for i in sorted(GAS_NAMES.keys())]

        # STEP 1: Train Baseline Random Forest (all 128 features)
        t_base_start = time.time()
        baseline_results = train_and_evaluate(
            'Random Forest', X_train, X_test, y_train, y_test,
            selected_features=None, class_names=class_names
        )
        t_base_dur = time.time() - t_base_start

        # STEP 2: Execute Genetic Algorithm
        t_ga_start = time.time()
        ga = GeneticAlgorithm(
            n_features=N_FEATURES,
            population_size=population_size,
            n_generations=n_generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            feature_penalty=feature_penalty,
            random_seed=random_seed,
            classifier_name='Random Forest',
            X_train=X_train,
            y_train=y_train
        )
        best_chromosome, best_fitness, history = ga.evolve()
        t_ga_dur = time.time() - t_ga_start

        # STEP 3: Identify Selected Features
        selected_indices = [int(i) for i, bit in enumerate(best_chromosome) if bit == 1]
        if len(selected_indices) == 0:
            # Fallback guarantee: pick highest variance feature
            selected_indices = [0]
            best_chromosome[0] = 1

        selected_feature_names = [FEATURE_NAMES[i] for i in selected_indices]
        feature_reduction_pct = ((N_FEATURES - len(selected_indices)) / N_FEATURES) * 100.0

        # STEP 4: Train GA-Optimized Random Forest
        t_opt_start = time.time()
        optimized_results = train_and_evaluate(
            'Random Forest', X_train, X_test, y_train, y_test,
            selected_features=selected_indices, class_names=class_names
        )
        t_opt_dur = time.time() - t_opt_start

        # Detailed Sensor Architecture & Feature Breakdown
        sensor_groups = get_detailed_sensor_groups(best_chromosome)
        sensor_summary = get_sensor_selection_summary(best_chromosome, FEATURE_NAMES)
        sensor_level = get_sensor_level_selection(best_chromosome)
        conclusion = generate_conclusion(baseline_results, optimized_results, N_FEATURES, len(selected_indices))

        total_pipeline_time = time.time() - start_pipeline_time

        training_metadata = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "population_size": population_size,
            "n_generations": n_generations,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
            "feature_penalty": feature_penalty,
            "random_seed": random_seed,
            "baseline_training_time": round(t_base_dur, 2),
            "ga_optimization_time": round(t_ga_dur, 2),
            "optimized_training_time": round(t_opt_dur, 2),
            "total_duration": round(total_pipeline_time, 2)
        }

        with state_lock:
            app_state['baseline_results'] = baseline_results
            app_state['ga_results'] = optimized_results
            app_state['best_chromosome'] = best_chromosome.tolist() if hasattr(best_chromosome, 'tolist') else list(best_chromosome)
            app_state['ga_history'] = history
            app_state['baseline_clf'] = baseline_results['model']
            app_state['optimized_clf'] = optimized_results['model']
            app_state['selected_indices'] = selected_indices
            app_state['selected_feature_names'] = selected_feature_names
            app_state['sensor_groups'] = sensor_groups
            app_state['training_metadata'] = training_metadata
            app_state['is_training'] = False

        return jsonify({
            "success": True,
            "pipeline_time": round(total_pipeline_time, 2),
            "classifier_name": "Random Forest Ensemble",
            "baseline": {
                "accuracy": round(baseline_results['accuracy'], 5),
                "precision": round(baseline_results['precision'], 5),
                "recall": round(baseline_results['recall'], 5),
                "f1_score": round(baseline_results['f1_score'], 5),
                "confusion_matrix": baseline_results['confusion_matrix'],
                "per_class_report": baseline_results['per_class_report'],
                "training_time": round(baseline_results['training_time'], 3),
                "n_features": N_FEATURES
            },
            "optimized": {
                "accuracy": round(optimized_results['accuracy'], 5),
                "precision": round(optimized_results['precision'], 5),
                "recall": round(optimized_results['recall'], 5),
                "f1_score": round(optimized_results['f1_score'], 5),
                "confusion_matrix": optimized_results['confusion_matrix'],
                "per_class_report": optimized_results['per_class_report'],
                "training_time": round(optimized_results['training_time'], 3),
                "n_features": len(selected_indices),
                "selected_features": selected_indices,
                "selected_feature_names": selected_feature_names,
                "feature_reduction_pct": round(feature_reduction_pct, 2)
            },
            "convergence_history": history,
            "sensor_groups": sensor_groups,
            "sensor_selection": sensor_level,
            "sensor_summary": sensor_summary,
            "best_fitness": round(float(best_fitness), 5),
            "conclusion": conclusion,
            "metadata": training_metadata
        })
    except Exception as e:
        with state_lock:
            app_state['is_training'] = False
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Training failed: {str(e)}"}), 500


@app.route('/api/selected_features', methods=['GET'])
def get_selected_features():
    """Return the list, count, and sensor groupings of the GA-selected features."""
    if not app_state['best_chromosome'] or not app_state['selected_indices']:
        return jsonify({
            "success": False,
            "error": "Genetic Algorithm has not been executed yet. Train models first."
        }), 400

    try:
        best_chromosome = app_state['best_chromosome']
        selected_indices = app_state['selected_indices']
        selected_names = app_state['selected_feature_names']
        sensor_groups = app_state['sensor_groups'] or get_detailed_sensor_groups(best_chromosome)

        # Build list of selected features with sensor info
        selected_features_list = []
        for idx in selected_indices:
            sensor_id = (idx // 8) + 1
            sensor_type = SENSOR_TYPES[sensor_id - 1]
            ft_code = FEATURE_TYPES[idx % 8]
            ft_meta = FEATURE_METADATA.get(ft_code, {})
            selected_features_list.append({
                "feature_index": int(idx),
                "feature_name": FEATURE_NAMES[idx],
                "sensor_id": sensor_id,
                "sensor_name": f"Sensor {sensor_id}",
                "sensor_type": sensor_type,
                "feature_code": ft_code,
                "feature_title": ft_meta.get('name', ft_code),
                "symbol": ft_meta.get('symbol', ft_code),
                "description": ft_meta.get('description', '')
            })

        feature_reduction_pct = ((N_FEATURES - len(selected_indices)) / N_FEATURES) * 100.0

        return jsonify({
            "success": True,
            "n_selected": len(selected_indices),
            "n_total": N_FEATURES,
            "feature_reduction_pct": round(feature_reduction_pct, 2),
            "selected_indices": selected_indices,
            "selected_feature_names": selected_names,
            "selected_features": selected_features_list,
            "sensor_groups": sensor_groups
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/get_test_sample', methods=['GET', 'POST'])
@app.route('/api/get_sample', methods=['GET', 'POST'])
def get_test_sample():
    """
    Retrieves a genuine sample from the held-out test dataset.
    Supports filtering by specific gas analyte (e.g., Ammonia, Acetone, etc.).
    Returns both the values for the GA-selected features and all features.
    """
    if not app_state['data_loaded']:
        return jsonify({"success": False, "error": "Dataset not loaded"}), 400

    try:
        processed = app_state['processed_data']
        n_test = processed['n_test']
        X_test_raw = processed['X_test_raw']
        test_meta = processed['test_meta']
        test_indices = processed['test_indices']

        req_data = request.get_json(silent=True) or {}
        req_index = request.args.get('sample_index') or req_data.get('sample_index')
        req_gas = request.args.get('gas') or req_data.get('gas') or request.args.get('gas_class') or req_data.get('gas_class')

        if req_index is not None and str(req_index).isdigit() and 0 <= int(req_index) < n_test:
            sample_rel_idx = int(req_index)
        elif req_gas and str(req_gas).strip().lower() not in ['', 'all', 'any', 'random']:
            target_str = str(req_gas).strip().lower()
            matching_indices = []
            for i in range(n_test):
                row_label = int(test_meta.iloc[i]['class_label'])
                row_gas = GAS_NAMES.get(row_label, "").lower()
                if target_str in row_gas or target_str == str(row_label):
                    matching_indices.append(i)
            if matching_indices:
                sample_rel_idx = random.choice(matching_indices)
            else:
                sample_rel_idx = random.randint(0, n_test - 1)
        else:
            sample_rel_idx = random.randint(0, n_test - 1)

        raw_feature_values = X_test_raw[sample_rel_idx].tolist()
        global_df_idx = test_indices[sample_rel_idx]
        meta_row = test_meta.iloc[sample_rel_idx]

        true_label = int(meta_row['class_label'])
        true_gas = GAS_NAMES.get(true_label, f"Class {true_label}")
        batch_id = int(meta_row['batch_id'])

        # Compute sensor averages for visual preview
        sensor_averages = []
        for s_idx in range(16):
            s_vals = raw_feature_values[s_idx * 8:(s_idx + 1) * 8]
            sensor_averages.append(round(float(np.mean(s_vals)), 4))

        # Build map of selected features with their raw values
        selected_features_data = {}
        if app_state['selected_indices']:
            for idx in app_state['selected_indices']:
                val = float(raw_feature_values[idx])
                ft_name = FEATURE_NAMES[idx]
                selected_features_data[ft_name] = round(val, 6)
                selected_features_data[str(idx)] = round(val, 6)

        return jsonify({
            "success": True,
            "test_sample_index": sample_rel_idx,
            "global_sample_index": global_df_idx,
            "batch_id": batch_id,
            "true_label": true_label,
            "true_gas": true_gas,
            "sensor_averages": sensor_averages,
            "selected_values": selected_features_data,
            "sample_values": raw_feature_values,
            "all_values": {FEATURE_NAMES[i]: round(raw_feature_values[i], 6) for i in range(N_FEATURES)},
            "total_test_samples": n_test
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/predict_selected', methods=['POST'])
def predict_selected():
    """
    CRITICAL TECHNICAL ENDPOINT:
    Predicts gas class from manually entered or sample-loaded feature values
    corresponding strictly to the GA-selected features.

    Applies exact training StandardScaler parameters:
    scaled_value = (input_value - scaler.mean_[feature_index]) / scaler.scale_[feature_index]
    """
    # STEP 1: Ensure dataset is loaded and preprocessed
    if not app_state['data_loaded']:
        try:
            df = load_all_batches('data')
            dataset_info = get_dataset_info(df)
            processed_data = preprocess_data(df, test_size=TEST_SIZE, random_state=42)
            app_state['df'] = df
            app_state['dataset_info'] = dataset_info
            app_state['processed_data'] = processed_data
            app_state['scaler'] = processed_data['scaler']
            app_state['label_encoder'] = processed_data['label_encoder']
            app_state['data_loaded'] = True
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to load dataset: {str(e)}"}), 500

    if app_state['optimized_clf'] is None or app_state['selected_indices'] is None:
        return jsonify({
            "success": False,
            "error": "GA-Optimized Random Forest model is not trained yet. Please train models first."
        }), 400

    req_data = request.json or {}
    input_features = req_data.get('features') or {}
    input_source = req_data.get('input_source', 'Manual Input')
    sample_id = req_data.get('sample_id')
    ground_truth = req_data.get('ground_truth')

    scaler = app_state['scaler']
    le = app_state['label_encoder']
    optimized_clf = app_state['optimized_clf']
    selected_indices = app_state['selected_indices']

    if not isinstance(input_features, dict):
        return jsonify({
            "success": False,
            "error": "Input format error. Expected JSON object mapping feature names or indices to numerical values."
        }), 400

    # Validate that every selected feature is provided with a valid numeric float
    missing_features = []
    invalid_features = []
    raw_selected_vector = []

    for idx in selected_indices:
        feat_name = FEATURE_NAMES[idx]
        val = None

        # Check by feature name first, then by string index
        if feat_name in input_features:
            val = input_features[feat_name]
        elif str(idx) in input_features:
            val = input_features[str(idx)]
        elif idx in input_features:
            val = input_features[idx]

        if val is None or val == "":
            missing_features.append(feat_name)
            continue

        try:
            val_float = float(val)
            if np.isnan(val_float) or np.isinf(val_float):
                invalid_features.append(f"{feat_name} (NaN or Inf)")
            else:
                raw_selected_vector.append((idx, val_float))
        except (ValueError, TypeError):
            invalid_features.append(f"{feat_name} ('{val}')")

    if missing_features:
        return jsonify({
            "success": False,
            "error": f"Missing required values for {len(missing_features)} selected feature(s): {', '.join(missing_features[:5])}" +
                     (f" and {len(missing_features)-5} more." if len(missing_features) > 5 else "."),
            "missing_features": missing_features
        }), 400

    if invalid_features:
        return jsonify({
            "success": False,
            "error": f"Invalid numeric format for feature(s): {', '.join(invalid_features[:5])}.",
            "invalid_features": invalid_features
        }), 400

    if len(raw_selected_vector) != len(selected_indices):
        return jsonify({
            "success": False,
            "error": f"Dimension mismatch: expected {len(selected_indices)} selected features, received {len(raw_selected_vector)}."
        }), 400

    try:
        # Standardize strictly using the fitted training scaler parameters for selected features
        scaled_selected_vector = []
        for idx, raw_val in raw_selected_vector:
            mean = scaler.mean_[idx]
            scale = scaler.scale_[idx]
            scaled_val = (raw_val - mean) / scale
            scaled_selected_vector.append(scaled_val)

        # Shape vector for model prediction: (1, n_selected)
        X_input = np.array(scaled_selected_vector, dtype=np.float64).reshape(1, -1)

        # Predict gas class
        pred_encoded = optimized_clf.predict(X_input)[0]
        pred_label = int(le.inverse_transform([pred_encoded])[0])
        pred_gas = GAS_NAMES.get(pred_label, f"Class {pred_label}")

        # Compute probabilities across all six classes
        probabilities = {}
        try:
            proba = optimized_clf.predict_proba(X_input)[0]
            classes = le.classes_
            for i, cls in enumerate(classes):
                gas_name = GAS_NAMES.get(int(cls), f"Class {cls}")
                probabilities[gas_name] = round(float(proba[i]), 5)
        except AttributeError:
            for gas_name in GAS_NAMES.values():
                probabilities[gas_name] = 1.0 if gas_name == pred_gas else 0.0

        confidence = probabilities.get(pred_gas, 0.0)

        # Record prediction in history
        with state_lock:
            app_state['prediction_counter'] += 1
            pred_id = f"PRED-{app_state['prediction_counter']:04d}"
            pred_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            history_entry = {
                "id": pred_id,
                "timestamp": pred_timestamp,
                "predicted_gas": pred_gas,
                "confidence": confidence,
                "probabilities": probabilities,
                "model_used": f"GA-Optimized Random Forest ({len(selected_indices)} features)",
                "n_features_used": len(selected_indices),
                "input_source": input_source,
                "sample_id": sample_id,
                "ground_truth": ground_truth
            }
            app_state['prediction_history'].insert(0, history_entry)
            # Cap history size at 100 entries
            if len(app_state['prediction_history']) > 100:
                app_state['prediction_history'].pop()

        return jsonify({
            "success": True,
            "prediction_id": pred_id,
            "predicted_gas": pred_gas,
            "predicted_class": pred_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "features_used": len(selected_indices),
            "model_type": f"GA-Optimized Random Forest ({len(selected_indices)} features)",
            "timestamp": pred_timestamp,
            "input_source": input_source,
            "ground_truth": ground_truth,
            "gas_metadata": GAS_METADATA.get(pred_gas, {})
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500


@app.route('/api/predict', methods=['POST'])
def predict_full_vector():
    """
    Robust prediction endpoint supporting:
    1. Selected features dictionary (from dynamic manual inputs)
    2. Full 128-element feature array
    3. Dataset sample index
    """
    if not app_state['data_loaded']:
        # Auto-load dataset if not yet in memory
        try:
            df = load_all_batches('data')
            dataset_info = get_dataset_info(df)
            processed_data = preprocess_data(df, test_size=TEST_SIZE, random_state=42)
            app_state['df'] = df
            app_state['dataset_info'] = dataset_info
            app_state['processed_data'] = processed_data
            app_state['scaler'] = processed_data['scaler']
            app_state['label_encoder'] = processed_data['label_encoder']
            app_state['data_loaded'] = True
        except Exception as e:
            return jsonify({"success": False, "error": f"Dataset not loaded: {str(e)}"}), 400

    req_data = request.json or {}
    
    # If client passed selected features dictionary, dispatch directly to predict_selected()
    if 'features' in req_data and app_state['optimized_clf'] is not None:
        return predict_selected()

    sample_index = req_data.get('sample_index')
    sensor_values = req_data.get('sensor_values')

    try:
        df = app_state['df']
        scaler = app_state['scaler']
        le = app_state['label_encoder']
        feature_cols = [f'feature_{i}' for i in range(1, 129)]

        true_gas = req_data.get('ground_truth')
        if sample_index is not None:
            row = df.iloc[int(sample_index)]
            sensor_values = [float(row[col]) for col in feature_cols]
            true_label = int(row['class_label'])
            true_gas = GAS_NAMES.get(true_label, "Unknown")

        if sensor_values is None:
            # If features dict was provided but optimized model not trained, or neither provided
            if 'features' in req_data:
                return jsonify({"success": False, "error": "Optimized model not trained yet. Train models first."}), 400
            return jsonify({"success": False, "error": f"Must provide sensor values (expected {N_FEATURES} values)"}), 400

        # Handle case where user sent only the selected feature values as a list
        if app_state['selected_indices'] is not None and len(sensor_values) == len(app_state['selected_indices']):
            padded_values = [0.0] * N_FEATURES
            for sel_pos, orig_idx in enumerate(app_state['selected_indices']):
                padded_values[orig_idx] = float(sensor_values[sel_pos])
            sensor_values = padded_values

        if len(sensor_values) != N_FEATURES:
            return jsonify({"success": False, "error": f"Must provide exactly {N_FEATURES} sensor values (received {len(sensor_values)})"}), 400

        input_array = np.array(sensor_values, dtype=np.float64).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        if app_state['optimized_clf'] is not None:
            clf = app_state['optimized_clf']
            selected = app_state['selected_indices']
            pred_input = scaled_input[:, selected]
            features_used = len(selected)
            model_type = f"GA-Optimized Random Forest ({features_used} features)"
        elif app_state['baseline_clf'] is not None:
            clf = app_state['baseline_clf']
            pred_input = scaled_input
            features_used = N_FEATURES
            model_type = f"Baseline Random Forest ({N_FEATURES} features)"
        else:
            return jsonify({"success": False, "error": "No model trained yet. Run optimization first."}), 400

        pred_encoded = clf.predict(pred_input)[0]
        pred_label = int(le.inverse_transform([pred_encoded])[0])
        pred_gas = GAS_NAMES.get(pred_label, f"Class {pred_label}")

        probabilities = {}
        try:
            proba = clf.predict_proba(pred_input)[0]
            classes = le.classes_
            for i, cls in enumerate(classes):
                gas_name = GAS_NAMES.get(int(cls), f"Class {cls}")
                probabilities[gas_name] = round(float(proba[i]), 5)
        except AttributeError:
            for gas_name in GAS_NAMES.values():
                probabilities[gas_name] = 1.0 if gas_name == pred_gas else 0.0

        confidence = probabilities.get(pred_gas, 0.0)

        # Log prediction to history
        with state_lock:
            app_state['prediction_counter'] += 1
            pred_id = f"PRED-{app_state['prediction_counter']:04d}"
            pred_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            history_entry = {
                "id": pred_id,
                "timestamp": pred_timestamp,
                "predicted_gas": pred_gas,
                "confidence": confidence,
                "probabilities": probabilities,
                "model_used": model_type,
                "n_features_used": features_used,
                "input_source": req_data.get('input_source', 'Manual Sensor Input'),
                "sample_id": req_data.get('sample_id'),
                "ground_truth": true_gas
            }
            app_state['prediction_history'].insert(0, history_entry)
            if len(app_state['prediction_history']) > 100:
                app_state['prediction_history'].pop()

        return jsonify({
            "success": True,
            "prediction_id": pred_id,
            "predicted_gas": pred_gas,
            "predicted_class": pred_label,
            "true_gas": true_gas,
            "ground_truth": true_gas,
            "confidence": confidence,
            "probabilities": probabilities,
            "features_used": features_used,
            "model_type": model_type,
            "timestamp": pred_timestamp,
            "gas_metadata": GAS_METADATA.get(pred_gas, {})
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/prediction_history', methods=['GET'])
def get_prediction_history():
    """Return historical gas prediction logs."""
    return jsonify({
        "success": True,
        "history": app_state['prediction_history'],
        "count": len(app_state['prediction_history'])
    })


@app.route('/api/clear_history', methods=['POST'])
def clear_prediction_history():
    """Clear all historical predictions."""
    with state_lock:
        app_state['prediction_history'] = []
    return jsonify({"success": True, "message": "Prediction history cleared successfully."})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("=" * 70)
    print("  BioSenseGA – Evolutionary Gas Sensor Array Optimization Platform")
    print("  Engineering Lab Project: Random Forest & Genetic Algorithm")
    print("=" * 70)
    print(f"  Server running at http://0.0.0.0:{port} (local: http://127.0.0.1:{port})")
    print("=" * 70)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
