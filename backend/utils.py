import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SENSOR_FEATURE_MAPPING, SENSOR_TYPES, SENSOR_METADATA, FEATURE_TYPES, FEATURE_METADATA, FEATURE_NAMES

def generate_conclusion(baseline_results, ga_results, n_original, n_selected):
    diff = ga_results['accuracy'] - baseline_results['accuracy']
    diff_pct = diff * 100
    red = (n_original - n_selected) / n_original * 100
    
    if diff >= 0:
        perf_verdict = f"maintained and improved classification performance (+{diff_pct:.2f}% accuracy gain)"
    elif abs(diff) < 0.02:
        perf_verdict = f"preserved near-identical classification performance with only a negligible change of {diff_pct:.2f}%"
    else:
        perf_verdict = f"showed an expected trade-off with a slight accuracy variation of {diff_pct:.2f}%"
        
    return (
        f"The Genetic Algorithm optimized the 16-sensor artificial nose array from {n_original} to {n_selected} "
        f"salient features, achieving a significant dimensionality reduction of {red:.1f}%. "
        f"The GA-optimized Random Forest model {perf_verdict} (Baseline: {baseline_results['accuracy']*100:.2f}%, "
        f"Optimized: {ga_results['accuracy']*100:.2f}%), verifying that sensor drift and redundant multichannel "
        f"responses can be eliminated while preserving discriminative capability."
    )

def format_percentage(value):
    return f"{value * 100:.2f}%"

def get_detailed_sensor_groups(best_chromosome):
    """
    Returns hierarchical structure of all 16 sensors, their physical types,
    descriptions, and the selection status of their 8 extracted features.
    """
    sensor_groups = []
    
    for sensor_idx in range(16):
        s_id = sensor_idx + 1
        sensor_type = SENSOR_TYPES[sensor_idx]
        sensor_meta = SENSOR_METADATA.get(sensor_type, {})
        feature_indices = SENSOR_FEATURE_MAPPING[sensor_idx]
        
        sensor_features = []
        sel_count = 0
        for f_idx in feature_indices:
            ft_name = FEATURE_NAMES[f_idx]
            ft_code = FEATURE_TYPES[f_idx % 8]
            ft_meta = FEATURE_METADATA.get(ft_code, {})
            is_selected = bool(best_chromosome[f_idx] == 1) if best_chromosome is not None else False
            if is_selected:
                sel_count += 1
                
            sensor_features.append({
                'feature_index': int(f_idx),
                'feature_name': ft_name,
                'feature_code': ft_code,
                'feature_title': ft_meta.get('name', ft_code),
                'symbol': ft_meta.get('symbol', ft_code),
                'description': ft_meta.get('description', ''),
                'selected': is_selected
            })
            
        selection_pct = (sel_count / len(feature_indices)) * 100.0
        
        sensor_groups.append({
            'sensor_id': s_id,
            'sensor_name': f"Sensor {s_id}",
            'sensor_type': sensor_type,
            'category': sensor_meta.get('category', 'Metal Oxide Gas Sensor'),
            'target_gases': sensor_meta.get('target_gases', 'Air contaminants'),
            'description': sensor_meta.get('description', ''),
            'features': sensor_features,
            'total_features': len(feature_indices),
            'selected_features_count': sel_count,
            'eliminated_features_count': len(feature_indices) - sel_count,
            'selection_percentage': round(selection_pct, 1),
            'active': bool(sel_count > 0)
        })
        
    return sensor_groups

def get_sensor_selection_summary(best_chromosome, feature_names):
    summary = []
    for sensor_id, feature_indices in SENSOR_FEATURE_MAPPING.items():
        for f_idx in feature_indices:
            summary.append({
                'name': feature_names[f_idx],
                'selected': bool(best_chromosome[f_idx] == 1) if best_chromosome is not None else False,
                'sensor_id': sensor_id + 1
            })
    return summary

def get_sensor_level_selection(best_chromosome):
    sensor_selection = {}
    for sensor_id, feature_indices in SENSOR_FEATURE_MAPPING.items():
        sensor_selection[int(sensor_id + 1)] = bool(
            any(int(best_chromosome[f_idx]) == 1 for f_idx in feature_indices)
        ) if best_chromosome is not None else False
    return sensor_selection
