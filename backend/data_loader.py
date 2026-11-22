import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import GAS_NAMES, FEATURE_NAMES, GAS_METADATA, SENSOR_TYPES

def parse_libsvm_line(line):
    parts = line.strip().split()
    if not parts:
        return None, None
    label = int(parts[0])
    feature_dict = {}
    for part in parts[1:]:
        if ':' in part:
            idx_str, val_str = part.split(':')
            feature_dict[int(idx_str)] = float(val_str)
    return label, feature_dict

def load_batch_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            label, features = parse_libsvm_line(line)
            if label is not None:
                data.append((label, features))
    return data

def load_all_batches(data_dir):
    """High-performance vectorized loader for all 10 batch files."""
    import numpy as np
    all_rows, labels, batches = [], [], []
    found_files = False
    
    for batch_id in range(1, 11):
        filename = f"batch{batch_id}.dat"
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            found_files = True
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    row = np.zeros(128, dtype=np.float32)
                    for p in parts[1:]:
                        if ':' in p:
                            c, v = p.split(':', 1)
                            idx = int(c) - 1
                            if 0 <= idx < 128:
                                row[idx] = float(v)
                    all_rows.append(row)
                    labels.append(int(parts[0]))
                    batches.append(batch_id)
        else:
            print(f"Warning: {filename} not found in {data_dir}. Skipping.")
            
    if not found_files:
        raise ValueError(f"No batch files found in directory: {data_dir}")
        
    X = np.vstack(all_rows)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 129)])
    df.insert(0, 'class_label', labels)
    df.insert(1, 'batch_id', batches)
    return df

def get_dataset_info(df):
    class_dist = df['class_label'].value_counts().to_dict()
    mapped_dist = {GAS_NAMES.get(k, f"Unknown_{k}"): int(v) for k, v in class_dist.items()}
    
    # Batch counts
    batch_counts = df['batch_id'].value_counts().sort_index().to_dict()
    mapped_batch_counts = {int(k): int(v) for k, v in batch_counts.items()}

    # Batch x Class breakdown
    cross_tab = pd.crosstab(df['batch_id'], df['class_label'])
    batch_class_breakdown = {}
    for batch_id in sorted(df['batch_id'].unique()):
        batch_class_breakdown[int(batch_id)] = {}
        for class_id in sorted(df['class_label'].unique()):
            gas_name = GAS_NAMES.get(class_id, f"Class_{class_id}")
            val = cross_tab.loc[batch_id, class_id] if (batch_id in cross_tab.index and class_id in cross_tab.columns) else 0
            batch_class_breakdown[int(batch_id)][gas_name] = int(val)
    
    return {
        'n_samples': int(len(df)),
        'n_features': 128,
        'n_sensors': 16,
        'n_classes': int(df['class_label'].nunique()),
        'class_names': [GAS_NAMES[i] for i in sorted(GAS_NAMES.keys())],
        'class_distribution': mapped_dist,
        'batch_ids': sorted([int(b) for b in df['batch_id'].unique().tolist()]),
        'batch_distribution': mapped_batch_counts,
        'batch_class_breakdown': batch_class_breakdown,
        'feature_names': FEATURE_NAMES,
        'gas_metadata': GAS_METADATA
    }
