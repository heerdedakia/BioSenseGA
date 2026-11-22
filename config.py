# config.py
"""
BioSenseGA Configuration File
Defines sensor hardware mapping, gas analytes, feature types, and GA defaults.
"""

GAS_NAMES = {
    1: 'Ethanol',
    2: 'Ethylene',
    3: 'Ammonia',
    4: 'Acetaldehyde',
    5: 'Acetone',
    6: 'Toluene'
}

GAS_METADATA = {
    'Ethanol': {
        'id': 1,
        'formula': 'C₂H₅OH',
        'type': 'Alcohol',
        'odor': 'Characteristic sweet wine-like',
        'description': 'Volatile solvent widely monitored in breath analysis, brewing, and industrial safety.',
        'color': '#4388FF'
    },
    'Ethylene': {
        'id': 2,
        'formula': 'C₂H₄',
        'type': 'Alkene Hydrocarbon',
        'odor': 'Faint sweet musky',
        'description': 'Critical plant hormone regulating fruit ripening; key analyte for agricultural freshness monitoring.',
        'color': '#10B981'
    },
    'Ammonia': {
        'id': 3,
        'formula': 'NH₃',
        'type': 'Inorganic Alkaline',
        'odor': 'Pungent, suffocating, sharp',
        'description': 'Toxic refrigerant and chemical precursor; essential target for industrial leaks and environmental safety.',
        'color': '#F59E0B'
    },
    'Acetaldehyde': {
        'id': 4,
        'formula': 'C₂H₄O',
        'type': 'Aldehyde',
        'odor': 'Pungent fruity, green apple',
        'description': 'Carcinogenic pollutant and oxidation intermediate in fermentation and chemical processing.',
        'color': '#EC4899'
    },
    'Acetone': {
        'id': 5,
        'formula': 'C₃H₆O',
        'type': 'Ketone',
        'odor': 'Fruity, sweet mint-like',
        'description': 'Key ketone biomarker in human breath for diabetic ketoacidosis and common industrial solvent.',
        'color': '#8B5CF6'
    },
    'Toluene': {
        'id': 6,
        'formula': 'C₇H₈',
        'type': 'Aromatic Hydrocarbon',
        'odor': 'Sweet, pungent paint-thinner',
        'description': 'Aromatic VOC in fuels, coatings, and adhesives; neurotoxic at elevated concentrations.',
        'color': '#06B6D4'
    }
}

# 16 sensors total: 4 of each Figaro metal oxide semiconductor (MOX) type
SENSOR_TYPES = ['TGS2600', 'TGS2602', 'TGS2610', 'TGS2620'] * 4

SENSOR_METADATA = {
    'TGS2600': {
        'type': 'TGS2600',
        'category': 'General Air Quality & Contaminants',
        'target_gases': 'Hydrogen (H₂), Carbon Monoxide (CO), Methane, Cigarette smoke',
        'heater_voltage': '5.0V ± 0.2V',
        'description': 'High sensitivity to gaseous air contaminants with low power consumption.'
    },
    'TGS2602': {
        'type': 'TGS2602',
        'category': 'Volatile Organic Compounds (VOCs) & Odors',
        'target_gases': 'Ammonia (NH₃), Hydrogen Sulfide (H₂S), VOCs, Ethanol, Toluene',
        'heater_voltage': '5.0V ± 0.2V',
        'description': 'High sensitivity to odorous air contaminants and ammonia compounds.'
    },
    'TGS2610': {
        'type': 'TGS2610',
        'category': 'Combustible Hydrocarbons & LP Gases',
        'target_gases': 'Propane (C₃H₈), Butane (C₄H₁₀), LP Gas, Methane',
        'heater_voltage': '5.0V ± 0.2V',
        'description': 'High sensitivity to liquefied petroleum gases and volatile hydrocarbons.'
    },
    'TGS2620': {
        'type': 'TGS2620',
        'category': 'Solvent & Alcohol Vapors',
        'target_gases': 'Ethanol (C₂H₅OH), Solvent vapors, Organic vapors',
        'heater_voltage': '5.0V ± 0.2V',
        'description': 'High sensitivity to alcohol and organic solvent vapors with rapid response time.'
    }
}

FEATURE_TYPES = ['DR', 'abs_DR', 'EMAi0.001', 'EMAi0.01', 'EMAi0.1', 'EMAd0.001', 'EMAd0.01', 'EMAd0.1']

FEATURE_METADATA = {
    'DR': {
        'name': 'Steady-state Differential Resistance',
        'symbol': 'ΔR',
        'description': 'Normalized difference between steady-state resistance and baseline resistance (R_max - R_0).'
    },
    'abs_DR': {
        'name': 'Absolute Differential Resistance',
        'symbol': '|ΔR|',
        'description': 'Absolute magnitude of steady-state resistance change across the sensor element.'
    },
    'EMAi0.001': {
        'name': 'EMA Rising Phase (α = 0.001)',
        'symbol': 'EMAi_0.001',
        'description': 'Exponential moving average of the rising/adsorption signal phase with smooth decay factor α=0.001.'
    },
    'EMAi0.01': {
        'name': 'EMA Rising Phase (α = 0.01)',
        'symbol': 'EMAi_0.01',
        'description': 'Exponential moving average of the rising signal phase with moderate smoothing factor α=0.01.'
    },
    'EMAi0.1': {
        'name': 'EMA Rising Phase (α = 0.1)',
        'symbol': 'EMAi_0.1',
        'description': 'Exponential moving average of the rising signal phase capturing fast transient rise with α=0.1.'
    },
    'EMAd0.001': {
        'name': 'EMA Decaying Phase (α = 0.001)',
        'symbol': 'EMAd_0.001',
        'description': 'Exponential moving average of the decaying/desorption recovery phase with smooth factor α=0.001.'
    },
    'EMAd0.01': {
        'name': 'EMA Decaying Phase (α = 0.01)',
        'symbol': 'EMAd_0.01',
        'description': 'Exponential moving average of the decaying recovery phase with moderate factor α=0.01.'
    },
    'EMAd0.1': {
        'name': 'EMA Decaying Phase (α = 0.1)',
        'symbol': 'EMAd_0.1',
        'description': 'Exponential moving average capturing rapid desorption recovery dynamics with α=0.1.'
    }
}

# 128 total feature names: S1_DR .. S16_EMAd0.1
FEATURE_NAMES = [f"S{sensor_idx+1}_{ft}" for sensor_idx in range(16) for ft in FEATURE_TYPES]

# Mapping sensor index (0..15) to feature indices list (each has 8 features)
SENSOR_FEATURE_MAPPING = {i: list(range(i * 8, (i + 1) * 8)) for i in range(16)}

DEFAULT_GA_PARAMS = {
    'population_size': 20,
    'n_generations': 8,
    'mutation_rate': 0.05,
    'crossover_rate': 0.8,
    'feature_penalty': 0.01,
    'random_seed': 42
}

N_FEATURES = 128
TEST_SIZE = 0.2
