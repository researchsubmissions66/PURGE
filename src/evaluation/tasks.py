"""
Task definitions.

A *task* is a (dataset, label mapping) pair. This separation matters for the
attack's central claim: the goal is to erase one downstream task for an organ,
not the organ itself.

Confound ladder, weakest to strongest control:

  cross-organ      erase lung, check ovarian     -> survival may just mean
                                                    "different organ"
  same-organ,      erase BRACS, check BACH       -> survival may just mean
  cross-cohort                                      "different scanner/site"
  SAME SLIDES      erase PANDA grading,          -> survival can only mean the
                   check PANDA detection            task was spared

Only the last is unconfounded, because target and control share every slide,
every scanner, and every patient. Tasks that map a label to None drop that slide.
"""

TASKS = {
    # ---- PANDA (prostate) : grade vs detect, on the same slides ------------ #
    'PANDA-grade': {
        'dataset': 'PANDA',
        'description': 'ISUP high (3-5) vs low (1-2) grade; benign slides excluded',
        'map': {'ISUP_1': 0, 'ISUP_2': 0, 'ISUP_3': 1, 'ISUP_4': 1, 'ISUP_5': 1},
    },
    'PANDA-detect': {
        'dataset': 'PANDA',
        'description': 'cancer detection: ISUP 0 vs ISUP 1-5',
        'map': {'ISUP_0': 0, 'ISUP_1': 1, 'ISUP_2': 1, 'ISUP_3': 1,
                'ISUP_4': 1, 'ISUP_5': 1},
    },
    'PANDA-isup': {
        'dataset': 'PANDA',
        'description': 'full 6-class ISUP grading',
        'map': {f'ISUP_{i}': i for i in range(6)},
    },

    # ---- BRACS (breast) : subtype vs detect, on the same slides ------------ #
    'BRACS-atypia': {
        'dataset': 'BRACS',
        'description': 'ADH vs FEA, the clinically hard atypical distinction',
        'map': {'ADH': 0, 'FEA': 1},
    },
    'BRACS-subtype': {
        'dataset': 'BRACS',
        'description': '7-class lesion subtyping',
        'map': {'ADH': 0, 'DCIS': 1, 'FEA': 2, 'IC': 3, 'N': 4, 'PB': 5, 'UDH': 6},
    },
    'BRACS-malignancy': {
        'dataset': 'BRACS',
        'description': 'malignant (DCIS, IC) vs everything else',
        'map': {'N': 0, 'PB': 0, 'UDH': 0, 'FEA': 0, 'ADH': 0, 'DCIS': 1, 'IC': 1},
    },
    'BRACS-coarse': {
        'dataset': 'BRACS',
        'description': 'benign / atypical / malignant (official BRACS grouping)',
        'map': {'N': 0, 'PB': 0, 'UDH': 0, 'FEA': 1, 'ADH': 1, 'DCIS': 2, 'IC': 2},
    },

    # ---- other organs / cohorts ------------------------------------------- #
    'BACH-histology': {
        'dataset': 'BACH',
        'description': '4-class breast histology',
        'map': {'Benign': 0, 'InSitu': 1, 'Invasive': 2, 'Normal': 3},
    },
    'BACH-malignancy': {
        'dataset': 'BACH',
        'description': 'malignant (InSitu, Invasive) vs (Normal, Benign)',
        'map': {'Normal': 0, 'Benign': 0, 'InSitu': 1, 'Invasive': 1},
    },
    'TCGA-LUNG-subtype': {
        'dataset': 'TCGA-LUNG',
        'description': 'LUAD vs LUSC',
        'map': {'LUAD': 0, 'LUSC': 1},
    },
    'TCGA-BRCA-subtype': {
        'dataset': 'TCGA-BRCA',
        'description': 'IDC vs ILC',
        'map': {'IDC': 0, 'ILC': 1},
    },
    'UBC-OCEAN-subtype': {
        'dataset': 'UBC-OCEAN',
        'description': '5-class ovarian carcinoma subtyping',
        'map': {'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4},
    },
}


def get_task(name):
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Known: {sorted(TASKS)}")
    return TASKS[name]


def relation(task_a, task_b, organ_of):
    """How strong a control is task_b for task_a? Used to label result rows."""
    da, db = TASKS[task_a]['dataset'], TASKS[task_b]['dataset']
    if da == db:
        return 'same-slides'
    if organ_of.get(da) == organ_of.get(db):
        return 'same-organ'
    return 'cross-organ'
