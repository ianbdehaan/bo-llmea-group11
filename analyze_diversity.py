import re
import sys
import json
import difflib
import itertools
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

# Measures how diverse the algorithms of a LLaMEA search are, from its history.jsonl.
# Every working candidate is classified by its main BO components (found with simple keyword
# rules on the code), and the mean pairwise code similarity (difflib ratio) is computed.
# Usage: python analyze_diversity.py <results_dir> [<results_dir> ...]   (default: outputs)


def surrogate_type(code):
    if re.search(r'RandomForest|ExtraTrees', code):
        return 'random forest'
    if re.search(r'gpytorch|botorch|SingleTaskGP', code):
        return 'GP (torch)'
    if re.search(r'GaussianProcessRegressor', code):
        return 'GP (sklearn)'
    if re.search(r'Rbf|RBFInterpolator', code):
        return 'RBF'
    if re.search(r'MLP|neural|nn\.Module', code):
        return 'neural net'
    return 'other'


def acquisition_types(code):
    found = []
    low = code.lower()
    if re.search(r'expected[_ ]improvement|\bei\b|norm\.cdf\(\s*z\s*\)|improvement \* norm\.cdf', low):
        found.append('EI')
    if re.search(r'\bucb\b|\blcb\b|confidence[_ ]bound|kappa|beta \* sigma|mu - .*\* ?sigma', low):
        found.append('UCB/LCB')
    if re.search(r'probability[_ ]of[_ ]improvement|\bpi\b', low):
        found.append('PI')
    if re.search(r'thompson|sample_y|posterior sample', low):
        found.append('Thompson')
    if re.search(r'entropy|\bmes\b|knowledge[_ ]gradient', low):
        found.append('information based')
    return found if found else ['other']


def optimizer_type(code):
    if re.search(r'differential_evolution\(', code):
        return 'differential evolution'
    if re.search(r'CMAEvolutionStrategy\(|cma\.fmin', code):
        return 'CMA-ES'
    if re.search(r'minimize\(', code):
        return 'gradient (L-BFGS) restarts'
    return 'candidate pool only'


def classify(code):
    return {
        'surrogate': surrogate_type(code),
        'acquisition': '+'.join(acquisition_types(code)),
        'acq_optimizer': optimizer_type(code),
        'trust_region': bool(re.search(r'trust|tr_radius|radius', code, re.IGNORECASE)),
        'local_perturbation': bool(re.search(r'best_x \+|x_best \+|perturb', code)),
    }


def mean_code_similarity(codes, max_pairs=3000, seed=0):
    # mean difflib similarity of random pairs of candidates (1 = identical code)
    pairs = list(itertools.combinations(range(len(codes)), 2))
    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    return float(np.mean([difflib.SequenceMatcher(None, codes[i], codes[j]).ratio() for i, j in pairs]))


def analyze(results_dir):
    rows = []
    for history in sorted(Path(results_dir).glob('*/ESPopulation*/history.jsonl')):
        records = [json.loads(line) for line in open(history, encoding='utf-8') if line.strip()]
        ok = [r for r in records if not r['error']]
        classes = [classify(r['code']) for r in ok]
        combos = Counter(tuple(c.values()) for c in classes)
        aocc = np.array([r['fitness'] for r in ok])
        rows.append({
            'search': f"{Path(results_dir).name}/{history.parent.parent.name}",
            'candidates': len(records),
            'working': len(ok),
            'distinct_component_combinations': len(combos),
            'surrogates': dict(Counter(c['surrogate'] for c in classes)),
            'acquisitions': dict(Counter(c['acquisition'] for c in classes)),
            'acq_optimizers': dict(Counter(c['acq_optimizer'] for c in classes)),
            'with_trust_region': sum(c['trust_region'] for c in classes),
            'mean_code_similarity': round(mean_code_similarity([r['code'] for r in ok]), 3),
            'aocc_std': round(float(aocc.std()), 4),
            'best_aocc': round(float(aocc.max()), 4),
        })
    return rows


if __name__ == '__main__':
    dirs = sys.argv[1:] if len(sys.argv) > 1 else ['outputs']
    rows = [row for d in dirs for row in analyze(d)]
    df = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.width', 250)
    print(df.T.to_string())
    out = Path(dirs[-1]) / 'convergence_analysis'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'diversity.csv', index=False)
