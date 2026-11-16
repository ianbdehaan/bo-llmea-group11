import json
import shutil
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import ioh

from analysis.plot_search_res import plot_search_result


def load_targets(cfg):
    # names and folders of the two targets, all taken from configs.yaml
    bbob = cfg['targets']['bbob']
    rw = cfg['targets']['real_world']
    save_dir = Path(cfg['logging']['save_dir'])
    return {
        'bbob': {
            'folder': save_dir / f"bbob_f{bbob['function_id']}",
            'stage2_label': f"BBOB_F{bbob['function_id']}",
            'title': f"BBOB F{bbob['function_id']}, instance {bbob['instance']}",
            'f_opt': float(ioh.get_problem(bbob['function_id'], bbob['instance'], cfg['experiment']['dimension']).optimum.y),
        },
        'real_world': {
            'folder': save_dir / f"real_world_{rw['problem_id']}",
            'stage2_label': f"RealWorld_{rw['problem_id']}",
            'title': f"Real-world problem {rw['problem_id']}",
            'f_opt': 0.0,
        },
    }


def run_stage1_plotter(targets, out_base):
    # the LLaMEA-BO plots (AOCC per generation and violin plot) from the final population files
    for key, target in targets.items():
        target_dir = target['folder']
        subdirs = [d for d in target_dir.iterdir() if d.is_dir()] if target_dir.exists() else []
        if not subdirs:
            print(f'No stage 1 search found in {target_dir}')
            continue

        # a search that was stopped has no final file yet, then the latest checkpoint is used
        for sub in subdirs:
            if not list(sub.glob('*final*.pkl')) and (target_dir / 'checkpoint_population.pkl').exists():
                shutil.copy2(target_dir / 'checkpoint_population.pkl', sub / 'ESPopulation_from_checkpoint_final.pkl')
                print(f'No final population in {sub}, using the checkpoint')

        fig_dir = out_base / f'stage1_{target_dir.name}'
        fig_dir.mkdir(parents=True, exist_ok=True)
        cache_pkl = fig_dir / f'cached_{target_dir.name}_df.pkl'
        # the cache would contain an older search, so it is always rebuilt
        if cache_pkl.exists():
            cache_pkl.unlink()
        plot_search_result(result_dir=str(target_dir), save_name=str(cache_pkl), extract_fn=None, fig_dir=str(fig_dir))
        print(f'Stage 1 LLaMEA plots saved in: {fig_dir}')


def plot_stage1_history(targets, cfg, out_base):
    # AOCC of every candidate in the order it was generated and the best AOCC so far,
    # read from the history.jsonl files written during the LLaMEA searches
    n_seeds = len(cfg['experiment']['seeds'])
    inner_budget = cfg['stage1_generation']['inner_budget']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    rows = []
    for ax, target in zip(axes, targets.values()):
        files = sorted(target['folder'].glob('*/history.jsonl'))
        if not files:
            ax.set_title(f"{target['title']}: no history.jsonl found")
            continue
        records = [json.loads(line) for line in open(files[-1], encoding='utf-8') if line.strip()]
        records.sort(key=lambda r: r['index'])
        idx = np.array([r['index'] for r in records])
        # failed candidates (no code, crash, timeout) get AOCC 0, as in the LLaMEA selection
        failed = np.array([bool(r['error']) for r in records])
        aocc = np.array([0.0 if r['error'] or r['fitness'] is None else r['fitness'] for r in records])
        ax.scatter(idx[~failed], aocc[~failed], s=14, color='#1f77b4', label='working candidate')
        ax.scatter(idx[failed], aocc[failed], s=14, color='#d62728', marker='x', label='failed candidate (AOCC 0)')
        ax.plot(idx, np.maximum.accumulate(aocc), color='black', linewidth=1.8, label='best AOCC so far')
        ax.set_title(f"{target['title']}: {failed.sum()}/{len(records)} candidates failed", fontsize=11)
        ax.set_xlabel('Candidate algorithm (generation order)')
        ax.set_ylabel(f'AOCC (mean over {n_seeds} seeds, {inner_budget} evals)')
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)

        best = records[int(np.argmax(aocc))]
        rows.append({'target': target['title'], 'candidates': len(records), 'failed': int(failed.sum()),
                     'best_index': best['index'], 'best_generation': best['generation'], 'best_name': best['name'],
                     'best_aocc': float(aocc.max()),
                     'prompt_tokens': sum(r['prompt_tokens'] for r in records),
                     'response_tokens': sum(r['response_tokens'] for r in records)})
    plt.tight_layout()
    plt.savefig(out_base / 'stage1_aocc_history.png')
    plt.savefig(out_base / 'stage1_aocc_history.pdf')
    plt.close()
    if rows:
        pd.DataFrame(rows).to_csv(out_base / 'stage1_summary.csv', index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved plot: {out_base / 'stage1_aocc_history.png'}")


def stage2_styles(algos):
    # colours, line styles and legend names; generated algorithm names come from the csv
    names = {'BoTorch_Vanilla': 'BoTorch vanilla BO (GP + EI)', 'CMA_ES': 'CMA-ES'}
    styles = {'BoTorch_Vanilla': ('#2ca02c', '-.'), 'CMA_ES': ('#d62728', ':')}
    order = []
    for a in algos:
        if a.startswith('Generated_BBOB'):
            names[a] = a.replace('Generated_BBOB', 'Generated on BBOB')
            styles[a] = ('#1f77b4', '-')
            order.insert(0, a)
        elif a.startswith('Generated_RealWorld'):
            names[a] = a.replace('Generated_RealWorld', 'Generated on real-world')
            styles[a] = ('#ff7f0e', '--')
            order.append(a)
    order += [a for a in ['BoTorch_Vanilla', 'CMA_ES'] if a in algos]
    return order, names, styles


def plot_stage2_convergence(targets, cfg, out_base):
    conv_path = Path(cfg['logging']['save_dir']) / 'stage2_benchmark' / 'stage2_convergence_data.csv'
    if not conv_path.exists():
        print(f'{conv_path} not found, run stage2_benchmark.py first.')
        return
    df_raw = pd.read_csv(conv_path)
    budget = cfg['stage2_generalization']['evaluation_budget']
    dim = cfg['experiment']['dimension']
    order, names, styles = stage2_styles(list(df_raw['Algorithm'].unique()))

    specs = [
        (targets['bbob'], 'Loss $f(x) - f_{opt}$, best so far (mean ± SE over 10 runs)', 'stage2_bbob_loss_convergence'),
        (targets['real_world'], 'Objective $f(x)$, best so far (mean ± SE over 10 runs)', 'stage2_realworld_objective_convergence'),
    ]
    for target, ylabel, stem in specs:
        sub = df_raw[df_raw['Problem'] == target['stage2_label']].copy()
        if sub.empty:
            continue
        # loss for BBOB (f_opt known), raw objective for the real-world problem (f_opt = 0)
        sub['Metric'] = np.maximum(sub['Best_So_Far'] - target['f_opt'], 1e-12)
        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
        for algo in order:
            pivot = sub[sub['Algorithm'] == algo].pivot(index='Evaluation', columns='Seed', values='Metric')
            if pivot.empty:
                continue
            mean_vals = pivot.mean(axis=1).values
            se_vals = pivot.std(axis=1, ddof=1).values / np.sqrt(pivot.shape[1])
            color, ls = styles[algo]
            ax.plot(pivot.index.values, mean_vals, label=names[algo], color=color, linestyle=ls, linewidth=2.1)
            ax.fill_between(pivot.index.values, np.maximum(mean_vals - se_vals, 1e-12), mean_vals + se_vals, color=color, alpha=0.18)
        ax.set_title(f"Stage 2: {target['title']} (d={dim})", fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Function evaluations (budget = {budget})', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_yscale('log')
        ax.set_xlim(1, budget)
        ax.grid(True, which='both', linestyle='--', alpha=0.35)
        ax.legend(fontsize=9.5, loc='upper right', framealpha=0.92)
        plt.tight_layout()
        plt.savefig(out_base / f'{stem}.png', dpi=300)
        plt.savefig(out_base / f'{stem}.pdf')
        plt.close()
        print(f"Saved plot: {out_base / (stem + '.png')}")

    stage2_statistics(df_raw, targets, order, names, budget, out_base)


def stage2_statistics(df_raw, targets, order, names, budget, out_base):
    # final performance per problem and a two-sided Wilcoxon rank-sum (Mann-Whitney U) test of every
    # method against the method with the best median, 10 runs vs 10 runs, Holm correction per problem
    rows = []
    for target in targets.values():
        final = df_raw[(df_raw['Problem'] == target['stage2_label']) & (df_raw['Evaluation'] == budget)]
        if final.empty:
            continue
        values = {a: final[final['Algorithm'] == a]['Best_So_Far'].values - target['f_opt'] for a in order}
        values = {a: v for a, v in values.items() if len(v) > 0}
        best = min(values, key=lambda a: np.median(values[a]))
        p_values = {a: (np.nan if a == best else mannwhitneyu(values[a], values[best], alternative='two-sided').pvalue) for a in values}

        # Holm correction of the p-values of this problem
        others = sorted([a for a in values if a != best], key=lambda a: p_values[a])
        holm, running_max = {}, 0.0
        for k, a in enumerate(others):
            running_max = max(running_max, min(1.0, (len(others) - k) * p_values[a]))
            holm[a] = running_max

        for a, v in values.items():
            rows.append({
                'problem': target['title'],
                'metric': 'loss f - f_opt' if target['f_opt'] != 0 else 'objective f',
                'algorithm': names.get(a, a),
                'mean': v.mean(), 'std': v.std(ddof=1), 'se': v.std(ddof=1) / np.sqrt(len(v)),
                'median': np.median(v), 'min': v.min(), 'max': v.max(), 'runs': len(v),
                'p_vs_best': p_values[a], 'p_vs_best_holm': holm.get(a, np.nan),
                'best_median': a == best,
            })
    stats_df = pd.DataFrame(rows).sort_values(['problem', 'median'])
    stats_df.to_csv(out_base / 'stage2_statistics.csv', index=False)
    print('\nStage 2 final performance (p-values: Wilcoxon rank-sum vs best median, Holm corrected)')
    print(stats_df.to_string(index=False, float_format=lambda x: f'{x:.4g}'))


def main():
    with open('configs.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    out_base = Path(cfg['logging']['save_dir']) / 'convergence_analysis'
    out_base.mkdir(parents=True, exist_ok=True)
    targets = load_targets(cfg)
    run_stage1_plotter(targets, out_base)
    plot_stage1_history(targets, cfg, out_base)
    plot_stage2_convergence(targets, cfg, out_base)
    print('\nAll plots generated in:', out_base)


if __name__ == '__main__':
    main()
