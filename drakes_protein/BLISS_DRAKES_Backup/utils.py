import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import floor, ceil

def get_eval_stats(df, target_protein = None, summary_func = np.median):
    if target_protein is not None:
        df = df[df['protein_name'] == target_protein + '.pdb']
    stats = {}
    if 'ddg_eval' in df:
        ddg_eval = df['ddg_eval'].values
        stats['ddg'] = summary_func(ddg_eval)
        stats['ddg_std'] = np.std(ddg_eval)
        stats['pos_ddg_prop'] = np.mean(ddg_eval > 0)
    if 'ddg' in df:
        ddg_align = df['ddg'].values
        stats['ddg_align'] = summary_func(ddg_align)
        stats['ddg_align_std'] = np.std(ddg_align)
    if 'scrmsd' in df:
        scrmsd_eval = df['scrmsd'].values
        stats['scrmsd'] = summary_func(scrmsd_eval)
        stats['scrmsd_std'] = np.std(scrmsd_eval)
        stats['low_scrmsd_prop'] = np.mean(scrmsd_eval < 2)
    if 'ddg_eval' in df and 'scrmsd' in df:
        success_rate = np.mean((df['ddg_eval'].values > 0) & (df['scrmsd'].values < 2))
        stats['success_rate'] = success_rate
    if 'loglikelihood' in df:
        ll_eval = df['loglikelihood'].values
        stats['ll'] = summary_func(ll_eval)
        stats['ll_std'] = np.std(ll_eval)
    return stats
    
    # # pred_ddg_med = np.median(rewards_eval)
    # # pos_ddg_prop = np.mean(rewards_eval > 0)
    # # scRMSD_med = np.median(scRMSD_eval)
    # # low_rmsd_prop = np.mean(scRMSD_eval < 2)
    # # success_rate = np.mean((rewards_eval > 0) & (scRMSD_eval < 2))
    # # log_likelihood_med = np.median(loglikelihoods)

    # # print(f"Pred-ddG (median)↑: {pred_ddg_med:.3f}")
    # # print(f"%(ddG > 0) (%)↑: {pos_ddg_prop * 100:.1f}")
    # # print(f"scRMSD (median)↓: {scRMSD_med:.3f}")
    # # print(f"%(scRMSD < 2)(%)↑: {low_rmsd_prop * 100:.1f}")
    # # print(f"Success Rate (%)↑: {success_rate * 100:.1f}")
    # # print(f"Log Likelihood (median): {log_likelihood_med:.3f}")

    # stats = {
    #     "pred_ddg_med" : pred_ddg_med,
    #     "pos_ddg_prop" : pos_ddg_prop,
    #     "scRMSD_med" : scRMSD_med,
    #     "low_rmsd_prop" : low_rmsd_prop,
    #     "success_rate" : success_rate,
    #     "ll_med" : log_likelihood_med
    # }
    return stats

def plot_line_prot_comp(key, groups, group_labels, item_labels, markers=None, title='', x_label='', y_label='', h_line_threshold=None, h_line_label=''):
    if markers == None:
        markers = ['o' for _ in range(len(groups))]
    for group, label, marker in zip(groups, group_labels, markers):
        values = [t[key] for t in group]
        plt.style.use('default')
        plt.plot(item_labels, values, marker=marker, label=label)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    if h_line_threshold is not None:
        plt.axhline(y=0, color='#E06455', linestyle='--', label=h_line_label)
    plt.legend()
    plt.show()

def analyze_protein_gen_helper(protein_name, dfs, dfs_labels, clrs, key, y_label='', v_line_thresh=None, v_line_label=''):
    energy_points = None
    group_labels = []
    for df, label in zip(dfs, dfs_labels):
        if protein_name is not None:
            pred_ddg = df[df['protein_name'] == protein_name + '.pdb'][key].values
        else:
            pred_ddg = df[key].values
        group_labels.extend([label] * pred_ddg.shape[0])
        energy_points = pred_ddg if (energy_points is None) else np.concatenate((energy_points, pred_ddg))

    data = pd.DataFrame({
    'Energy': np.array(energy_points),
    'Group': group_labels
    })
    plt.style.use('default')
    plt.figure(figsize=(8, 6))
    for label, clr in zip(dfs_labels, clrs):
      sns.kdeplot(data=data[data['Group'] == label], x='Energy', color=clr, label=label, fill=True, alpha=0.4, bw_adjust=0.8)

    if v_line_thresh is not None:
      plt.axvline(x=v_line_thresh, color='#E06455', linestyle='--', label=v_line_label)

    plt.xlabel(y_label, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    #plt.legend(title='', loc='lower center', fontsize=8, ncol=2)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection

def analyze_protein_gen_helper_violin(protein_name, dfs, dfs_labels, clrs, key, y_label='', v_line_thresh=None, v_line_label='', legend_pos='top', title='', out_path=None):
    energy_points = None
    group_labels = []
    for df, label in zip(dfs, dfs_labels):
        if protein_name is not None:
            pred_ddg = df[df['protein_name'] == protein_name + '.pdb'][key].values
        else:
            pred_ddg = df[key].values
        group_labels.extend([label] * pred_ddg.shape[0])
        energy_points = pred_ddg if (energy_points is None) else np.concatenate((energy_points, pred_ddg))

    data = pd.DataFrame({
        'Energy': np.array(energy_points),
        'Group': group_labels
    })

    plt.style.use('default')
    if legend_pos == 'right':
        plt.figure(figsize=(10, 6))
    else:
        plt.figure(figsize=(8, 6))
    
    ax = sns.boxplot(
        data=data,
        x='Group',
        y='Energy',
        hue='Group',
        palette=clrs,
        showfliers=False,
        # cut=0,
        # inner='box',
        # dodge=False,       # Important so the violins are not split
        legend=False       # Do NOT let seaborn auto-create legend
    )

    if v_line_thresh is not None:
        plt.axhline(y=v_line_thresh, color='#E06455', linestyle='--', label=v_line_label)

    plt.ylabel(y_label, fontsize=14)
    plt.xlabel('')
    plt.xticks([])

    # Remove violin borders (the full violin shapes are 'collections')
    for collection in ax.collections:
        collection.set_edgecolor('none')
        
    # Manually build legend
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(clrs, dfs_labels)]
    if legend_pos == 'right':
        plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    else:
        num_items = len(clrs)
        ncol = num_items if num_items <= 3 else num_items // 2
        plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=ncol, frameon=False)
    
    if title != "":
        plt.title(title)

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
    plt.show()

def plot_reward_comparison(iterations, rewards, colors, linestyles, labels, title, reward_label):
  plt.style.use('default')

  fig, ax = plt.subplots(figsize=(8, 6))
  ax.grid(color='gray', linewidth=0.5)
  for i, reward in enumerate(rewards):
      ax.plot(iterations, reward, color=colors[i], label=labels[i], linestyle=linestyles[i], linewidth=2)
  ax.set_ylabel(reward_label)
  ax.set_xlabel('Diffusion Iteration')
  ax.tick_params(axis='y', labelcolor='black')
  all_reward = np.concatenate(rewards)
  ax.set_ylim(min(all_reward), max(all_reward))
  ax.set_xlim(min(iterations), max(iterations))
  y_min = floor(min(all_reward) * 2) / 2
  y_max = ceil(max(all_reward) * 2) / 2
  y_max += ceil(np.abs(y_max) * 0.025)
  ax.set_yticks(np.linspace(y_min, y_max, num=5))

  num_items = len(colors)
  ncol = num_items if num_items <= 3 else num_items // 2
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=ncol, frameon=False)
  plt.title(title, pad=40)
  plt.tight_layout()
  plt.show()
