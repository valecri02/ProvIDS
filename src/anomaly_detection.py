import argparse
import os

import numpy as np
import pandas as pd
import wandb
from matplotlib.pyplot import text
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import roc_auc_score, average_precision_score


def _normalize_hash_ids(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Normalize a hash id column to int64, dropping non-parsable rows.

    Ground-truth CSVs sometimes contain ids in non-canonical string formats
    (e.g., scientific notation). Normalizing to integers makes matching robust.
    """
    if col not in df.columns:
        raise KeyError(f"Missing column: {col}")
    out = df.copy()
    s = pd.to_numeric(out[col], errors='coerce')
    out[col] = s
    out = out.dropna(subset=[col])
    out[col] = out[col].astype('int64')
    return out


def compute_detection_performance(
    prediction_folder,
    ground_truth_path,
    model_name,
    conf_id,
    dataset,
    num_seeds,
    log_wandb,
    save_folder,
    split=None,
    scatter_seed=0,
    scatter_num_bins=200,
    scatter_max_marker_size=80.0,
    scatter_min_marker_size=6.0,
):
    save_folder = os.path.join(save_folder, model_name)
    os.makedirs(save_folder, exist_ok=True)
    if log_wandb:
        run = wandb.init(project=f"darpa_{dataset}_anomaly_detection", name=model_name)
    if dataset == 'trace':
        split = "0to210"
        # Ground Truth
        ## part1
        firefox_backdoor = pd.read_csv(
            os.path.join(ground_truth_path, "TC3_trace_firefox_backdoor_final_aggregated.csv"),
            dtype={'edge_hash_id': 'string', 'srcnode_hash_id': 'string', 'label': 'string'},
        )
        ## part2
        browser_extension = pd.read_csv(
            os.path.join(ground_truth_path, "TC3_trace_browser_extension_final_aggregated.csv"),
            dtype={'edge_hash_id': 'string', 'srcnode_hash_id': 'string', 'label': 'string'},
        )
        pine_phishing_exe = pd.read_csv(
            os.path.join(ground_truth_path, "TC3_trace_pine_phishing_exe_final._aggregated.csv"),
            dtype={'edge_hash_id': 'string', 'srcnode_hash_id': 'string', 'label': 'string'},
        )
        trace_thunderbird_phishing_exe = pd.read_csv(
            os.path.join(ground_truth_path, "TC3_trace_thunderbird_phishing_exe_final_aggregated.csv"),
            dtype={'edge_hash_id': 'string', 'srcnode_hash_id': 'string', 'label': 'string'},
        )
        attacks_dict = {"firefox_backdoor":firefox_backdoor, "browser_extension":browser_extension, "pine_phishing_exe":pine_phishing_exe, "trace_thunderbird_phishing_exe":trace_thunderbird_phishing_exe}
    elif dataset == 'theia':
        if split is not None:
            split = str(split)
        else:
            split = "0to25"
        firefox_backdoor = pd.read_csv(
            os.path.join(ground_truth_path, "TC3_theia_firefox_backdoor_final_aggregated.csv"),
            dtype={'edge_hash_id': 'string', 'srcnode_hash_id': 'string', 'label': 'string'},
        )
        browser_extension = pd.read_csv(
            os.path.join(ground_truth_path, "TC3_theia_browser_extension_final_aggregated.csv"),
            dtype={'edge_hash_id': 'string', 'srcnode_hash_id': 'string', 'label': 'string'},
        )
        attacks_dict = {"firefox_backdoor":firefox_backdoor, "browser_extension":browser_extension}
    else:
        raise NotImplemented
    modes = ['success', 'failure']
    keys = list(attacks_dict.keys())
    
    tpr = []
    fpr = []
    auc = []
    ap = []
    attack_detections = {}
    prediction_path = os.path.join(prediction_folder, f"split_conf_{conf_id}_detection_results-{split}_seed_0.csv")
    predictions = pd.read_csv(prediction_path, dtype={'prob': 'float64'})
    predictions = _normalize_hash_ids(predictions, 'hash_id')
    predictions_hashes = set(predictions['hash_id'].tolist())
    for k in attacks_dict:
        attacks_dict[k] = _normalize_hash_ids(attacks_dict[k], 'edge_hash_id')
        attacks_dict[k] = attacks_dict[k][attacks_dict[k]['edge_hash_id'].isin(predictions_hashes)]
        attack_detections[k] = {}

    preds_total = np.zeros(len(predictions)) 
    preds_all = []
    y_true_global = None
    for seed in range(0, num_seeds):
        prediction_path = os.path.join(prediction_folder, f"split_conf_{conf_id}_detection_results-{split}_seed_{seed}.csv")
        predictions = pd.read_csv(prediction_path, dtype={'prob': 'float64'})
        predictions = _normalize_hash_ids(predictions, 'hash_id')
        predictions['attack'] = ['benign'] * len(predictions)
        predictions['mode'] = ['other'] * len(predictions)
        for k in keys:
            mal_mask = predictions["hash_id"].isin(attacks_dict[k]['edge_hash_id'])
            predictions.loc[mal_mask, 'attack'] = k
            
            for mode in modes:
                mask = predictions["hash_id"].isin(attacks_dict[k]['edge_hash_id'][attacks_dict[k]['label'] == mode])
                predictions.loc[mask, 'mode'] = mode
                preds = (1 - predictions.loc[mask].prob.values).round().astype(int)
                y_true = mask.values.astype(int)
                size = len(attacks_dict[k]['edge_hash_id'][attacks_dict[k]['label'] == mode])
                if size > 0:
                    if mode in attack_detections[k]:
                        attack_detections[k][mode].append(preds.sum())
                    else:
                        attack_detections[k][mode] = [preds.sum()]
                    if log_wandb:
                        wandb.log({f"{k} total {size}, {mode}": preds.sum()}, step = 0)

        preds = (1 - predictions.prob.values)
        y_true = (predictions['attack']!='benign').values.astype(int)
        preds_label = preds.round().astype(int)

        # store per-seed predictions for later multi-run histogram analysis
        preds_all.append(preds.copy())
        if y_true_global is None:
            y_true_global = y_true.copy()

        fp = (preds_label.astype(bool) & ~y_true.astype(bool)).sum()
        tn = (~preds_label.astype(bool) & ~y_true.astype(bool)).sum()
        tp = (preds_label.astype(bool) & y_true.astype(bool)).sum()
        fn = (~preds_label.astype(bool) & y_true.astype(bool)).sum()
        auc.append(roc_auc_score(y_true, preds))
        ap.append(average_precision_score(y_true, preds))
        fpr.append(fp / (fp + tn))
        tpr.append(tp / ( tp + fn ))
        preds_total += preds / num_seeds


    metrics = {'fpr': fpr, 'auc': auc, 'ap': ap, 'tpr': tpr}
    results = {}
    for metric_name, metric in metrics.items():
        mean = np.array(metric).mean()
        std = np.array(metric).std()
        results[f"{metric_name} total mean"] = [mean]
        results[f"{metric_name} total std"] = [std]
        if log_wandb:
            wandb.log({f"{metric_name} total mean": mean}, step = 0)
            wandb.log({f"{metric_name} total std": std}, step = 0)
    df = DataFrame(results)
    df.to_csv(os.path.join(save_folder, "detection_stats_%s.csv"%model_name))

    if log_wandb:
        _cm_name = f"conf_mat total "
        _cm = wandb.plot.confusion_matrix(preds=preds_label,
                                            y_true=y_true.astype(int),
                                            class_names=["benign", "anomaly"],
                                            title=_cm_name)
        wandb.log({_cm_name : _cm}, step = 0)
        wandb.log({f"roc_curve total": wandb.plot.roc_curve(y_true, np.concatenate((1 - preds_total[:, None], preds_total[:, None]), axis=1), labels=['benign', 'anomaly'])}, step = 0)
        wandb.log({f"pr_curve total": wandb.plot.pr_curve(y_true, np.concatenate((1 - preds_total[:, None], preds_total[:, None]), axis=1), labels=['benign', 'anomaly'])}, step = 0)

    # multi-run histogram: compute per-seed histograms for malicious and benign, then plot mean +/- std
    if len(preds_all) == 0:
        raise ValueError("No prediction seeds found to build histogram")
    bins = np.linspace(0, 1, 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    preds_array = np.stack(preds_all)  # shape: (num_seeds, n_samples)
    mal_mask = (y_true_global == 1)
    ben_mask = (y_true_global == 0)

    mal_hist_per_seed = []
    ben_hist_per_seed = []
    for s in range(preds_array.shape[0]):
        mal_scores = preds_array[s][mal_mask]
        ben_scores = preds_array[s][ben_mask]
        if mal_scores.size > 0:
            mal_counts, _ = np.histogram(mal_scores, bins=bins)
            mal_counts = mal_counts.astype(float) / max(1, mal_scores.size)
        else:
            mal_counts = np.zeros(len(bin_centers), dtype=float)
        if ben_scores.size > 0:
            ben_counts, _ = np.histogram(ben_scores, bins=bins)
            ben_counts = ben_counts.astype(float) / max(1, ben_scores.size)
        else:
            ben_counts = np.zeros(len(bin_centers), dtype=float)
        mal_hist_per_seed.append(mal_counts)
        ben_hist_per_seed.append(ben_counts)

    mal_hist_per_seed = np.vstack(mal_hist_per_seed)
    ben_hist_per_seed = np.vstack(ben_hist_per_seed)

    mal_mean = mal_hist_per_seed.mean(axis=0)
    mal_std = mal_hist_per_seed.std(axis=0)
    ben_mean = ben_hist_per_seed.mean(axis=0)
    ben_std = ben_hist_per_seed.std(axis=0)

    width = bins[1] - bins[0]

    # 1) Mean-only bar plot
    fig_mean, ax_mean = plt.subplots(figsize=(6.4,4.8*1.4))
    plt.rcParams.update({'font.size': 19})
    plt.rcParams['axes.axisbelow'] = True
    ax_mean.bar(bin_centers - width*0.15, mal_mean, width=width*0.3, label='Malicious', color='#fe6100', alpha=0.9)
    ax_mean.bar(bin_centers + width*0.15, ben_mean, width=width*0.3, label='Benign', color='#648fff', alpha=0.9)
    ax_mean.grid(axis='y')
    ax_mean.set_ylim(0, 1)
    step = 0.25
    ax_mean.set_xticks(np.arange(0, 1 + step, step))
    ax_mean.legend(loc='upper right')
    ax_mean.set_xlabel("Anomaly score")
    mean_path = os.path.join(save_folder, f"hist_mean_{model_name}.pdf")
    fig_mean.savefig(mean_path, bbox_inches='tight')
    plt.close(fig_mean)

    # 2) Histogram with uncertainty bars (mean ± std) as bar plot
    fig_anom, ax_anom = plt.subplots(figsize=(6.4,4.8*1.4))
    plt.rcParams.update({'font.size': 19})
    plt.rcParams['axes.axisbelow'] = True
    # side-by-side bars with error bars
    ax_anom.bar(bin_centers - width*0.15, mal_mean, width=width*0.3, yerr=mal_std, label='Malicious', color='#fe6100', alpha=0.9, capsize=3)
    ax_anom.bar(bin_centers + width*0.15, ben_mean, width=width*0.3, yerr=ben_std, label='Benign', color='#648fff', alpha=0.9, capsize=3)
    ax_anom.grid(axis='y')
    ax_anom.set_ylim(0, 1)
    ax_anom.set_xticks(np.arange(0, 1 + step, step))
    ax_anom.legend(loc='upper right')
    ax_anom.set_xlabel("Anomaly score")
    anom_path = os.path.join(save_folder, f"hist_anom_{model_name}.pdf")
    fig_anom.savefig(anom_path, bbox_inches='tight')
    plt.close(fig_anom)

    # Optional wandb logging for both images
    if log_wandb:
        wandb.log({"hist_mean": wandb.Image(mean_path)})
        wandb.log({"hist_anom": wandb.Image(anom_path)})
        run.finish()
    print(model_name, "done")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prediction_folder', required=True)
    parser.add_argument('--ground_truth_path', required=True)
    parser.add_argument('--split', default=None, help='Optional split identifier to override dataset-derived split (e.g. 10 or 0to25)')

    parser.add_argument('--model_name', default="TGN")
    parser.add_argument('--conf_id', default=0)
    parser.add_argument('--dataset', type=str.lower, default='trace', choices=['trace', 'theia'])
    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--log_wandb', action="store_true")
    parser.add_argument('--save_folder', default="figures/")

    parser.add_argument('--scatter_seed', type=int, default=0)
    parser.add_argument('--scatter_num_bins', type=int, default=200)
    parser.add_argument('--scatter_max_marker_size', type=float, default=80.0)
    parser.add_argument('--scatter_min_marker_size', type=float, default=6.0)


    args = parser.parse_args()
    print(args)

    compute_detection_performance(**vars(args))