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

    scatter_scores = None
    scatter_y_true = None
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

        if seed == scatter_seed:
            scatter_scores = preds.copy()
            scatter_y_true = y_true.copy()
            n_mal = int(scatter_y_true.sum())
            n_tot = int(scatter_y_true.shape[0])
            print(
                f"[scatter_seed={scatter_seed}] malicious edges matched via ground truth: {n_mal}/{n_tot} ({(100.0*n_mal/max(n_tot,1)):.4f}%)"
            )

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
        

    malicious = preds_total[y_true==1]
    benign = preds_total[y_true==0]

    if scatter_scores is None or scatter_y_true is None:
        raise ValueError(
            f"scatter_seed={scatter_seed} not found in evaluated seeds [0, {num_seeds - 1}]"
        )

    # Binned scatter: group similar scores into bins and show counts per class.
    scatter_bins = np.linspace(0.0, 1.0, int(scatter_num_bins) + 1)
    benign_counts, _ = np.histogram(scatter_scores[scatter_y_true == 0], bins=scatter_bins)
    malicious_counts, _ = np.histogram(scatter_scores[scatter_y_true == 1], bins=scatter_bins)
    x_centers = (scatter_bins[:-1] + scatter_bins[1:]) / 2.0

    max_count = float(max(benign_counts.max(initial=0), malicious_counts.max(initial=0), 1))
    benign_sizes = (benign_counts.astype(float) / max_count) * float(scatter_max_marker_size)
    malicious_sizes = (malicious_counts.astype(float) / max_count) * float(scatter_max_marker_size)

    benign_sizes = np.where(
        benign_counts > 0,
        np.maximum(benign_sizes, float(scatter_min_marker_size)),
        0.0,
    )
    malicious_sizes = np.where(
        malicious_counts > 0,
        np.maximum(malicious_sizes, float(scatter_min_marker_size)),
        0.0,
    )

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    plt.rcParams.update({'font.size': 19})
    ax.set_axisbelow(True)
    ax.scatter(
        x_centers,
        np.zeros_like(x_centers),
        s=benign_sizes,
        alpha=0.8,
        color='#648fff',
        label='Benign',
    )
    ax.scatter(
        x_centers,
        np.ones_like(x_centers),
        s=malicious_sizes,
        alpha=0.8,
        color='#fe6100',
        label='Malicious',
    )
    ax.set_yticks([0, 1], labels=['Benign', 'Malicious'])
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.25)
    ax.grid(axis='y', alpha=0.25)
    ax.set_xlabel('Anomaly score')
    ax.set_ylabel('Class')
    ax.legend(loc='best')
    scatter_path = os.path.join(
        save_folder,
        f"scatter_anom_{model_name}_seed_{scatter_seed}.png",
    )
    fig.savefig(scatter_path, bbox_inches='tight', dpi=200)
    if log_wandb:
        wandb.log({"scatter_anomaly_score": wandb.Image(fig)}, step=0)
    plt.close(fig)

    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(6.4,4.8*1.4))
    plt.rcParams.update({'font.size': 19})
    plt.rcParams['axes.axisbelow'] = True
    plt.hist([malicious, benign], bins, label=['Malicious', 'Benign'], weights=[np.ones(len(malicious))/len(malicious), np.ones(len(benign))/len(benign)], color=['#fe6100', '#648fff'])
    plt.vlines(x=0.5, ymin=0, ymax=1, colors='black', ls='--', lw=2, label='Threshold')
    plt.grid(axis='y')
    plt.ylim(0, 1)
    step = 0.25
    plt.xticks(np.arange(0, 1 + step, step))
    text(0.5, 0.5, " $FPR_{AD}$ : %.1f%% \n $TPR_{AD}$ : %.1f%%"%(np.array(fpr).mean()*100, np.array(tpr).mean()*100), rotation=0, verticalalignment='center')
    plt.legend(loc='upper right')
    plt.xlabel("Anomaly score")
    plt.ylabel("Normalized counts")
    plt.savefig(os.path.join(save_folder, "hist_anom_%s.pdf"%model_name), bbox_inches='tight')

    if log_wandb:
        wandb.log({"threshold": wandb.Image(plt)})
        run.finish()
    print(model_name, "done")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prediction_folder', required=True)
    parser.add_argument('--ground_truth_path', required=True)

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