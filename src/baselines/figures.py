"""Overlap scatterplots in the paper's style: one point per dyad, predicted (x) against empirical (y)
overlap, dashed identity line. Baseline panels show child sites, where the models are scored."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from determiner.paths import RESULTS_DIR

KW = dict(s=80, edgecolor="black", linewidth=0.7, zorder=3)


def _fmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _setup(ax, title, lim=0.7):
    ax.plot([0, lim], [0, lim], "r--", lw=1.2, zorder=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted overlap"); ax.set_ylabel("Empirical overlap")


def overlap_figure(points: pd.DataFrame, results: pd.DataFrame, out_path, ngram: bool = True):
    """points: rows (panel, label, child_name, predicted, empirical) for child sites;
    results: rows (label, dxn_p) for the legends."""
    sns.set_style("whitegrid")
    hb = pd.read_csv(RESULTS_DIR / "overlap" / "overlap_human_baseline.csv")
    hb["spk"] = np.where(hb.speaker.str.contains("_mot"), "mother", "child")
    panels = ["Memory model", "Coin-toss model"] + (["N-gram model"] if ngram else [])
    fig, axs = plt.subplots(1, 1 + len(panels), figsize=(5.3 * (1 + len(panels)), 5.6))
    _setup(axs[0], "Humans")
    for spk, col, lab in [("child", "skyblue", "Children"), ("mother", "pink", "Caretakers")]:
        h = hb[hb.spk == spk]
        axs[0].scatter(h.naive_zipf_predicted, h.empirical, color=col, label=lab, **KW)
    axs[0].legend(loc="upper left", fontsize=9)
    styles = iter([("tab:purple", "o"), ("tab:gray", "o"), ("tab:orange", "s"), ("tab:red", "^"),
                   ("tab:green", "D"), ("tab:olive", "v")])
    for ax, panel in zip(axs[1:], panels):
        _setup(ax, panel)
        for lab, g in points[points.panel == panel].groupby("label", sort=False):
            col, mk = next(styles)
            p = results.set_index("label").dxn_p.get(lab, np.nan)
            ax.scatter(g.predicted, g.empirical, color=col, marker=mk,
                       label=lab if np.isnan(p) else f"{lab}: D×N {_fmt(p)}", **KW)
        ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def zipf_audit_figure(coin_points: pd.DataFrame, self_check_points: pd.DataFrame, out_path):
    """Left: rank-frequency of every sample against Zipf(1). Middle: 0.82 coin on transcript nouns
    vs Zipf nouns. Right: data generated from the formula's own assumptions, scored as in the paper."""
    from baselines.common import load_inventory
    sns.set_style("whitegrid")
    inv = load_inventory().dropna(subset=["noun"])
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
    ax = axs[0]
    for spk, col, lab in [("child", "skyblue", "Children"), ("mother", "pink", "Caretakers")]:
        for i, (_, g) in enumerate(inv[inv.speaker == spk].groupby("child_name")):
            f = np.sort(g.noun.value_counts().values)[::-1]
            ax.loglog(np.arange(1, len(f) + 1), f / f.sum(), color=col, lw=1.2, label=lab if i == 0 else None)
    for N, ls in [(340, "--"), (728, ":")]:
        r = np.arange(1, N + 1)
        ax.loglog(r, 1 / (r * np.sum(1 / r)), "k" + ls, lw=1.5, label=f"Zipf(1), N = {N}")
    ax.set_xlabel("Noun rank"); ax.set_ylabel("Share of D×N tokens"); ax.set_title("Noun frequencies", fontsize=13)
    ax.legend(fontsize=9)
    _setup(axs[1], "0.82 coin: transcript nouns vs. Zipf nouns")
    for (nouns, g), (col, mk) in zip(coin_points.groupby("nouns", sort=False), [("tab:purple", "o"), ("tab:orange", "s")]):
        axs[1].scatter(g.pred, g.emp, color=col, marker=mk, label=f"nouns: {nouns}", **KW)
    axs[1].legend(loc="upper left", fontsize=9)
    _setup(axs[2], "Formula's own assumptions, scored as in the paper")
    for spk, col in [("child", "skyblue"), ("mother", "pink")]:
        g = self_check_points[self_check_points.speaker == spk]
        axs[2].scatter(g.pred, g.emp, color=col, label=f"{spk}-sized samples", **KW)
    axs[2].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def tpr_figure(points: pd.DataFrame, adult_mean: float, out_path):
    """points: rows (label, speaker, child_name, tpr), per-dyad TPR; one column per label,
    child sites (blue) and caretaker sites (pink), dashed line at the pooled adult mean."""
    sns.set_style("whitegrid")
    labels = list(dict.fromkeys(points.label))
    fig, ax = plt.subplots(figsize=(1.45 * len(labels) + 2, 5.4))
    rng = np.random.default_rng(0)
    for i, lab in enumerate(labels):
        for spk, col, dx in [("child", "skyblue", -0.17), ("mother", "pink", 0.17)]:
            g = points[(points.label == lab) & (points.speaker == spk)]
            ax.scatter(i + dx + rng.uniform(-0.06, 0.06, len(g)), g.tpr, color=col, **KW)
            ax.plot([i + dx - 0.13, i + dx + 0.13], [g.tpr.mean()] * 2, color="black", lw=2, zorder=4)
    ax.axhline(adult_mean, color="red", ls="--", lw=1.2, zorder=1)
    ax.text(len(labels) - 0.5, adult_mean + 0.008, f"adult mean {adult_mean:.3f}", color="red", ha="right", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("TPR per dyad")
    ax.set_title("Transition probability ratio (TPR)", fontsize=13)
    ax.scatter([], [], color="skyblue", label="child sites", **KW)
    ax.scatter([], [], color="pink", label="caretaker sites", **KW)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
