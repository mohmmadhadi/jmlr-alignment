import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Optional

sns.set_theme(style="whitegrid", font_scale=1.1)
SBERT_COL = "sbert_alignment_score"
TFIDF_COL = "tfidf_alignment_score"


def plot_score_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot histogram of SBERT and TF-IDF alignment score distributions.

    Args:
        df: Scored paper DataFrame.
        save_path: If provided, save figure to this path instead of showing.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, color, label in [
        (axes[0], SBERT_COL, "#4C72B0", "SBERT"),
        (axes[1], TFIDF_COL, "#DD8452", "TF-IDF"),
    ]:
        ax.hist(df[col], bins=40, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(df[col].mean(), color="crimson", linestyle="--", linewidth=1.5, label=f"Mean: {df[col].mean():.3f}")
        ax.set_title(f"{label} alignment score distribution")
        ax.set_xlabel("Alignment score")
        ax.set_ylabel("Number of papers")
        ax.legend()

    fig.suptitle("JMLR — Thematic alignment score distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_temporal_drift(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot average SBERT alignment score per year to detect thematic drift.

    Args:
        df: Scored paper DataFrame with a 'year' column.
        save_path: If provided, save figure to this path instead of showing.
    """
    yearly = (
        df.groupby("year")[SBERT_COL]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_score", "std": "std_score", "count": "n_papers"})
    )
    yearly = yearly[yearly["n_papers"] >= 3]

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(yearly["year"], yearly["avg_score"], marker="o", linewidth=2, color="#4C72B0", label="Mean alignment")
    ax.fill_between(
        yearly["year"],
        yearly["avg_score"] - yearly["std_score"],
        yearly["avg_score"] + yearly["std_score"],
        alpha=0.15,
        color="#4C72B0",
        label="±1 std dev",
    )

    z = np.polyfit(yearly["year"], yearly["avg_score"], 1)
    trend = np.poly1d(z)
    ax.plot(yearly["year"], trend(yearly["year"]), linestyle="--", color="crimson", linewidth=1.5, label=f"Trend (slope={z[0]:+.4f}/yr)")

    ax.set_title("JMLR — Thematic alignment drift over time (SBERT)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean alignment score")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_outliers(
    df: pd.DataFrame,
    n: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """Horizontal bar chart of the top and bottom N papers by SBERT score.

    Args:
        df: Scored paper DataFrame.
        n: Number of papers to show on each end.
        save_path: If provided, save figure to this path instead of showing.
    """
    top = df.nlargest(n, SBERT_COL).copy()
    bottom = df.nsmallest(n, SBERT_COL).copy()
    combined = pd.concat([bottom, top])

    labels = [t[:55] + "…" if len(t) > 55 else t for t in combined["title"]]
    scores = combined[SBERT_COL].values
    colors = ["#DD8452" if s < 0 else "#4C72B0" for s in scores]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(scores)), scores, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(n - 0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("SBERT alignment score")
    ax.set_title(f"JMLR — Top {n} and bottom {n} papers by alignment score", fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_topic_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of paper count per topic, colored by mean alignment score.

    Args:
        df: DataFrame with topic_id, topic_label, and sbert_alignment_score.
        save_path: If provided, save figure to this path.
    """
    summary = (
        df[df["topic_id"] != -1]
        .groupby("topic_label")
        .agg(n_papers=("title", "count"), mean_alignment=("sbert_alignment_score", "mean"))
        .reset_index()
        .sort_values("n_papers", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.RdYlGn(
        (summary["mean_alignment"] - summary["mean_alignment"].min()) /
        (summary["mean_alignment"].max() - summary["mean_alignment"].min())
    )
    bars = ax.barh(summary["topic_label"], summary["n_papers"], color=colors, edgecolor="white")
    ax.set_xlabel("Number of papers")
    ax.set_title("JMLR — Papers per topic (color = mean alignment score)", fontsize=13)

    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn",
        norm=plt.Normalize(summary["mean_alignment"].min(), summary["mean_alignment"].max()),
    )
    plt.colorbar(sm, ax=ax, label="Mean SBERT alignment score", shrink=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_topic_drift(
    df: pd.DataFrame,
    top_n_topics: int = 6,
    save_path: Optional[str] = None,
) -> None:
    """Line chart of each topic's share of publications per year.

    Args:
        df: DataFrame with topic_label, year, and title columns.
        top_n_topics: How many of the largest topics to plot.
        save_path: If provided, save figure to this path.
    """
    top_topics = (
        df[df["topic_id"] != -1]
        .groupby("topic_label")["title"]
        .count()
        .nlargest(top_n_topics)
        .index.tolist()
    )

    yearly = (
        df[df["topic_label"].isin(top_topics)]
        .groupby(["year", "topic_label"])["title"]
        .count()
        .reset_index(name="count")
    )
    totals = df.groupby("year")["title"].count().reset_index(name="total")
    yearly = yearly.merge(totals, on="year")
    yearly["share"] = yearly["count"] / yearly["total"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for topic, group in yearly.groupby("topic_label"):
        ax.plot(group["year"], group["share"], marker="o", linewidth=2, label=topic)

    ax.set_title(f"JMLR — Publication share of top {top_n_topics} topics over time", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of papers")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()