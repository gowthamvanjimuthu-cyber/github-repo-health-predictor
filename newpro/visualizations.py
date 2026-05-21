"""
Matplotlib visualizations for repository activity analysis.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_activity_bar(stats: dict, save_path: Path | None = None) -> plt.Figure:
    """Bar chart for key repository activity metrics."""
    labels = [
        "Commits/Week",
        "Pull Requests",
        "Issues Opened",
        "Issues Closed",
        "Contributors",
        "Contrib. Days",
    ]
    values = [
        stats.get("commits_per_week", 0),
        stats.get("pull_requests", 0),
        stats.get("issues_opened", 0),
        stats.get("issues_closed", 0),
        stats.get("contributors_count", 0),
        stats.get("contribution_days", 0),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels)))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Repository Activity Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02 + 0.1,
            f"{val:.0f}" if isinstance(val, float) and val == int(val) else f"{val}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_issue_pie(stats: dict, save_path: Path | None = None) -> plt.Figure:
    """Pie chart for issue status (opened vs closed)."""
    opened = stats.get("issues_opened", 0)
    closed = stats.get("issues_closed", 0)

    if opened == 0 and closed == 0:
        opened, closed = 1, 0  # avoid empty pie

    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [opened, closed]
    labels = [f"Opened ({opened})", f"Closed ({closed})"]
    colors = ["#ff6b6b", "#51cf66"]
    explode = (0.05, 0)

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=False,
        startangle=90,
    )
    ax.set_title("Issue Status Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(model, feature_names: list, save_path: Path | None = None) -> plt.Figure:
    """Horizontal bar chart of feature importance from trained model."""
    if not hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Feature importance not available", ha="center", va="center")
        return fig

    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(indices))),
    )
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_engagement_metrics(stats: dict, save_path: Path | None = None) -> plt.Figure:
    """Bar chart for stars, forks, watchers."""
    labels = ["Stars", "Forks", "Watchers"]
    values = [
        stats.get("stars", 0),
        stats.get("forks", 0),
        stats.get("watchers", 0),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#ffd43b", "#74c0fc", "#b197fc"])
    ax.set_title("Repository Engagement", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
