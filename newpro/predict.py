"""
CLI: Predict repository Active/Inactive status from a GitHub URL.

Usage:
    python predict.py https://github.com/python/cpython
    set GITHUB_TOKEN=your_token && python predict.py <url>
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from config import FEATURE_COLUMNS, LABEL_ENCODER_PATH, MODEL_PATH, OUTPUT_DIR
from explanation import activity_score_summary, explain_prediction
from github_fetcher import GitHubRepositoryFetcher
from visualizations import plot_activity_bar, plot_feature_importance, plot_issue_pie


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python train_model.py"
        )
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder


def display_statistics(stats: dict):
    """Print repository statistics in a clean table."""
    rows = [
        ["Repository", stats.get("full_name", "N/A")],
        ["URL", stats.get("url", "N/A")],
        ["Language", stats.get("language", "N/A")],
        ["Commits per week", stats.get("commits_per_week", 0)],
        ["Pull Requests", stats.get("pull_requests", 0)],
        ["Issues Opened", stats.get("issues_opened", 0)],
        ["Issues Closed", stats.get("issues_closed", 0)],
        ["Contributors", stats.get("contributors_count", 0)],
        ["Contribution Days", stats.get("contribution_days", 0)],
        ["Last Commit (days ago)", stats.get("last_commit_days_ago", "N/A")],
        ["Stars", stats.get("stars", 0)],
        ["Forks", stats.get("forks", 0)],
        ["Watchers", stats.get("watchers", 0)],
    ]
    print("\n" + "=" * 60)
    print("REPOSITORY STATISTICS")
    print("=" * 60)
    print(f"{'Metric':<28} | {'Value'}")
    print("-" * 45)
    for metric, value in rows:
        print(f"{metric:<28} | {value}")


def predict_repository(repo_url: str, token: str | None = None, save_plots: bool = True):
    """Fetch stats, predict status, display results."""
    print(f"\nFetching data for: {repo_url}")
    fetcher = GitHubRepositoryFetcher(token=token)
    stats = fetcher.fetch(repo_url)

    display_statistics(stats)

    model, label_encoder = load_model()
    features = fetcher.to_feature_vector(stats)
    X = pd.DataFrame([features])[FEATURE_COLUMNS]

    prediction_idx = model.predict(X)[0]
    prediction = label_encoder.inverse_transform([prediction_idx])[0]
    probabilities = model.predict_proba(X)[0]
    prob_dict = dict(zip(label_encoder.classes_, probabilities))

    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)
    print(f"\n  Repository Status = {prediction}")
    print(f"  Confidence: Active={prob_dict.get('Active', 0):.1%}, "
          f"Inactive={prob_dict.get('Inactive', 0):.1%}")

    print("\n" + "-" * 60)
    print("EXPLANATION")
    print("-" * 60)
    print(explain_prediction(stats, prediction))

    scores = activity_score_summary(stats)
    print("\n" + "-" * 60)
    print("ACTIVITY SCORES")
    print("-" * 60)
    for key, val in scores.items():
        print(f"  {key.replace('_', ' ').title():30} {val}")

    if save_plots:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        repo_slug = stats.get("full_name", "repo").replace("/", "_")
        plot_activity_bar(stats, OUTPUT_DIR / f"{repo_slug}_activity.png")
        plot_issue_pie(stats, OUTPUT_DIR / f"{repo_slug}_issues.png")
        plot_feature_importance(
            model, FEATURE_COLUMNS, OUTPUT_DIR / f"{repo_slug}_importance.png"
        )
        print(f"\nCharts saved to: {OUTPUT_DIR}")

    return stats, prediction, prob_dict


def main():
    parser = argparse.ArgumentParser(
        description="Predict GitHub repository Active/Inactive status"
    )
    parser.add_argument("url", help="GitHub repository URL")
    parser.add_argument("--token", help="GitHub API token (or set GITHUB_TOKEN)")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    try:
        predict_repository(args.url, token=args.token, save_plots=not args.no_plots)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
