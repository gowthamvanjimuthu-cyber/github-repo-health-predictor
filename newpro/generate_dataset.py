"""
Generate a sample training dataset for repository activity classification.

Run: python generate_dataset.py
"""

import pandas as pd
from pathlib import Path

from config import DATA_DIR, DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def create_training_dataset() -> pd.DataFrame:
    """
    Manually curated sample dataset mimicking real GitHub repository metrics.
    Active repos: high commits, PRs, closed issues, recent activity.
    Inactive repos: low commits, few PRs, stale last_commit_days_ago.
    """
    data = [
        # Active repositories
        {"commits_per_week": 45, "pull_requests": 28, "issues_opened": 120, "issues_closed": 115,
         "contribution_days": 280, "contributors_count": 85, "stars": 52000, "forks": 12000,
         "watchers": 2100, "last_commit_days_ago": 1, "status": "Active"},
        {"commits_per_week": 32, "pull_requests": 18, "issues_opened": 90, "issues_closed": 88,
         "contribution_days": 250, "contributors_count": 42, "stars": 15000, "forks": 3200,
         "watchers": 800, "last_commit_days_ago": 2, "status": "Active"},
        {"commits_per_week": 25, "pull_requests": 12, "issues_opened": 60, "issues_closed": 55,
         "contribution_days": 200, "contributors_count": 25, "stars": 8000, "forks": 1500,
         "watchers": 400, "last_commit_days_ago": 3, "status": "Active"},
        {"commits_per_week": 18, "pull_requests": 10, "issues_opened": 45, "issues_closed": 42,
         "contribution_days": 180, "contributors_count": 18, "stars": 3500, "forks": 600,
         "watchers": 200, "last_commit_days_ago": 5, "status": "Active"},
        {"commits_per_week": 15, "pull_requests": 8, "issues_opened": 35, "issues_closed": 32,
         "contribution_days": 150, "contributors_count": 12, "stars": 2000, "forks": 400,
         "watchers": 120, "last_commit_days_ago": 7, "status": "Active"},
        {"commits_per_week": 12, "pull_requests": 6, "issues_opened": 28, "issues_closed": 25,
         "contribution_days": 120, "contributors_count": 8, "stars": 1200, "forks": 250,
         "watchers": 80, "last_commit_days_ago": 10, "status": "Active"},
        {"commits_per_week": 10, "pull_requests": 5, "issues_opened": 22, "issues_closed": 20,
         "contribution_days": 100, "contributors_count": 6, "stars": 800, "forks": 150,
         "watchers": 50, "last_commit_days_ago": 14, "status": "Active"},
        {"commits_per_week": 8, "pull_requests": 4, "issues_opened": 18, "issues_closed": 16,
         "contribution_days": 90, "contributors_count": 5, "stars": 500, "forks": 100,
         "watchers": 35, "last_commit_days_ago": 20, "status": "Active"},
        {"commits_per_week": 6, "pull_requests": 3, "issues_opened": 15, "issues_closed": 14,
         "contribution_days": 75, "contributors_count": 4, "stars": 350, "forks": 70,
         "watchers": 25, "last_commit_days_ago": 25, "status": "Active"},
        {"commits_per_week": 5, "pull_requests": 2, "issues_opened": 12, "issues_closed": 11,
         "contribution_days": 60, "contributors_count": 3, "stars": 200, "forks": 45,
         "watchers": 18, "last_commit_days_ago": 30, "status": "Active"},
        # Borderline active
        {"commits_per_week": 4, "pull_requests": 2, "issues_opened": 10, "issues_closed": 9,
         "contribution_days": 50, "contributors_count": 3, "stars": 150, "forks": 30,
         "watchers": 15, "last_commit_days_ago": 35, "status": "Active"},
        {"commits_per_week": 3, "pull_requests": 1, "issues_opened": 8, "issues_closed": 7,
         "contribution_days": 45, "contributors_count": 2, "stars": 100, "forks": 20,
         "watchers": 12, "last_commit_days_ago": 40, "status": "Active"},
        # Inactive repositories
        {"commits_per_week": 0, "pull_requests": 0, "issues_opened": 5, "issues_closed": 1,
         "contribution_days": 10, "contributors_count": 1, "stars": 50, "forks": 8,
         "watchers": 5, "last_commit_days_ago": 365, "status": "Inactive"},
        {"commits_per_week": 0, "pull_requests": 0, "issues_opened": 3, "issues_closed": 0,
         "contribution_days": 5, "contributors_count": 1, "stars": 20, "forks": 3,
         "watchers": 2, "last_commit_days_ago": 500, "status": "Inactive"},
        {"commits_per_week": 1, "pull_requests": 0, "issues_opened": 8, "issues_closed": 2,
         "contribution_days": 15, "contributors_count": 1, "stars": 80, "forks": 12,
         "watchers": 8, "last_commit_days_ago": 200, "status": "Inactive"},
        {"commits_per_week": 0, "pull_requests": 0, "issues_opened": 12, "issues_closed": 3,
         "contribution_days": 8, "contributors_count": 2, "stars": 120, "forks": 15,
         "watchers": 10, "last_commit_days_ago": 300, "status": "Inactive"},
        {"commits_per_week": 1, "pull_requests": 0, "issues_opened": 6, "issues_closed": 1,
         "contribution_days": 12, "contributors_count": 1, "stars": 40, "forks": 5,
         "watchers": 4, "last_commit_days_ago": 180, "status": "Inactive"},
        {"commits_per_week": 0, "pull_requests": 0, "issues_opened": 2, "issues_closed": 0,
         "contribution_days": 3, "contributors_count": 1, "stars": 10, "forks": 1,
         "watchers": 1, "last_commit_days_ago": 700, "status": "Inactive"},
        {"commits_per_week": 2, "pull_requests": 0, "issues_opened": 15, "issues_closed": 4,
         "contribution_days": 20, "contributors_count": 2, "stars": 200, "forks": 25,
         "watchers": 15, "last_commit_days_ago": 150, "status": "Inactive"},
        {"commits_per_week": 1, "pull_requests": 0, "issues_opened": 4, "issues_closed": 1,
         "contribution_days": 7, "contributors_count": 1, "stars": 30, "forks": 4,
         "watchers": 3, "last_commit_days_ago": 250, "status": "Inactive"},
        {"commits_per_week": 0, "pull_requests": 0, "issues_opened": 1, "issues_closed": 0,
         "contribution_days": 2, "contributors_count": 1, "stars": 5, "forks": 0,
         "watchers": 1, "last_commit_days_ago": 900, "status": "Inactive"},
        {"commits_per_week": 1, "pull_requests": 0, "issues_opened": 10, "issues_closed": 2,
         "contribution_days": 18, "contributors_count": 1, "stars": 60, "forks": 8,
         "watchers": 6, "last_commit_days_ago": 120, "status": "Inactive"},
        {"commits_per_week": 0, "pull_requests": 0, "issues_opened": 7, "issues_closed": 1,
         "contribution_days": 6, "contributors_count": 1, "stars": 25, "forks": 2,
         "watchers": 2, "last_commit_days_ago": 400, "status": "Inactive"},
        {"commits_per_week": 2, "pull_requests": 0, "issues_opened": 20, "issues_closed": 5,
         "contribution_days": 25, "contributors_count": 2, "stars": 150, "forks": 18,
         "watchers": 12, "last_commit_days_ago": 90, "status": "Inactive"},
        # More active samples for balance
        {"commits_per_week": 50, "pull_requests": 35, "issues_opened": 150, "issues_closed": 145,
         "contribution_days": 300, "contributors_count": 100, "stars": 80000, "forks": 20000,
         "watchers": 3500, "last_commit_days_ago": 0, "status": "Active"},
        {"commits_per_week": 22, "pull_requests": 14, "issues_opened": 55, "issues_closed": 50,
         "contribution_days": 190, "contributors_count": 20, "stars": 5000, "forks": 900,
         "watchers": 300, "last_commit_days_ago": 4, "status": "Active"},
        {"commits_per_week": 7, "pull_requests": 3, "issues_opened": 16, "issues_closed": 15,
         "contribution_days": 85, "contributors_count": 4, "stars": 400, "forks": 80,
         "watchers": 30, "last_commit_days_ago": 28, "status": "Active"},
        {"commits_per_week": 0, "pull_requests": 0, "issues_opened": 0, "issues_closed": 0,
         "contribution_days": 0, "contributors_count": 1, "stars": 3, "forks": 0,
         "watchers": 1, "last_commit_days_ago": 1000, "status": "Inactive"},
        {"commits_per_week": 1, "pull_requests": 0, "issues_opened": 3, "issues_closed": 0,
         "contribution_days": 4, "contributors_count": 1, "stars": 15, "forks": 2,
         "watchers": 2, "last_commit_days_ago": 600, "status": "Inactive"},
    ]

    df = pd.DataFrame(data)
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = create_training_dataset()
    df.to_csv(DATASET_PATH, index=False)

    print(f"Dataset saved to: {DATASET_PATH}")
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:\n{df[TARGET_COLUMN].value_counts()}")
    print(f"\nFeature columns: {FEATURE_COLUMNS}")
    print("\nSample rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
