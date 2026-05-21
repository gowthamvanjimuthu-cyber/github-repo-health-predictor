"""
Fetch GitHub repository activity metrics using PyGithub.

Requires GITHUB_TOKEN environment variable for higher rate limits.
"""

import os
import re
from datetime import datetime, timezone
from typing import Optional

from github import Github, GithubException


def parse_repo_url(url: str) -> tuple[str, str]:
    """Extract owner and repo name from a GitHub URL."""
    url = url.strip().rstrip("/")
    patterns = [
        r"github\.com/([^/]+)/([^/]+?)(?:\.git)?$",
        r"^([^/]+)/([^/]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            owner, repo = match.group(1), match.group(2)
            return owner, repo.replace(".git", "")
    raise ValueError(
        f"Invalid GitHub URL: {url}. "
        "Expected format: https://github.com/owner/repo"
    )


class GitHubRepositoryFetcher:
    """Fetches repository statistics from the GitHub API."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.client = Github(self.token) if self.token else Github()

    def fetch(self, repo_url: str) -> dict:
        """
        Fetch all required metrics for a repository.

        Returns a dict with keys matching FEATURE_COLUMNS plus metadata.
        """
        owner, repo_name = parse_repo_url(repo_url)
        try:
            repo = self.client.get_repo(f"{owner}/{repo_name}")
        except GithubException as e:
            raise ValueError(f"Could not access repository {owner}/{repo_name}: {e.data}") from e

        stats = {
            "owner": owner,
            "repo_name": repo_name,
            "full_name": repo.full_name,
            "url": repo.html_url,
            "description": repo.description or "",
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "watchers": repo.watchers_count,
            "open_issues": repo.open_issues_count,
            "created_at": repo.created_at.isoformat() if repo.created_at else "",
            "updated_at": repo.updated_at.isoformat() if repo.updated_at else "",
            "language": repo.language or "Unknown",
        }

        stats.update(self._fetch_commit_metrics(repo))
        stats.update(self._fetch_pull_requests(repo))
        stats.update(self._fetch_issues(repo))
        stats.update(self._fetch_contributors(repo))

        return stats

    def _fetch_commit_metrics(self, repo) -> dict:
        """Compute commits per week and days since last commit."""
        commits_per_week = 0
        last_commit_days_ago = 999
        contribution_days = 0

        try:
            commits = repo.get_commits()
            if commits.totalCount > 0:
                latest = commits[0]
                if latest.commit.author.date:
                    delta = datetime.now(timezone.utc) - latest.commit.author.date
                    last_commit_days_ago = max(0, delta.days)

                # Sample recent commits for weekly rate (last 90 days)
                recent_count = 0
                cutoff = datetime.now(timezone.utc).timestamp() - (90 * 86400)
                commit_dates = set()
                for i, commit in enumerate(commits):
                    if i >= 200:
                        break
                    date = commit.commit.author.date
                    if date and date.timestamp() >= cutoff:
                        recent_count += 1
                        commit_dates.add(date.date())
                contribution_days = len(commit_dates)
                commits_per_week = round((recent_count / 13), 1)  # ~13 weeks in 90 days
        except GithubException:
            pass

        return {
            "commits_per_week": commits_per_week,
            "last_commit_days_ago": last_commit_days_ago,
            "contribution_days": contribution_days,
        }

    def _fetch_pull_requests(self, repo) -> dict:
        """Count open and closed pull requests."""
        pull_requests = 0
        try:
            open_prs = repo.get_pulls(state="open")
            closed_prs = repo.get_pulls(state="closed")
            pull_requests = min(open_prs.totalCount + closed_prs.totalCount, 500)
        except GithubException:
            pass
        return {"pull_requests": pull_requests}

    def _fetch_issues(self, repo) -> dict:
        """Count opened and closed issues (excluding PRs)."""
        issues_opened = 0
        issues_closed = 0
        try:
            open_issues = repo.get_issues(state="open")
            closed_issues = repo.get_issues(state="closed")
            for issue in open_issues:
                if not issue.pull_request:
                    issues_opened += 1
                if issues_opened >= 250:
                    break
            for issue in closed_issues:
                if not issue.pull_request:
                    issues_closed += 1
                if issues_closed >= 250:
                    break
        except GithubException:
            issues_opened = repo.open_issues_count
        return {
            "issues_opened": issues_opened,
            "issues_closed": issues_closed,
        }

    def _fetch_contributors(self, repo) -> dict:
        """Count repository contributors."""
        contributors_count = 0
        try:
            contributors = repo.get_contributors()
            contributors_count = min(contributors.totalCount, 500)
        except GithubException:
            contributors_count = 1
        return {"contributors_count": contributors_count}

    def to_feature_vector(self, stats: dict) -> dict:
        """Extract feature columns for ML prediction."""
        return {col: stats.get(col, 0) for col in [
            "commits_per_week",
            "pull_requests",
            "issues_opened",
            "issues_closed",
            "contribution_days",
            "contributors_count",
            "stars",
            "forks",
            "watchers",
            "last_commit_days_ago",
        ]}


def fetch_repository(repo_url: str, token: Optional[str] = None) -> dict:
    """Convenience function to fetch repository stats."""
    fetcher = GitHubRepositoryFetcher(token=token)
    return fetcher.fetch(repo_url)
