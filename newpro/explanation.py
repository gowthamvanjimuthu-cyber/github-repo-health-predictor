"""
Generate human-readable explanations for Active/Inactive predictions.
"""


def explain_prediction(stats: dict, prediction: str) -> str:
    """
    Explain why a repository was predicted as Active or Inactive
    based on activity metrics.
    """
    commits = stats.get("commits_per_week", 0)
    prs = stats.get("pull_requests", 0)
    opened = stats.get("issues_opened", 0)
    closed = stats.get("issues_closed", 0)
    contrib_days = stats.get("contribution_days", 0)
    last_commit = stats.get("last_commit_days_ago", 999)
    contributors = stats.get("contributors_count", 0)

    reasons = []

    if prediction == "Active":
        if commits >= 5:
            reasons.append(f"Strong commit activity ({commits} commits/week)")
        elif commits >= 2:
            reasons.append(f"Moderate commit activity ({commits} commits/week)")

        if prs >= 3:
            reasons.append(f"Active pull request workflow ({prs} PRs)")
        elif prs >= 1:
            reasons.append(f"Some pull request activity ({prs} PRs)")

        if closed >= opened * 0.5 and closed > 0:
            reasons.append(
                f"Good issue resolution rate ({closed} closed vs {opened} opened)"
            )

        if contrib_days >= 30:
            reasons.append(f"Regular contributions over {contrib_days} days")
        elif contrib_days >= 10:
            reasons.append(f"Some contribution activity ({contrib_days} days)")

        if last_commit <= 30:
            reasons.append(f"Recent commits (last commit {last_commit} days ago)")
        elif last_commit <= 60:
            reasons.append(f"Fairly recent activity (last commit {last_commit} days ago)")

        if contributors >= 3:
            reasons.append(f"Multiple contributors ({contributors})")

        if not reasons:
            reasons.append(
                "Overall activity metrics indicate ongoing maintenance and development"
            )

        header = (
            "This repository is predicted as **Active** because it shows signs of "
            "regular development and maintenance:"
        )
    else:
        if commits <= 1:
            reasons.append(f"Very low commit activity ({commits} commits/week)")

        if prs == 0:
            reasons.append("No pull request activity")
        elif prs <= 1:
            reasons.append(f"Minimal pull request activity ({prs} PRs)")

        if opened > closed and opened > 5:
            reasons.append(
                f"Many unresolved issues ({opened} opened vs {closed} closed)"
            )

        if last_commit > 90:
            reasons.append(
                f"Stale repository — last commit was {last_commit} days ago"
            )
        elif last_commit > 60:
            reasons.append(f"No recent commits (last commit {last_commit} days ago)")

        if contrib_days < 15:
            reasons.append(f"Low contribution frequency ({contrib_days} days)")

        if contributors <= 2:
            reasons.append(f"Few contributors ({contributors})")

        if not reasons:
            reasons.append(
                "Overall metrics suggest limited development and maintenance activity"
            )

        header = (
            "This repository is predicted as **Inactive** because it shows signs of "
            "low or abandoned development:"
        )

    bullet_points = "\n".join(f"  • {r}" for r in reasons)
    return f"{header}\n{bullet_points}"


def activity_score_summary(stats: dict) -> dict:
    """Compute simple activity scores for dashboard display."""
    commit_score = min(100, stats.get("commits_per_week", 0) * 10)
    pr_score = min(100, stats.get("pull_requests", 0) * 15)
    issue_resolution = 0
    opened = stats.get("issues_opened", 0)
    closed = stats.get("issues_closed", 0)
    if opened + closed > 0:
        issue_resolution = min(100, (closed / (opened + closed)) * 100)
    recency = max(0, 100 - stats.get("last_commit_days_ago", 999) / 10)
    contrib_score = min(100, stats.get("contribution_days", 0) / 3)

    overall = (commit_score + pr_score + issue_resolution + recency + contrib_score) / 5
    return {
        "commit_score": round(commit_score, 1),
        "pr_score": round(pr_score, 1),
        "issue_resolution_score": round(issue_resolution, 1),
        "recency_score": round(recency, 1),
        "contribution_score": round(contrib_score, 1),
        "overall_activity_score": round(overall, 1),
    }
