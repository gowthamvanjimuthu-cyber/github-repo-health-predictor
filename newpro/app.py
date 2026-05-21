"""
Streamlit web app: GitHub Repository Active/Inactive Prediction Dashboard.

Run: streamlit run app.py
"""

import io
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import FEATURE_COLUMNS, LABEL_ENCODER_PATH, MODEL_PATH, OUTPUT_DIR
from explanation import activity_score_summary, explain_prediction
from github_fetcher import GitHubRepositoryFetcher
from visualizations import (
    plot_activity_bar,
    plot_engagement_metrics,
    plot_feature_importance,
    plot_issue_pie,
)


st.set_page_config(
    page_title="GitHub Repo Activity Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("GitHub Repository Active/Inactive Prediction")
st.markdown(
    "Predict whether a GitHub repository is **Active** or **Inactive** "
    "using machine learning based on real-time activity metrics."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(LABEL_ENCODER_PATH)


def main():
    with st.sidebar:
        st.header("Configuration")
        token = st.text_input(
            "GitHub Token",
            value=os.environ.get("GITHUB_TOKEN", ""),
            type="password",
            help="Create at: github.com/settings/tokens (repo scope)",
        )
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "- Fetches live repo metrics via GitHub API\n"
            "- Random Forest classifier\n"
            "- Visual activity dashboard"
        )

    model, label_encoder = load_model()
    if model is None:
        st.error(
            "Model not trained yet. Run:\n\n"
            "```\npython generate_dataset.py\npython train_model.py\n```"
        )
        return

    tab1, tab2, tab3 = st.tabs(["Single Repository", "Multiple Repositories", "Model Info"])

    with tab1:
        url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/python/cpython",
        )
        if st.button("Analyze Repository", type="primary") and url:
            analyze_single(url.strip(), token, model, label_encoder)

    with tab2:
        urls_text = st.text_area(
            "GitHub URLs (one per line)",
            placeholder="https://github.com/python/cpython\nhttps://github.com/torvalds/linux",
            height=120,
        )
        if st.button("Analyze All", type="primary") and urls_text:
            urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]
            results = []
            progress = st.progress(0)
            for i, u in enumerate(urls):
                try:
                    fetcher = GitHubRepositoryFetcher(token=token or None)
                    stats = fetcher.fetch(u)
                    features = fetcher.to_feature_vector(stats)
                    X = pd.DataFrame([features])[FEATURE_COLUMNS]
                    pred_idx = model.predict(X)[0]
                    pred = label_encoder.inverse_transform([pred_idx])[0]
                    stats["prediction"] = pred
                    stats["input_url"] = u
                    results.append(stats)
                except Exception as e:
                    st.warning(f"Failed for {u}: {e}")
                progress.progress((i + 1) / len(urls))

            if results:
                df = pd.DataFrame(results)
                display_cols = [
                    "full_name", "commits_per_week", "pull_requests",
                    "issues_opened", "issues_closed", "stars", "prediction",
                ]
                st.dataframe(df[[c for c in display_cols if c in df.columns]], use_container_width=True)

                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name=f"repo_analytics_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                )

    with tab3:
        st.subheader("Trained Model")
        st.write(f"**Model path:** `{MODEL_PATH}`")
        st.write(f"**Features:** {', '.join(FEATURE_COLUMNS)}")
        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)
            st.dataframe(imp_df, use_container_width=True)
            fig = plot_feature_importance(model, FEATURE_COLUMNS)
            st.pyplot(fig)
            plt.close(fig)


def analyze_single(url: str, token: str, model, label_encoder):
    with st.spinner("Fetching repository data from GitHub..."):
        try:
            fetcher = GitHubRepositoryFetcher(token=token or None)
            stats = fetcher.fetch(url)
        except Exception as e:
            st.error(f"Failed to fetch repository: {e}")
            return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📦 {stats.get('full_name', url)}")
        if stats.get("description"):
            st.caption(stats["description"])

        metrics_df = pd.DataFrame([
            {"Metric": "Commits per week", "Value": stats.get("commits_per_week", 0)},
            {"Metric": "Pull Requests", "Value": stats.get("pull_requests", 0)},
            {"Metric": "Issues Opened", "Value": stats.get("issues_opened", 0)},
            {"Metric": "Issues Closed", "Value": stats.get("issues_closed", 0)},
            {"Metric": "Contributors", "Value": stats.get("contributors_count", 0)},
            {"Metric": "Contribution Days", "Value": stats.get("contribution_days", 0)},
            {"Metric": "Last Commit (days ago)", "Value": stats.get("last_commit_days_ago", "N/A")},
            {"Metric": "Stars", "Value": stats.get("stars", 0)},
            {"Metric": "Forks", "Value": stats.get("forks", 0)},
            {"Metric": "Watchers", "Value": stats.get("watchers", 0)},
            {"Metric": "Language", "Value": stats.get("language", "N/A")},
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    features = fetcher.to_feature_vector(stats)
    X = pd.DataFrame([features])[FEATURE_COLUMNS]
    pred_idx = model.predict(X)[0]
    prediction = label_encoder.inverse_transform([pred_idx])[0]
    probs = dict(zip(label_encoder.classes_, model.predict_proba(X)[0]))

    with col2:
        st.subheader("Prediction")
        color = "green" if prediction == "Active" else "orange"
        st.markdown(
            f"<h2 style='color:{color};'>Status: {prediction}</h2>",
            unsafe_allow_html=True,
        )
        st.metric("Active probability", f"{probs.get('Active', 0):.1%}")
        st.metric("Inactive probability", f"{probs.get('Inactive', 0):.1%}")

        scores = activity_score_summary(stats)
        st.metric("Overall Activity Score", f"{scores['overall_activity_score']}/100")

    st.markdown("### Explanation")
    st.info(explain_prediction(stats, prediction))

    st.markdown("### Visualizations")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig1 = plot_activity_bar(stats)
        st.pyplot(fig1)
        plt.close(fig1)
    with c2:
        fig2 = plot_issue_pie(stats)
        st.pyplot(fig2)
        plt.close(fig2)
    with c3:
        fig3 = plot_engagement_metrics(stats)
        st.pyplot(fig3)
        plt.close(fig3)

    fig4 = plot_feature_importance(model, FEATURE_COLUMNS)
    st.pyplot(fig4)
    plt.close(fig4)

    # CSV export
    export_df = pd.DataFrame([{**stats, "prediction": prediction, **probs}])
    st.download_button(
        "Export Analytics (CSV)",
        export_df.to_csv(index=False),
        file_name=f"{stats.get('full_name', 'repo').replace('/', '_')}_analytics.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
