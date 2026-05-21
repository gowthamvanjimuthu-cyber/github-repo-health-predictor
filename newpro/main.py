"""
Main entry point: train model and run demo prediction.

Run: python main.py
"""

import sys

from generate_dataset import main as generate_main
from train_model import train_and_evaluate


def main():
    print("Step 1: Generating training dataset...")
    generate_main()

    print("\nStep 2: Training ML models...")
    train_and_evaluate()

    print("\nStep 3: Ready for predictions!")
    print("  CLI:      python predict.py <github-url>")
    print("  Web app:  streamlit run app.py")
    print("\nSet GITHUB_TOKEN for higher API rate limits.")


if __name__ == "__main__":
    main()
