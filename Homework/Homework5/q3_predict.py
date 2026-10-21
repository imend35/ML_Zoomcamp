
"""
Machine Learning Zoomcamp 2026 - Homework 5
Question 3: Load the trained model and predict conversion probability.
"""

import pickle
from pathlib import Path


# Define the location of my trained model artifact.
MODEL_PATH = Path(__file__).resolve().parent / "pipeline.bin"


# Prepare the lead information provided in Question 3.
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}


def main():
    # Load the verified model artifact from my project directory.
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    # Calculate the probability of conversion for the given lead.
    conversion_probability = float(model.predict_proba([lead])[0, 1])

    # Display the probability and its rounded homework answer.
    print(f"Conversion probability: {conversion_probability:.6f}")
    print(f"Question 3 answer: {conversion_probability:.3f}")


if __name__ == "__main__":
    main()
