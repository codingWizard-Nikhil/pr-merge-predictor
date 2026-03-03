# GitHub PR Merge Predictor

Machine learning model that predicts whether a GitHub pull request will be merged based on metadata features, achieving **90.7% test accuracy**.

## Overview

This project analyzes 10,000 pull requests from `pytorch/pytorch` and `microsoft/vscode` to predict merge outcomes using Random Forest classification.

## Key Results

- **Best Model:** Random Forest
- **Test Accuracy:** 90.7%
- **Precision/Recall:** 0.90-0.91
- **Top Features:** Comments (-0.52 correlation), Time Open (-0.29), Description Length (-0.12)

## Project Structure
```
pr-merge-predictor/
├── collect_data.py          # GitHub API data collection
├── explore_data.py          # Exploratory data analysis
├── engineer_features.py     # Feature engineering
├── prepare_data.py          # Data cleaning & train/test split
├── train_model.py           # Model training & evaluation
└── pr_merge_model.pkl       # Saved Random Forest model
```

## Features Engineered

1. **Time-based:** Hour created, day of week, time PR stayed open
2. **Derived:** Total lines changed, lines per commit, description length
3. **Code metrics:** Additions, deletions, changed files, commits, comments

## Model Comparison

| Model | Training Acc | Test Acc | F1 Score |
|-------|--------------|----------|----------|
| Logistic Regression | 88.2% | 87.6% | 0.88 |
| Random Forest | 100% | **90.7%** | **0.91** |
| XGBoost | 99.2% | 90.0% | 0.90 |

## Feature Importance

Top 3 features driving 78% of predictions:
1. **Comments** (43.5%) - Discussion volume
2. **Time Open** (25.0%) - Hours between creation and closure
3. **Description Length** (10.6%) - PR description character count

## Technologies

- **ML:** scikit-learn, XGBoost
- **Data:** pandas, PyGithub
- **Deployment:** FastAPI (local testing)

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run data collection (requires GitHub token in config.py)
python3 collect_data.py

# Train models
python3 train_model.py
```

## Key Insights

- **High comment volume correlates with rejection:** PRs with many comments (indicating debate/issues) are less likely to merge
- **Quick resolutions indicate acceptance:** PRs that close quickly tend to be merged; prolonged discussion suggests problems
- **Description length weakly predicts rejection:** Longer descriptions may indicate complexity or defensive justification


## Future Improvements

- Collect full PR history to calculate accurate author experience metrics
- Deploy to cloud platform (Railway, Render) for live predictions
- Build simple web UI to significantly improve user experience