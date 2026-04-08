"""
NBA MVP Prediction Model — With Defensive Metrics
===================================================
Loads MVP candidate stats from a local CSV dataset,
trains a KNN classifier, and measures how much defensive metrics
contribute to predicting MVP winners.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# Step 1: Load data from CSV
# -----------------------------------------------------------------------
def load_data():
    """Load NBA MVP candidate stats from CSV."""
    csv_path = os.path.join(DATA_DIR, "nba_mvp_stats.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


# -----------------------------------------------------------------------
# Step 2: Label MVPs using MVP_Share column
# -----------------------------------------------------------------------
def label_mvps(df):
    """Add MVP column: 1 for the top vote-getter each season."""
    df["MVP"] = 0
    for season in df["Season"].unique():
        mask = df["Season"] == season
        top_idx = df.loc[mask, "MVP_Share"].idxmax()
        df.loc[top_idx, "MVP"] = 1
    return df


# -----------------------------------------------------------------------
# Step 3: Feature engineering
# -----------------------------------------------------------------------
OFFENSIVE_FEATURES = ["PTS", "AST"]
OVERALL_FEATURES = ["PER", "WS"]
DEFENSIVE_FEATURES = ["STL", "BLK", "DRB", "DWS", "DBPM"]

ALL_FEATURES = OFFENSIVE_FEATURES + OVERALL_FEATURES + DEFENSIVE_FEATURES


def add_def_score(df):
    """Create a composite Defense Score."""
    df["DEF_SCORE"] = (
        df["STL"] * 1.5 +
        df["BLK"] * 1.5 +
        df["DWS"] * 2 +
        df["DBPM"] * 2
    )
    return df


# -----------------------------------------------------------------------
# Step 4: Train and evaluate models
# -----------------------------------------------------------------------
def train_and_evaluate(df, feature_cols, label="Model"):
    """Train KNN with cross-validation and return accuracy."""
    X = df[feature_cols].dropna()
    y = df.loc[X.index, "MVP"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X_scaled, y, cv=5, scoring="accuracy")

    print(f"  {label}: accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean(), knn, scaler, X, y


# -----------------------------------------------------------------------
# Step 5: Feature importance via permutation
# -----------------------------------------------------------------------
def feature_importance(df, feature_cols):
    """Compute feature importance by measuring accuracy drop when each feature is shuffled."""
    X = df[feature_cols].dropna()
    y = df.loc[X.index, "MVP"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y)
    baseline = knn.score(X_scaled, y)

    importances = {}
    for i, col in enumerate(feature_cols):
        X_shuffled = X_scaled.copy()
        np.random.seed(42)
        X_shuffled[:, i] = np.random.permutation(X_shuffled[:, i])
        shuffled_score = knn.score(X_shuffled, y)
        importances[col] = baseline - shuffled_score

    return importances


# -----------------------------------------------------------------------
# Step 6: Predict current top MVP candidates
# -----------------------------------------------------------------------
def predict_top_candidates(df, knn, scaler, feature_cols, latest_year):
    """Show the top MVP candidates for the most recent season."""
    latest = df[df["Season"] == latest_year].copy()
    X_latest = latest[feature_cols].dropna()
    latest = latest.loc[X_latest.index]

    X_scaled = scaler.transform(X_latest)

    probs = knn.predict_proba(X_scaled)
    if probs.shape[1] == 2:
        latest["MVP_Prob"] = probs[:, 1]
    else:
        latest["MVP_Prob"] = probs[:, 0]

    top = latest.nlargest(10, "MVP_Prob")[["Player", "Team", "PTS", "AST", "DWS", "DBPM", "MVP_Prob"]]
    return top


# -----------------------------------------------------------------------
# Step 7: Visualizations
# -----------------------------------------------------------------------
def plot_feature_importance(importances, save_path):
    """Bar chart of feature importance."""
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    colors = []
    for feat in sorted_imp:
        if feat in DEFENSIVE_FEATURES or feat == "DEF_SCORE":
            colors.append("#e74c3c")
        elif feat in OFFENSIVE_FEATURES:
            colors.append("#3498db")
        else:
            colors.append("#2ecc71")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(sorted_imp.keys()), list(sorted_imp.values()), color=colors, edgecolor="black")
    ax.set_xlabel("Importance (accuracy drop when feature shuffled)")
    ax.set_title("Feature Importance for MVP Prediction\n(Red = Defense, Blue = Offense, Green = Overall)", fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results, save_path):
    """Bar chart comparing model accuracy with/without defense."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    accs = list(results.values())
    colors = ["#3498db", "#e74c3c", "#9b59b6"][:len(names)]

    bars = ax.bar(names, accs, color=colors, edgecolor="black", width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{acc:.4f}", ha="center", fontweight="bold")

    ax.set_ylabel("Cross-Validation Accuracy")
    ax.set_title("Does Defense Matter for MVP Prediction?", fontsize=14, fontweight="bold")
    ax.set_ylim(min(accs) - 0.02, max(accs) + 0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_mvp_profile(df, save_path):
    """Grouped bar chart: MVP winners vs non-MVPs."""
    mvps = df[df["MVP"] == 1]
    non_mvps = df[df["MVP"] == 0]

    stats = ALL_FEATURES
    mvp_means = mvps[stats].mean()
    non_mvp_means = non_mvps[stats].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(stats))
    width = 0.35

    ax.bar(x - width/2, mvp_means, width, label="MVP Winners", color="#f39c12", edgecolor="black")
    ax.bar(x + width/2, non_mvp_means, width, label="Non-MVP Players", color="#95a5a6", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(stats, rotation=45, ha="right")
    ax.set_ylabel("Average Value")
    ax.set_title("MVP Winners vs Non-MVP Players: Stat Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("NBA MVP Prediction — How Much Does Defense Matter?")
    print("=" * 60)

    # Step 1: Load data from CSV
    df = load_data()
    print(f"Seasons: {sorted(df['Season'].unique())}")

    # Step 2: Label MVPs
    df = label_mvps(df)
    mvp_count = df["MVP"].sum()
    print(f"MVP labels found: {mvp_count}")

    # Step 3: Add defense score
    df = add_def_score(df)

    # Step 4: Train models — compare with and without defense
    print("\n--- Model Comparison ---")

    offense_only = OFFENSIVE_FEATURES + OVERALL_FEATURES
    with_defense = ALL_FEATURES
    with_def_score = OFFENSIVE_FEATURES + OVERALL_FEATURES + ["DEF_SCORE"]

    acc_offense, _, _, _, _ = train_and_evaluate(df, offense_only, "Offense Only (PTS, AST, PER, WS)")
    acc_defense, knn_full, scaler_full, X_full, y_full = train_and_evaluate(df, with_defense, "With Defense (+ STL, BLK, DRB, DWS, DBPM)")
    acc_defscore, _, _, _, _ = train_and_evaluate(df, with_def_score, "With DEF_SCORE (composite)")

    results = {
        "Offense Only": acc_offense,
        "With Defense": acc_defense,
        "With DEF_SCORE": acc_defscore,
    }

    # Step 5: Feature importance
    print("\n--- Feature Importance ---")
    importances = feature_importance(df, with_defense)
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        tag = " (DEF)" if feat in DEFENSIVE_FEATURES else " (OFF)" if feat in OFFENSIVE_FEATURES else " (OVR)"
        print(f"  {feat}{tag}: {imp:.4f}")

    # Step 6: Top MVP candidates for latest year
    latest_year = df["Season"].max()
    print(f"\n--- Top MVP Candidates ({latest_year}) ---")
    knn_full.fit(scaler_full.transform(X_full), y_full)
    top = predict_top_candidates(df, knn_full, scaler_full, with_defense, latest_year)
    print(top.to_string(index=False))

    # Step 7: Visualizations
    print("\n--- Generating Charts ---")
    plot_feature_importance(importances, os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plot_model_comparison(results, os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plot_mvp_profile(df, os.path.join(OUTPUT_DIR, "mvp_profile.png"))

    # Save summary
    summary = {
        "offense_only_accuracy": round(acc_offense, 4),
        "with_defense_accuracy": round(acc_defense, 4),
        "with_def_score_accuracy": round(acc_defscore, 4),
        "top_feature": max(importances, key=importances.get),
        "top_defensive_feature": max(
            {k: v for k, v in importances.items() if k in DEFENSIVE_FEATURES},
            key=lambda k: importances[k]
        ),
        "seasons": len(df["Season"].unique()),
        "total_players": len(df),
        "mvp_labels": int(mvp_count),
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print(f"  Saved: {summary_path}")

    print("\n" + "=" * 60)
    print("DONE! Check the output/ folder for charts and results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
