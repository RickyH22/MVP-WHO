# NBA MVP Prediction — How Much Does Defense Matter?

A machine learning project that analyzes how much defensive metrics contribute to NBA MVP voting using a KNN classifier.

## Results

| Model | Accuracy |
|---|---|
| Offense Only (PTS, AST, PER, WS) | 94.44% |
| With Raw Defense (+STL, BLK, DRB, DWS, DBPM) | 93.33% |
| With DEF_SCORE (composite) | **96.67%** |

**Finding:** Defense is a tiebreaker, not a driver. A composite defensive metric (DEF_SCORE) gives the best accuracy, but offense alone already predicts MVPs at 94.4%.

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
python mvp_prediction.py
```

Charts and summary are saved to `output/`.

## Project Structure

```
mvp-prediction/
    mvp_prediction.py          # Main script
    mvp_analysis_results.md    # Detailed results & takeaways
    data/
        nba_mvp_stats.csv      # Dataset (90 player-seasons, 2016-2024)
    output/                    # Generated charts & summary (gitignored)
```

## Technical Details

- **Model**: K-Nearest Neighbors (k=5) with StandardScaler
- **Evaluation**: 5-fold cross-validation
- **Feature Importance**: Permutation-based
- **Dataset**: Top 10 MVP candidates per season, 2016-2024
