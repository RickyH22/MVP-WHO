# NBA MVP Prediction — Analysis Results

## Question: How Much Does Defense Actually Matter in MVP Voting?

### Model Comparison

| Model | Features | Cross-Val Accuracy |
|---|---|---|
| Offense Only | PTS, AST, PER, WS | 94.44% |
| With Raw Defense | + STL, BLK, DRB, DWS, DBPM | 93.33% |
| With DEF_SCORE (composite) | + STL×1.5 + BLK×1.5 + DWS×2 + DBPM×2 | **96.67%** |

### Feature Importance (Permutation-Based)

| Feature | Type | Importance |
|---|---|---|
| DRB | Defense | 0.0111 |
| PER | Overall | 0.0000 |
| PTS | Offense | -0.0111 |
| WS | Overall | -0.0111 |
| BLK | Defense | -0.0111 |
| DWS | Defense | -0.0111 |
| DBPM | Defense | -0.0111 |
| AST | Offense | -0.0222 |
| STL | Defense | -0.0333 |

### Top 2024 MVP Candidates (Model Prediction)

| Player | Team | PTS | AST | DWS | DBPM | MVP Prob |
|---|---|---|---|---|---|---|
| Nikola Jokic | DEN | 26.4 | 9.0 | 4.3 | 3.5 | 0.60 |
| Shai Gilgeous-Alexander | OKC | 30.1 | 6.2 | 3.8 | 2.6 | 0.20 |
| Luka Doncic | DAL | 33.9 | 9.8 | 1.6 | -0.8 | 0.00 |
| Giannis Antetokounmpo | MIL | 30.4 | 6.5 | 3.1 | 1.5 | 0.00 |
| Jayson Tatum | BOS | 26.9 | 4.9 | 3.6 | 1.0 | 0.00 |

### Key Takeaways

1. **Offense alone predicts MVPs at 94.4%** — MVP voting is overwhelmingly driven by scoring and overall production (PTS, PER, WS).

2. **Adding raw defensive stats slightly hurts accuracy (93.3%)** — the 5 extra features add noise because most MVP candidates aren't elite defenders. The model gets confused by the extra dimensions with only 90 data points.

3. **A single composite DEF_SCORE performs best (96.7%)** — collapsing defense into one engineered feature (weighted STL + BLK + DWS + DBPM) captures the useful defensive signal without the noise.

4. **DRB (defensive rebounds) was the only defensive feature with positive importance** — big men who grab boards (Jokic, Giannis, Embiid) have dominated recent MVP races.

5. **The model correctly ranked Nikola Jokic as the #1 2024 MVP candidate** (60% probability), with Shai Gilgeous-Alexander second (20%).

### Bottom Line

Defense is a tiebreaker, not a driver. MVPs are picked for their offense, but a well-rounded defensive profile (captured by DEF_SCORE) gives the model a small edge. The era of "best scorer on the best team wins MVP" holds up.

### Technical Details

- **Model**: K-Nearest Neighbors (k=5) with StandardScaler normalization
- **Evaluation**: 5-fold cross-validation
- **Feature Importance**: Permutation-based (accuracy drop when feature shuffled)
- **Dataset**: 90 player-seasons (2016–2024), top 10 MVP candidates per season
- **Charts**: feature_importance.png, model_comparison.png, mvp_profile.png (in output/)
