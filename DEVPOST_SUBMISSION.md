# Devpost Submission Text — PitchPulse

## Inspiration
Most cricket dashboards stop at "who scored the most runs." Real front offices need harder answers: who's quietly being overworked before an injury shows up, who actually performs when a match is on the line, and whether a match outcome can be honestly predicted at all — or whether that honesty itself is the insight. PitchPulse was built to answer those questions using 19 seasons of real IPL history.

## What it does
PitchPulse is a cricket intelligence dashboard built on 1,243 IPL matches (2008–2026, 800+ players):
- **Workload & Fatigue Risk Index** — flags players whose recent schedule has been densest in short-rest turnarounds, using a rest-gap-weighted strain score (0–100), not just raw match count.
- **Clutch Performance Index** — weights Player-of-the-Match awards 3× when earned in an eliminator/knockout, surfacing genuine big-match performers.
- **Form Trajectory** — 15-match rolling POTM rate tracing momentum across a career.
- **Toss & Venue Intelligence** — bat/field-first win rates sliced per venue rather than blindly averaged.
- **Team Rivalries** — all-time head-to-head records.
- **Win Predictor** — a Logistic Regression + Random Forest model trained on chronologically leak-free rolling features, running live in-browser off the model's actual learned coefficients, honestly benchmarked against naive baselines.

## How we built it
Python (pandas) for data cleaning and feature engineering, scikit-learn for the prediction models, and a dependency-light HTML/CSS/JS dashboard (Chart.js) so the entire prototype runs standalone in any browser with zero install. All computed insights are pre-baked into a single JSON payload consumed by the dashboard.

## Challenges we ran into
The biggest challenge was avoiding data leakage in the win-predictor: naive approaches compute "historical" team strength using the full dataset, silently leaking future results into past predictions. We rebuilt every rolling feature (team win-rate, head-to-head, venue tendency) by walking matches strictly in chronological order and updating state only after using it for that row. After fixing leakage and adding venue-tendency features, our model moved from performing *worse* than naive baselines to beating them by ~5.3 accuracy points — a real result we report honestly rather than an inflated one.

## Accomplishments that we're proud of
Building genuinely novel, actionable metrics (workload risk, clutch weighting) that don't exist in the raw data, and being disciplined enough to honestly report a modest ML edge instead of overclaiming — which we think is exactly the kind of statistical soundness this hackathon's judging criteria rewards.

## What we learned
T20 cricket outcomes are famously high-variance at the macro level — pre-match features alone (team strength, toss, venue) top out around a 5-point edge over coin-flip-style baselines. The real signal in this dataset lives at the player level, not the match level, which is why we built four player-facing modules and one honestly-caveated match predictor.

## What's next for PitchPulse
Incorporating ball-by-ball data for in-play win probability, extending the workload index with travel distance between venues, and generalizing the pipeline to other T20 leagues (BBL, PSL, The Hundred) using the same Cricsheet data format.
