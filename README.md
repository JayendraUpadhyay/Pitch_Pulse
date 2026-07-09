# 🩺 PitchPulse — A Cricket Intelligence Engine

**Built for: AQX Sports Analytics Data Bowl 2.0**

PitchPulse reads the vitals of the game — player workload strain, clutch-moment impact, momentum, and win probability — from 19 seasons of real IPL history (2008–2026, 1,243 matches, 800+ players). It goes beyond surface-level box scores to answer questions a coach, fitness staff, or front office would actually ask.

**Live dashboard:** open `pitchpulse_dashboard.html` in any browser — fully self-contained, no server or install required.

---

## The Problem

Most fan-facing cricket stats stop at "who scored the most runs." Real analytics departments need answers to harder questions:
- *Which players are we quietly overworking before an injury shows up?*
- *Who actually performs when the match is on the line, not just on average?*
- *Does our toss strategy at this specific venue match the data, or just habit?*
- *Can we honestly quantify how predictable a match is — and where the real signal lives?*

## What PitchPulse Builds

| Module | What it does | Who benefits |
|---|---|---|
| **Workload & Fatigue Risk Index** | Weights each player's rest-gap history (back-to-back matches weighted 3× a normal turnaround) into a 0–100 risk score | Fitness/medical staff — rotation & injury-prevention decisions |
| **Clutch Performance Index** | Triples the weight of Player-of-the-Match awards earned in eliminators/knockouts vs league matches | Team management — identifying genuine big-match players |
| **Form Trajectory** | 15-match rolling POTM rate, traced over a career | Selection committees — momentum, not just career totals |
| **Toss & Venue Intelligence** | Bat/field-first win rates sliced per venue, not blindly aggregated | Captains — venue-specific toss strategy |
| **Team Rivalries** | All-time head-to-head records between franchises | Fans, analysts, broadcast narrative |
| **Win Predictor** | Logistic Regression + Random Forest trained on **chronologically leak-free** rolling features, live in-browser | Anyone wanting a probability, honestly benchmarked against naive baselines |

## Why This Is Analytically Sound (not just a dashboard)

- **No data leakage.** Every "historical" feature (team win-rate, head-to-head, venue tendency) is computed by walking matches in date order and updating state *after* using it — a common mistake in amateur sports-ML projects that we explicitly guard against. See `notebook/PitchPulse_Methodology.ipynb`.
- **Honest benchmarking.** Our win predictor is compared against two naive baselines ("toss winner always wins", "higher win-rate team always wins"). It beats them by ~5.3 points — a real, modest, honestly-reported edge, not an inflated accuracy number.
- **Novel derived metrics.** Workload risk and clutch-weighting aren't in the raw data — they're original feature engineering built specifically to answer practical questions.

## Data

Source: [Cricsheet](https://cricsheet.org/) via [ritesh-ojha/IPL-DATASET](https://github.com/ritesh-ojha/IPL-DATASET) (`Match_Info.csv`) — 1,243 IPL matches, 2008–2026, including full playing XIs, toss data, venues, and results.

## Repo Structure

```
├── pitchpulse_dashboard.html      # Standalone interactive dashboard (open directly)
├── analysis.py                    # Full analytics + ML pipeline (run: python3 analysis.py)
├── insights.json                  # Pre-computed output consumed by the dashboard
├── data/Match_Info.csv            # Raw dataset
├── notebook/
│   ├── PitchPulse_Methodology.ipynb   # Executable methodology walkthrough
│   └── pitchpulse_analysis.py
└── README.md
```

## Run It Yourself

```bash
pip install pandas numpy scikit-learn
python3 analysis.py          # regenerates insights.json from data/Match_Info.csv
open pitchpulse_dashboard.html
```

## Stack

Python · pandas · scikit-learn (Logistic Regression, Random Forest) · Chart.js · vanilla HTML/CSS/JS (no build step, no dependencies to install for the dashboard itself)

## Team

Jayendra Upadhyay ([@JayendraUpadhyay](https://github.com/JayendraUpadhyay)) — Data Analyst & ML Engineer

---

*Submitted to AQX Sports Analytics Data Bowl 2.0. Projects unrelated to sports analytics are not eligible; this one lives entirely in cricket analytics from data ingestion to deployed dashboard.*
