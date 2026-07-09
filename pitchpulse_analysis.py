"""
PulseIQ Cricket Analytics - Core Analysis Pipeline
Dataset: IPL matches 2008-2026 (1243 matches, ritesh-ojha/IPL-DATASET, sourced from Cricsheet)

Computes:
1. Player Workload & Fatigue Risk Index (rest-day based)
2. Clutch Performance Score (POTM awards weighted by match stakes)
3. Form Trajectory (rolling POTM rate per player over career)
4. Toss Impact Analysis (bat/field first win rates by venue)
5. Team Head-to-Head matrix
6. ML Match-Winner Predictor (Logistic Regression + Random Forest)
"""

import pandas as pd
import numpy as np
import json
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('data/Match_Info.csv')
df['match_date'] = pd.to_datetime(df['match_date'])
df = df.sort_values('match_date').reset_index(drop=True)
df['winner'] = df['winner'].fillna('No Result')

# canonical team name normalization (franchises renamed over the years)
TEAM_ALIAS = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Deccan Chargers': 'Sunrisers Hyderabad',
}
def norm_team(t):
    return TEAM_ALIAS.get(t, t)

df['team1'] = df['team1'].apply(norm_team)
df['team2'] = df['team2'].apply(norm_team)
df['winner_norm'] = df['winner'].apply(norm_team)
df['toss_winner'] = df['toss_winner'].apply(norm_team)

print(f"Loaded {len(df)} matches from {df['match_date'].min().date()} to {df['match_date'].max().date()}")

# --------------------------------------------------------------------------
# 1. Build long-format player-match table
# --------------------------------------------------------------------------
rows = []
for _, m in df.iterrows():
    for team_col, players_col, opp_col in [('team1', 'team1_players', 'team2'), ('team2', 'team2_players', 'team1')]:
        team = m[team_col]
        opp = m[opp_col]
        players = [p.strip() for p in str(m[players_col]).split(',') if p.strip()]
        is_winner_team = (m['winner_norm'] == team)
        for p in players:
            rows.append({
                'player': p,
                'team': team,
                'opponent': opp,
                'match_id': m['match_number'],
                'date': m['match_date'],
                'is_win': is_winner_team,
                'is_potm': (m['player_of_match'] == p),
                'is_eliminator': (str(m['eliminator']) not in ('NA', 'nan')),
                'venue': m['venue'],
            })
pm = pd.DataFrame(rows)
print(f"Player-match rows: {len(pm)}, unique players: {pm['player'].nunique()}")

# --------------------------------------------------------------------------
# 2. Workload & Fatigue Risk Index
#    Rationale: back-to-back matches (<=1 rest day) compound physical strain.
#    Risk score = weighted count of short-rest matches in trailing 21 days,
#    normalized against matches played (so it reflects *density*, not just volume).
# --------------------------------------------------------------------------
pm = pm.sort_values(['player', 'date'])
pm['prev_date'] = pm.groupby('player')['date'].shift(1)
pm['rest_days'] = (pm['date'] - pm['prev_date']).dt.days

def rest_weight(days):
    if pd.isna(days):
        return 0
    if days <= 1:
        return 3      # back-to-back or 1-day rest -> highest strain
    if days == 2:
        return 1.5
    if days <= 3:
        return 0.5
    return 0

pm['strain_pts'] = pm['rest_days'].apply(rest_weight)

workload = pm.groupby('player').agg(
    matches_played=('match_id', 'count'),
    total_strain=('strain_pts', 'sum'),
    last_match=('date', 'max'),
    first_match=('date', 'min'),
).reset_index()
workload['career_span_days'] = (workload['last_match'] - workload['first_match']).dt.days.clip(lower=1)
workload['strain_per_match'] = (workload['total_strain'] / workload['matches_played']).round(3)
# Only meaningfully rank players with a real sample size
workload_ranked = workload[workload['matches_played'] >= 15].copy()
workload_ranked['risk_score'] = (
    (workload_ranked['strain_per_match'] - workload_ranked['strain_per_match'].min()) /
    (workload_ranked['strain_per_match'].max() - workload_ranked['strain_per_match'].min()) * 100
).round(1)
workload_ranked = workload_ranked.sort_values('risk_score', ascending=False)

# --------------------------------------------------------------------------
# 3. Clutch Performance Score
#    POTM in eliminators/finals weighted higher than league-stage POTM.
# --------------------------------------------------------------------------
potm = pm[pm['is_potm']].copy()
potm['weight'] = np.where(potm['is_eliminator'], 3, 1)
clutch = potm.groupby('player').agg(
    potm_total=('match_id', 'count'),
    potm_eliminator=('is_eliminator', 'sum'),
    clutch_weighted=('weight', 'sum'),
).reset_index().sort_values('clutch_weighted', ascending=False)

# merge with total matches for a rate-based view
clutch = clutch.merge(workload[['player', 'matches_played']], on='player', how='left')
clutch['potm_rate_pct'] = (clutch['potm_total'] / clutch['matches_played'] * 100).round(2)
clutch_ranked = clutch[clutch['matches_played'] >= 15].sort_values('clutch_weighted', ascending=False)

# --------------------------------------------------------------------------
# 4. Form Trajectory - rolling POTM rate (last 20 matches) for top players
# --------------------------------------------------------------------------
top_players = clutch_ranked.head(12)['player'].tolist()
form_traj = {}
for p in top_players:
    ppm = pm[pm['player'] == p].sort_values('date').copy()
    ppm['potm_int'] = ppm['is_potm'].astype(int)
    ppm['rolling_form'] = ppm['potm_int'].rolling(window=15, min_periods=5).mean() * 100
    traj = ppm[['date', 'rolling_form']].dropna()
    form_traj[p] = [
        {'date': d.strftime('%Y-%m-%d'), 'form': round(f, 1)}
        for d, f in zip(traj['date'], traj['rolling_form'])
    ][::4]  # thin out points for compact payload

# --------------------------------------------------------------------------
# 5. Toss Impact Analysis
# --------------------------------------------------------------------------
df['toss_winner_won_match'] = (df['toss_winner'] == df['winner_norm'])
toss_overall = df.groupby('toss_decision')['toss_winner_won_match'].agg(['mean', 'count']).reset_index()
toss_overall['mean'] = (toss_overall['mean'] * 100).round(1)

venue_toss = df.groupby(['venue', 'toss_decision']).agg(
    win_rate=('toss_winner_won_match', 'mean'),
    n=('toss_winner_won_match', 'count')
).reset_index()
venue_toss = venue_toss[venue_toss['n'] >= 8]
venue_toss['win_rate'] = (venue_toss['win_rate'] * 100).round(1)
venue_toss = venue_toss.sort_values('n', ascending=False)

# --------------------------------------------------------------------------
# 6. Team head-to-head matrix (top teams)
# --------------------------------------------------------------------------
team_counts = pd.concat([df['team1'], df['team2']]).value_counts()
top_teams = team_counts.head(10).index.tolist()

h2h = {}
for t1 in top_teams:
    h2h[t1] = {}
    for t2 in top_teams:
        if t1 == t2:
            continue
        sub = df[((df['team1'] == t1) & (df['team2'] == t2)) | ((df['team1'] == t2) & (df['team2'] == t1))]
        if len(sub) == 0:
            continue
        wins = (sub['winner_norm'] == t1).sum()
        h2h[t1][t2] = {'played': int(len(sub)), 'wins': int(wins)}

# team overall win rate (for ML feature + dashboard)
team_stats = {}
for t in top_teams:
    played = df[(df['team1'] == t) | (df['team2'] == t)]
    wins = (played['winner_norm'] == t).sum()
    team_stats[t] = {'played': int(len(played)), 'wins': int(wins), 'win_rate': round(wins / len(played) * 100, 1)}

# --------------------------------------------------------------------------
# 7. ML Match-Winner Predictor
#    Features: toss winner==team1, toss_decision, team1/team2 rolling win-rate
#    entering the match (no lookahead leakage), h2h win rate entering match, venue.
# --------------------------------------------------------------------------
ml_df = df[df['winner_norm'].isin(df['team1'].unique().tolist() + df['team2'].unique().tolist())].copy()
ml_df = ml_df[ml_df['winner'] != 'No Result'].reset_index(drop=True)

# rolling team win rate computed strictly on matches BEFORE current match (no leakage)
team_history = {}  # team -> list of (date, win 0/1)

def get_rolling_winrate(team, current_date):
    hist = team_history.get(team, [])
    past = [w for d, w in hist if d < current_date]
    if len(past) < 5:
        return 0.5
    recent = past[-15:]
    return sum(recent) / len(recent)

venue_history = {}  # venue -> list of (date, bat_first_won 0/1)

features = []
for _, m in ml_df.iterrows():
    t1, t2, d, venue = m['team1'], m['team2'], m['match_date'], m['venue']
    t1_wr = get_rolling_winrate(t1, d)
    t2_wr = get_rolling_winrate(t2, d)
    h2h_all = team_history.get(f"{t1}|vs|{t2}", [])
    h2h_played = len([dd for dd, w in h2h_all if dd < d])
    h2h_wins = len([1 for dd, w in h2h_all if dd < d and w == 1])
    h2h_rate = (h2h_wins / h2h_played) if h2h_played >= 3 else 0.5

    v_hist = [w for dd, w in venue_history.get(venue, []) if dd < d]
    venue_bat_first_rate = (sum(v_hist) / len(v_hist)) if len(v_hist) >= 5 else 0.5

    team1_bats_first = (
        (m['toss_winner'] == t1 and m['toss_decision'] == 'bat') or
        (m['toss_winner'] == t2 and m['toss_decision'] == 'field')
    )
    aligned_with_venue_trend = int(
        (team1_bats_first and venue_bat_first_rate > 0.5) or
        (not team1_bats_first and venue_bat_first_rate <= 0.5)
    )

    features.append({
        'team1_winrate': t1_wr,
        'team2_winrate': t2_wr,
        'winrate_diff': t1_wr - t2_wr,
        'toss_won_by_team1': int(m['toss_winner'] == t1),
        'toss_decision_bat': int(m['toss_decision'] == 'bat'),
        'h2h_team1_rate': h2h_rate,
        'venue_bat_first_rate': venue_bat_first_rate,
        'team1_bats_first': int(team1_bats_first),
        'aligned_with_venue_trend': aligned_with_venue_trend,
        'venue': venue,
        'target_team1_wins': int(m['winner_norm'] == t1),
    })

    team_history.setdefault(t1, []).append((d, int(m['winner_norm'] == t1)))
    team_history.setdefault(t2, []).append((d, int(m['winner_norm'] == t2)))
    team_history.setdefault(f"{t1}|vs|{t2}", []).append((d, int(m['winner_norm'] == t1)))
    team_history.setdefault(f"{t2}|vs|{t1}", []).append((d, int(m['winner_norm'] == t2)))
    bat_first_won = int((team1_bats_first and m['winner_norm'] == t1) or ((not team1_bats_first) and m['winner_norm'] == t2))
    venue_history.setdefault(venue, []).append((d, bat_first_won))

feat_df = pd.DataFrame(features)
le_venue = LabelEncoder()
feat_df['venue_enc'] = le_venue.fit_transform(feat_df['venue'])

X = feat_df[['team1_winrate', 'team2_winrate', 'winrate_diff', 'toss_won_by_team1',
             'toss_decision_bat', 'h2h_team1_rate', 'venue_bat_first_rate',
             'team1_bats_first', 'aligned_with_venue_trend', 'venue_enc']]
y = feat_df['target_team1_wins']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=2000, C=0.7)
log_reg.fit(X_train_sc, y_train)
lr_pred = log_reg.predict(X_test_sc)

rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

def metrics(y_true, y_pred, name):
    return {
        'model': name,
        'accuracy': round(accuracy_score(y_true, y_pred) * 100, 2),
        'precision': round(precision_score(y_true, y_pred) * 100, 2),
        'recall': round(recall_score(y_true, y_pred) * 100, 2),
        'f1': round(f1_score(y_true, y_pred) * 100, 2),
    }

ml_results = [metrics(y_test, lr_pred, 'Logistic Regression'), metrics(y_test, rf_pred, 'Random Forest')]
print(ml_results)

# baseline: always predict "toss winner wins" and "higher winrate team wins"
baseline_toss = (feat_df.iloc[X_test.index]['toss_won_by_team1'] == feat_df.iloc[X_test.index]['target_team1_wins']).mean()
baseline_wr = ((feat_df.iloc[X_test.index]['winrate_diff'] > 0).astype(int) == y_test).mean()
print("baseline toss-winner-wins acc:", round(baseline_toss*100,2))
print("baseline higher-winrate-wins acc:", round(baseline_wr*100,2))

feat_importance = sorted(
    zip(X.columns, rf.feature_importances_), key=lambda x: -x[1]
)
feat_importance = [{'feature': f, 'importance': round(float(i), 4)} for f, i in feat_importance]

# --------------------------------------------------------------------------
# Export everything for the dashboard
# --------------------------------------------------------------------------
out = {
    'meta': {
        'total_matches': int(len(df)),
        'date_range': [df['match_date'].min().strftime('%Y-%m-%d'), df['match_date'].max().strftime('%Y-%m-%d')],
        'unique_players': int(pm['player'].nunique()),
        'seasons': int(df['match_date'].dt.year.nunique()),
    },
    'workload_risk': workload_ranked.head(15)[['player', 'matches_played', 'strain_per_match', 'risk_score']].to_dict('records'),
    'clutch_scores': clutch_ranked.head(15)[['player', 'potm_total', 'potm_eliminator', 'clutch_weighted', 'potm_rate_pct', 'matches_played']].to_dict('records'),
    'form_trajectory': form_traj,
    'toss_overall': toss_overall.to_dict('records'),
    'venue_toss_top': venue_toss.head(15).to_dict('records'),
    'team_stats': team_stats,
    'h2h': h2h,
    'ml_results': ml_results,
    'ml_baselines': {
        'toss_winner_wins_acc': round(baseline_toss * 100, 2),
        'higher_winrate_wins_acc': round(baseline_wr * 100, 2),
    },
    'feature_importance': feat_importance,
    'top_teams': top_teams,
    'predictor': {
        'feature_order': list(X.columns),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'coef': log_reg.coef_[0].tolist(),
        'intercept': float(log_reg.intercept_[0]),
    },
    'venue_lookup': {
        v: round(sum(w for _, w in venue_history[v]) / len(venue_history[v]) * 100, 1)
        for v in venue_history if len(venue_history[v]) >= 6
    },
    'venue_enc_map': {v: int(le_venue.transform([v])[0]) for v in venue_history if len(venue_history[v]) >= 6},
}

with open('insights.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)

print("\nSaved insights.json")
print(f"LogReg accuracy: {ml_results[0]['accuracy']}%  |  RandomForest accuracy: {ml_results[1]['accuracy']}%")
print(f"Baseline (toss winner always wins): {out['ml_baselines']['toss_winner_wins_acc']}%")
