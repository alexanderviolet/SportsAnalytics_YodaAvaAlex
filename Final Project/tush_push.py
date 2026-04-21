"""
GO BIRDS GO!
Eagles Tush Push Opponent Recommender
--------------------------------------
Given an opponent, answers: should the Eagles use the Tush Push against
this team, or is there a specific alternative play that converts better?

DECISION LOG — all thresholds in this file
──────────────────────────────────────────
  SHORT_YARDAGE_MAX_DISTANCE = 3
    Source: project spec ("3rd and 1-3 and 4th and 1-3")

  TUSH_PUSH_MAX_DISTANCE = 1
    Project spec says "2 yards or less" but empirical validation against
    TushPush.fyi (Jan 2026) showed all distance=2 plays in our dataset were
    QB draws or keepers gaining 6-12 yards — not sneaks. Distance=1 matches
    TushPush.fyi attempt counts and aligns with how the play is actually run.

  GOAL_LINE_YARDS = 5
    Source: project spec ("1st or 2nd down within 5 yards of the goal line")

  PLAY_TYPE_PRIORITY_ORDER: most specific pattern wins; see classify_play_type()
    Scramble > Play Action > Outside Run > Power Run > Pass Short > Pass Deep
    Rationale: scrambles and play-action are unambiguous; run direction is more
    specific than generic run; pass depth is the least ambiguous pass split.

  TUSH PUSH DIRECTION KEYWORDS: up the middle, right/left guard,
    right/left tackle, right/left end
    Source: exhaustive inspection of all QB runs on 3rd/4th & 1-2 league-wide
    in the dataset. These are the only direction keywords that appear in
    QB short-yardage run descriptions. Scrambles are excluded separately
    because they represent unplanned QB runs, not designed sneaks.

  OVERALL_TP_RATE = 0.852
    Eagles Tush Push conversion rate across full 2021-2025 dataset (121/142).
    Used as the safe fallback when per-opponent sample is too small to trust.

  TRUST BLENDING SCHEDULE (trust_adjusted_tp_rate):
    0-4 attempts → use overall rate entirely (sample too small to learn from)
    5+ attempts  → blend weight is derived directly from the Wilson lower bound:
                   weight = wilson_lower_bound(raw_rate, n) / raw_rate
                   This is the fraction of the raw rate the CI says is
                   statistically defensible. No hardcoded percentages —
                   the weight is computed fresh for each opponent's actual data.
    Rationale: the blend weight IS the Wilson CI, not an approximation of it.

  MIN_ALT_ATTEMPTS = 3
    Minimum attempts for an alternative play to qualify as a recommendation
    vs this specific opponent. Lower than the global 5 because per-opponent
    samples are inherently smaller than full-dataset samples.
"""

import math
import os
import re
import sys
import glob
import argparse
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

EAGLES_NAME = "Philadelphia Eagles"

# Short yardage thresholds
SHORT_YARDAGE_MAX_DISTANCE = 3   # 3rd/4th & 1-3  (project spec)
TUSH_PUSH_MAX_DISTANCE     = 1   # ≤ 1 yard to go  (empirically validated)
GOAL_LINE_YARDS            = 5   # 1st/2nd within 5 yds of goal line  (project spec)

# Minimum attempts for a play type to appear in the non-TP breakdown table.
# Below this the raw rate is too noisy even to display.
MIN_DISPLAY_ATTEMPTS = 2

# Eagles overall TP rate — fallback when per-opponent sample is too small
OVERALL_TP_RATE = 0.852   # 85.2% — 121/142 attempts, 2021-2025

# Wilson CI confidence level (95%)
WILSON_Z = 1.96

# Play type classification — ordered most specific to least specific.
# First matching pattern wins; order matters.
PLAY_TYPE_PRIORITY = [
    ("Scramble",     [r"\bscrambles\b"]),
    ("Play Action",  [r"\b(play.?action)\b"]),
    ("Outside Run",  [r"\b(right end|left end|right sweep|left sweep|sweep|toss|pitch)\b"]),
    ("Power Run",    [r"\b(right guard|left guard|right tackle|left tackle|up the middle)\b"]),
    ("Pass Short",   [r"\bpass short\b"]),
    ("Pass Deep",    [r"\bpass deep\b"]),
    ("Pass (Other)", [r"\bpass\b"]),
    ("Run (Other)",  [r"\d+ yard run|\bright\b|\bleft\b"]),
]

# Direction keywords that indicate a QB sneak / Tush Push run.
# Source: exhaustive inspection of all QB short-yardage runs in the dataset.
SNEAK_DIRECTIONS = re.compile(
    r"\b(up the middle|right guard|left guard|right tackle|left tackle|right end|left end)\b",
    re.IGNORECASE,
)

# Team name resolution maps
TEAM_ABBR_TO_FULL = {
    "ARI": "Arizona Cardinals",    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",     "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",   "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",       "DEN": "Denver Broncos",
    "DET": "Detroit Lions",        "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",       "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC":  "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LV":  "Las Vegas Raiders",    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",    "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",   "NYG": "New York Giants",
    "NYJ": "New York Jets",        "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",  "SEA": "Seattle Seahawks",
    "SF":  "San Francisco 49ers",  "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",     "WAS": "Washington Commanders",
}

TEAM_NICKNAMES = {
    "cardinals": "Arizona Cardinals",    "falcons":    "Atlanta Falcons",
    "ravens":    "Baltimore Ravens",     "bills":      "Buffalo Bills",
    "panthers":  "Carolina Panthers",   "bears":      "Chicago Bears",
    "bengals":   "Cincinnati Bengals",  "browns":     "Cleveland Browns",
    "cowboys":   "Dallas Cowboys",      "dallas":     "Dallas Cowboys",
    "broncos":   "Denver Broncos",      "lions":      "Detroit Lions",
    "packers":   "Green Bay Packers",   "texans":     "Houston Texans",
    "colts":     "Indianapolis Colts",  "jaguars":    "Jacksonville Jaguars",
    "chiefs":    "Kansas City Chiefs",  "chargers":   "Los Angeles Chargers",
    "rams":      "Los Angeles Rams",    "raiders":    "Las Vegas Raiders",
    "dolphins":  "Miami Dolphins",      "vikings":    "Minnesota Vikings",
    "patriots":  "New England Patriots","saints":     "New Orleans Saints",
    "giants":    "New York Giants",     "jets":       "New York Jets",
    "eagles":    "Philadelphia Eagles", "steelers":   "Pittsburgh Steelers",
    "seahawks":  "Seattle Seahawks",    "49ers":      "San Francisco 49ers",
    "buccaneers":"Tampa Bay Buccaneers","bucs":       "Tampa Bay Buccaneers",
    "titans":    "Tennessee Titans",    "commanders": "Washington Commanders",
    "washington":"Washington Commanders",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_plays(data_dir: str) -> pd.DataFrame:
    """
    Load all plays CSVs, searching recursively through subfolders.
    Handles both flat (data/2021_WEEK_1_plays.csv) and
    year-subfolder (data/2021/WEEK_1_plays.csv) layouts.
    """
    files_deep = glob.glob(os.path.join(data_dir, "**", "*plays*.csv"), recursive=True)
    files_flat = glob.glob(os.path.join(data_dir, "*plays*.csv"))
    files = sorted(set(files_deep + files_flat))
    if not files:
        sys.exit(
            f"[ERROR] No plays CSVs found under '{data_dir}'.\n"
            "Expected files matching *plays*.csv in that directory or subfolders."
        )
    chunks = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "Season" in df.columns:
                df = df[df["Season"].between(2021, 2025)]
            chunks.append(df)
        except Exception as e:
            print(f"  [WARN] Could not load {f}: {e}")
    if not chunks:
        sys.exit("[ERROR] All plays CSVs failed to load.")
    plays = pd.concat(chunks, ignore_index=True)
    print(f"[INFO] Loaded {len(plays):,} plays from {len(files)} files.")
    seasons = sorted(plays["Season"].dropna().unique().astype(int)) if "Season" in plays.columns else []
    for s in seasons:
        weeks = sorted(plays[plays["Season"] == s]["Week"].dropna().unique())
        print(f"  {s}: {len(weeks)} weeks")
    return plays


# ─────────────────────────────────────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_play_start(plays: pd.DataFrame) -> pd.DataFrame:
    """Parse PlayStart like '3rd & 1 at PHI 5' into down, distance, field_team, field_pos."""
    pattern = r"(?P<down>\d+)(?:st|nd|rd|th)\s*&\s*(?P<dist>\d+)\s+at\s+(?P<fteam>\w+)\s+(?P<fpos>\d+)"
    extracted = plays["PlayStart"].str.extract(pattern)
    plays = plays.copy()
    plays["down"]       = pd.to_numeric(extracted["down"],  errors="coerce")
    plays["distance"]   = pd.to_numeric(extracted["dist"],  errors="coerce")
    plays["field_team"] = extracted["fteam"]
    plays["field_pos"]  = pd.to_numeric(extracted["fpos"],  errors="coerce")
    return plays


def parse_yards_gained(plays: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric yards gained from PlayOutcome, e.g. '3 Yard Run' → 3."""
    plays = plays.copy()
    gained = plays["PlayOutcome"].str.extract(r"(-?\d+)\s+Yard", expand=False)
    plays["yards_gained"] = pd.to_numeric(gained, errors="coerce")
    plays.loc[plays["PlayOutcome"].str.contains("Touchdown", case=False, na=False), "yards_gained"] = 99
    return plays


def classify_play_type(description: str) -> str:
    """Classify a play description using the priority-ordered pattern list."""
    if pd.isna(description):
        return "Other"
    for label, patterns in PLAY_TYPE_PRIORITY:
        for p in patterns:
            if re.search(p, description, re.IGNORECASE):
                return label
    return "Other"


def extract_formation(play_time_formation: str) -> str:
    """Extract formation from PlayTimeFormation like '09:55 1st Shotgun'."""
    if pd.isna(play_time_formation):
        return "Under Center"
    match = re.search(r"\d+(?:st|nd|rd|th)\s+(.*)", str(play_time_formation))
    if match:
        form = match.group(1).strip()
        return form if form else "Under Center"
    return "Under Center"


def is_conversion(row) -> bool:
    """A play converts if it scores a touchdown or gains at least the yards needed."""
    if pd.isna(row.get("yards_gained")) or pd.isna(row.get("distance")):
        return False
    if "Touchdown" in str(row.get("PlayOutcome", "")):
        return True
    return row["yards_gained"] >= row["distance"]


# ─────────────────────────────────────────────────────────────────────────────
# TUSH PUSH DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def build_qb_set(plays: pd.DataFrame) -> set:
    """
    Build the set of all QB names by finding every player who appears
    as a passer in any PlayDescription. Data-driven — no hardcoded names.
    Validated: correctly identifies Hurts, all backup QBs; excludes all RBs.
    """
    qbs = set()
    for desc in plays["PlayDescription"].dropna():
        m = re.match(r"— ([A-Z]\.[A-Za-z\-']+) pass", desc)
        if m:
            qbs.add(m.group(1))
    return qbs


def _team_abbr(full_name: str) -> str:
    """Map full team name to abbreviation for goal-line field position check."""
    return {v: k for k, v in TEAM_ABBR_TO_FULL.items()}.get(full_name, full_name)


def is_tush_push(row, qb_set: set) -> bool:
    """
    Classify a play as a Tush Push when ALL of the following are true:
      1. Run direction is interior (up the middle, guard, tackle, or end gap)
      2. Not a scramble (unplanned QB run)
      3. Carrier is a QB (cross-referenced against qb_set)
      4. Distance ≤ TUSH_PUSH_MAX_DISTANCE yards to go
      5. Situation: 3rd/4th down, OR 1st/2nd within GOAL_LINE_YARDS of end zone
    """
    desc      = str(row.get("PlayDescription", ""))
    dist      = row.get("distance", np.nan)
    down      = row.get("down", np.nan)
    field_pos = row.get("field_pos", np.nan)
    field_team= str(row.get("field_team", ""))
    poss_team = str(row.get("TeamWithPossession", ""))

    if not SNEAK_DIRECTIONS.search(desc):
        return False
    if "scrambles" in desc.lower():
        return False

    m = re.match(r"— ([A-Z]\.[A-Za-z\-']+)\s", desc)
    carrier = m.group(1) if m else None
    if not carrier or carrier not in qb_set:
        return False

    if pd.isna(dist) or dist > TUSH_PUSH_MAX_DISTANCE:
        return False
    if pd.isna(down):
        return False

    if down in (3, 4):
        return True

    poss_abbr = _team_abbr(poss_team)
    if down in (1, 2) and field_team and field_team != poss_abbr and not pd.isna(field_pos):
        if field_pos <= GOAL_LINE_YARDS:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# TRUST-ADJUSTED TP RATE
# ─────────────────────────────────────────────────────────────────────────────

def wilson_lower_bound(successes: int, attempts: int) -> float:
    """
    Returns the pessimistic end of a 95% Wilson confidence interval.
    Plain English: the worst conversion rate we can reasonably justify
    given the data. Small samples produce wide intervals (low lower bound).
    Large samples produce tight intervals (lower bound close to raw rate).
    """
    if attempts == 0:
        return 0.0
    p = successes / attempts
    n = attempts
    center = (p + WILSON_Z**2 / (2*n)) / (1 + WILSON_Z**2 / n)
    spread  = (WILSON_Z * math.sqrt(p*(1-p)/n + WILSON_Z**2/(4*n**2))) / (1 + WILSON_Z**2/n)
    return max(0.0, center - spread)


def trust_adjusted_tp_rate(succ: int, att: int) -> tuple[float, str]:
    """
    Returns the adjusted TP rate used for the decision, plus a plain-English
    explanation of how it was calculated.

    The blend weight is derived directly from the Wilson confidence interval:
      weight = wilson_lower_bound(raw_rate, n) / raw_rate

    This is the fraction of the raw rate the CI says is statistically
    defensible. The remaining (1 - weight) goes to the overall historical
    rate of 85.2%. No hardcoded percentages — the weight is computed fresh
    from each opponent's actual data.

    Below 5 attempts the sample is too small to learn anything from, so
    the overall rate is used entirely.
    """
    if att == 0:
        rate = OVERALL_TP_RATE * 100
        return rate, f"No attempts vs this opponent — using Eagles overall rate ({rate:.1f}%)."

    p = succ / att

    if att < 5:
        rate = OVERALL_TP_RATE * 100
        return rate, (
            f"Only {att} attempt(s) — not enough data to learn from. "
            f"Using Eagles overall rate ({rate:.1f}%)."
        )

    lb = wilson_lower_bound(succ, att)  # pass raw counts, not proportion

    # Weight = fraction of the raw rate that the Wilson CI says is defensible.
    # At low sample sizes this is small (CI is wide, lower bound is far below raw).
    # At large sample sizes this approaches 1.0 (CI is tight, lb ≈ raw rate).
    w = lb / p if p > 0 else 0.0

    rate = w * p * 100 + (1 - w) * OVERALL_TP_RATE * 100

    expl = (
        f"{att} attempts, {succ} successful ({p*100:.1f}% raw). "
        f"Wilson lower bound: {lb*100:.1f}% — that is {w*100:.0f}% of the raw rate. "
        f"Blending {w*100:.0f}% opponent ({p*100:.1f}%) with "
        f"{(1-w)*100:.0f}% overall (85.2%) → {rate:.1f}%."
    )

    return rate, expl


def trust_adjusted_play_rate(
    opp_succ: int,
    opp_att: int,
    overall_succ: int,
    overall_att: int,
    play_type: str = "this play",
) -> tuple[float, str]:
    """
    Calculates the trust-adjusted conversion rate for an alternative play,
    using the exact same method as trust_adjusted_tp_rate:

      1. Blend opponent-specific rate with overall Eagles rate, weighted
         by the Wilson lower bound fraction at the given sample size.
      2. Apply Wilson lower bound to the blended rate for conservatism.

    overall_succ / overall_att is the Eagles' full-dataset rate for this
    play type — the equivalent of OVERALL_TP_RATE for the Tush Push.
    """
    if overall_att == 0:
        return 0.0, "No overall Eagles data for this play type."

    overall_rate = overall_succ / overall_att

    # Step 1: blend opponent-specific rate with overall rate
    if opp_att == 0:
        blended = overall_rate
        blend_expl = (
            f"Eagles have never run {play_type} vs this opponent — "
            f"using overall Eagles {play_type} rate ({overall_rate*100:.1f}%, "
            f"{overall_succ}/{overall_att} attempts across all seasons)."
        )
    elif opp_att < 5:
        blended = overall_rate
        blend_expl = (
            f"Eagles have only run {play_type} {opp_att} time(s) vs this opponent — "
            f"not enough to trust. Using overall Eagles {play_type} rate "
            f"({overall_rate*100:.1f}%, {overall_succ}/{overall_att} attempts across all seasons)."
        )
    else:
        opp_rate = opp_succ / opp_att
        lb_opp   = wilson_lower_bound(opp_succ, opp_att)
        w        = lb_opp / opp_rate if opp_rate > 0 else 0.0
        blended  = w * opp_rate + (1 - w) * overall_rate
        blend_expl = (
            f"Eagles ran {play_type} {opp_att} times vs this opponent "
            f"({opp_succ}/{opp_att} = {opp_rate*100:.1f}% raw). "
            f"Wilson weight: {w*100:.0f}% opponent-specific, {(1-w)*100:.0f}% overall "
            f"({overall_rate*100:.1f}%) → blended rate {blended*100:.1f}%."
        )

    # Step 2: apply Wilson lower bound to the blended rate
    blended_succ = round(blended * overall_att)
    lb_final = wilson_lower_bound(blended_succ, overall_att)
    rate     = lb_final * 100

    expl = (
        f"{blend_expl} "
        f"Conservative adjusted rate: {rate:.1f}%."
    )

    return rate, expl


# ─────────────────────────────────────────────────────────────────────────────
# TEAM NAME RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def resolve_team_name(raw: str) -> str | None:
    """
    Resolve user input (abbreviation, nickname, or full name) to the exact
    full name used in the TeamWithPossession column. Returns None if unknown.
    """
    raw = raw.strip()
    if raw in TEAM_ABBR_TO_FULL.values():
        return raw
    if raw.upper() in TEAM_ABBR_TO_FULL:
        return TEAM_ABBR_TO_FULL[raw.upper()]
    if raw.lower() in TEAM_NICKNAMES:
        return TEAM_NICKNAMES[raw.lower()]
    matches = [full for full in TEAM_ABBR_TO_FULL.values() if raw.lower() in full.lower()]
    if len(matches) == 1:
        return matches[0]
    return None


def tag_opponent(eagles_plays: pd.DataFrame, all_plays: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'opponent' column to every Eagles play by finding the non-Eagles
    team with possession in the same game (matched by Season + Week + AwayTeam
    + HomeTeam). Uses all_plays separately to avoid the multiple-teams-per-game
    issue that drop_duplicates would cause on Eagles rows alone.
    """
    game_keys = ["Season", "Week", "AwayTeam", "HomeTeam"]
    game_opponents = (
        all_plays[all_plays["TeamWithPossession"] != EAGLES_NAME]
        [game_keys + ["TeamWithPossession"]]
        .drop_duplicates(subset=game_keys)
        .rename(columns={"TeamWithPossession": "opponent"})
    )
    return eagles_plays.merge(game_opponents, on=game_keys, how="left")


# ─────────────────────────────────────────────────────────────────────────────
# OPPONENT RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────

def section_header(title: str):
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subsection(title: str):
    print(f"\n── {title} ──")


def recommend(plays: pd.DataFrame, opponent_name: str, qb_set: set):
    section_header(f"EAGLES vs. {opponent_name.upper()} — TUSH PUSH RECOMMENDER")

    # Tag all Eagles plays with opponent
    eagles_all = plays[plays["TeamWithPossession"] == EAGLES_NAME].copy()
    eagles_all = tag_opponent(eagles_all, plays)

    # Filter to this opponent
    vs_opp = eagles_all[eagles_all["opponent"] == opponent_name]
    if vs_opp.empty:
        print(f"\n  [No Eagles plays found vs {opponent_name}]")
        print(f"  Available opponents: {sorted(eagles_all['opponent'].dropna().unique())}")
        return

    # Short yardage only
    vs_short = vs_opp[
        vs_opp["down"].isin([3, 4]) &
        vs_opp["distance"].between(1, SHORT_YARDAGE_MAX_DISTANCE)
    ].copy()

    vs_tp    = vs_short[vs_short["is_tush_push"]]
    vs_no_tp = vs_short[~vs_short["is_tush_push"]]

    # ── Games played ─────────────────────────────────────────────────────────
    games = vs_opp[["Season", "Week"]].drop_duplicates().sort_values(["Season", "Week"])
    print(f"\n  Games in dataset vs {opponent_name}: {len(games)}")
    for _, g in games.iterrows():
        print(f"    {int(g['Season'])} {g['Week']}")
    if len(games) < 4:
        print(f"  [NOTE: Load full 2021-2025 data for more reliable per-opponent rates.]")

    # ── Tush Push performance ─────────────────────────────────────────────────
    subsection(f"Tush Push vs {opponent_name}")
    tp_att  = len(vs_tp)
    tp_succ = int(vs_tp["converted"].sum())
    tp_tds  = int(vs_tp["PlayOutcome"].str.contains("Touchdown", case=False, na=False).sum())
    tp_rate = 100 * tp_succ / tp_att if tp_att > 0 else None

    print(f"  Attempts  : {tp_att}")
    print(f"  Successful: {tp_succ}")
    print(f"  Touchdowns: {tp_tds}")
    print(f"  Conv%     : {f'{tp_rate:.1f}%' if tp_rate is not None else 'N/A'}")

    # Year-over-year trend
    seasons_vs = sorted(vs_tp["Season"].dropna().unique().astype(int))
    if len(seasons_vs) > 1:
        print(f"\n  Season-by-season trend vs {opponent_name}:")
        season_rates = []
        for s in seasons_vs:
            s_tp   = vs_tp[vs_tp["Season"] == s]
            s_succ = int(s_tp["converted"].sum())
            s_rate = 100 * s_succ / len(s_tp) if len(s_tp) > 0 else 0
            season_rates.append(s_rate)
            print(f"    {s}: {s_succ}/{len(s_tp)} = {s_rate:.1f}%")
        mid        = len(season_rates) // 2
        early_avg  = sum(season_rates[:mid]) / mid if mid > 0 else season_rates[0]
        late_avg   = sum(season_rates[mid:]) / len(season_rates[mid:])
        trend      = late_avg - early_avg
        if abs(trend) >= 10:
            direction = f"DECLINING ({trend:+.1f}pp)" if trend < 0 else f"IMPROVING ({trend:+.1f}pp)"
            print(f"    Trend: {direction}")

    tp_rate_adj, trust_expl = trust_adjusted_tp_rate(tp_succ, tp_att)
    print(f"\n  How much we trust this rate:")
    print(f"  {trust_expl}")

    # ── Non-TP play breakdown ─────────────────────────────────────────────────
    subsection(f"Non-Tush Push short yardage vs {opponent_name}")
    print(f"  Total non-TP plays: {len(vs_no_tp)}")

    if len(vs_no_tp) > 0:
        pt_vs_opp = (
            vs_no_tp.groupby("play_type")
            .agg(plays=("converted", "count"), conversions=("converted", "sum"))
            .assign(conv_rate=lambda x: x["conversions"] / x["plays"] * 100)
            .sort_values("conv_rate", ascending=False)
        )
        print(pt_vs_opp.to_string(float_format="%.1f"))

    # ── Find best alternative play ────────────────────────────────────────────
    # For each play type, compute a trust-adjusted rate using the same method
    # as the Tush Push: blend opponent-specific data with overall Eagles data
    # weighted by the Wilson CI, then apply the Wilson lower bound to the result.
    # This makes both sides of the comparison statistically grounded identically.

    EXCLUDED = {"Other", "Pass (Other)"}  # not actionable play calls

    # Overall Eagles non-TP rates (the "historical average" for each play type)
    eagles_no_tp_all = plays[
        (plays["TeamWithPossession"] == EAGLES_NAME) &
        plays["down"].isin([3, 4]) &
        plays["distance"].between(1, SHORT_YARDAGE_MAX_DISTANCE) &
        ~plays["is_tush_push"]
    ]
    pt_overall = (
        eagles_no_tp_all.groupby("play_type")
        .agg(overall_plays=("converted", "count"), overall_conv=("converted", "sum"))
    )

    # Opponent-specific counts
    pt_opp = (
        vs_no_tp.groupby("play_type")
        .agg(opp_plays=("converted", "count"), opp_conv=("converted", "sum"))
    ) if len(vs_no_tp) > 0 else pd.DataFrame(
        columns=["opp_plays", "opp_conv"]
    )

    best_alt_play = None
    best_alt_rate = 0.0
    best_alt_expl = ""

    for play_type, overall_row in pt_overall.iterrows():
        if play_type in EXCLUDED:
            continue
        opp_row  = pt_opp.loc[play_type] if play_type in pt_opp.index else None
        opp_succ = int(opp_row["opp_conv"])  if opp_row is not None else 0
        opp_att  = int(opp_row["opp_plays"]) if opp_row is not None else 0
        # Only consider plays the Eagles have actually run against this opponent.
        # If opp_att == 0 we have no opponent-specific signal — the adjusted rate
        # would be pure overall Eagles data with no relevance to this defense.
        if opp_att == 0:
            continue
        adj_rate, expl = trust_adjusted_play_rate(
            opp_succ, opp_att,
            int(overall_row["overall_conv"]), int(overall_row["overall_plays"]),
            play_type=play_type,
        )
        if adj_rate > best_alt_rate:
            best_alt_rate = adj_rate
            best_alt_play = play_type
            best_alt_expl = expl

    # ── Decision ─────────────────────────────────────────────────────────────
    subsection("RECOMMENDATION")

    alt_beats_tp = best_alt_play is not None and best_alt_rate > tp_rate_adj
    use_tp = not alt_beats_tp

    if use_tp:
        verdict = "USE THE TUSH PUSH"
        if best_alt_play:
            reason = (
                f"Adjusted Tush Push rate vs {opponent_name}: {tp_rate_adj:.1f}%. "
                f"Best alternative ({best_alt_play}) adjusted rate: {best_alt_rate:.1f}% "
                f"— the Tush Push still wins."
            )
        else:
            reason = (
                f"Adjusted Tush Push rate vs {opponent_name}: {tp_rate_adj:.1f}%. "
                f"No alternative play in the data outperforms it."
            )
    else:
        verdict = f"USE {best_alt_play.upper()} INSTEAD"
        reason  = (
            f"Adjusted Tush Push rate vs {opponent_name}: {tp_rate_adj:.1f}%. "
            f"{best_alt_play} adjusted rate: {best_alt_rate:.1f}% "
            f"— that's the stronger call against this defense."
        )

    width = 60
    print()
    print("  " + "▓" * width)
    print(f"  ▓{'VERDICT':^{width-2}}▓")
    print(f"  ▓{verdict:^{width-2}}▓")
    print("  " + "▓" * width)
    print()

    words, line = reason.split(), "  "
    for w in words:
        if len(line) + len(w) + 1 > 72:
            print(line)
            line = "  " + w
        else:
            line += (" " if line.strip() else "") + w
    if line.strip():
        print(line)

    print()
    print(f"  Adjusted TP rate : {tp_rate_adj:.1f}%")
    if best_alt_play and alt_beats_tp:
        print(f"  Best alternative : {best_alt_play} ({best_alt_rate:.1f}% adjusted)  ← better than TP")
        print(f"  How we got there : {best_alt_expl}")
    elif best_alt_play:
        print(f"  Best alternative : {best_alt_play} ({best_alt_rate:.1f}% adjusted)  ← TP still better")
        print(f"  How we got there : {best_alt_expl}")
    print()
    print("  🦅  GO BIRDS GO!  🦅")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="tush_push.py",
        description="Eagles Tush Push Opponent Recommender 🦅 — Go Birds Go!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESCRIPTION
-----------
  Given an opponent, answers: should the Eagles use the Tush Push
  against this team, or is there a better play to run instead?

OPPONENT FORMATS
  Full name    : "Dallas Cowboys", "Kansas City Chiefs"
  Nickname     : Cowboys, Chiefs, Giants, Ravens, Packers, ...
  Abbreviation : DAL, KC, NYG, BAL, GB, ...

EXAMPLES
  python tush_push.py ./data/ --opponent Cowboys
  python tush_push.py ./data/ --opponent DAL
  python tush_push.py ./data/ --opponent "Kansas City Chiefs"
        """,
    )
    parser.add_argument(
        "data_dir",
        help="Path to directory containing play-by-play CSV files (2021-2025)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        required=True,
        metavar="TEAM",
        help="Opponent to analyze (required). Accepts full name, abbreviation, or nickname.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        parser.error(
            f"Data directory not found: '{args.data_dir}'\n"
            f"  Example: python tush_push.py ./data/ --opponent Cowboys"
        )

    opponent_full = resolve_team_name(args.opponent)
    if opponent_full is None:
        print(f"\n[ERROR] Could not resolve opponent: '{args.opponent}'")
        print(f"\n  Valid abbreviations : {', '.join(sorted(TEAM_ABBR_TO_FULL.keys()))}")
        print(f"\n  Valid nicknames     : {', '.join(k.title() for k in sorted(TEAM_NICKNAMES.keys()))}")
        print(f"\n  Example: python tush_push.py {args.data_dir} --opponent Cowboys")
        sys.exit(1)

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("       EAGLES TUSH PUSH OPPONENT RECOMMENDER  🦅  GO BIRDS GO!          ")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\nOpponent       : {opponent_full}")
    print(f"Data directory : {os.path.abspath(args.data_dir)}")
    print(f"Short yardage  : 3rd/4th & 1-{SHORT_YARDAGE_MAX_DISTANCE}  (from project spec)")
    print(f"Tush Push dist : ≤ {TUSH_PUSH_MAX_DISTANCE} yard  (empirically validated vs TushPush.fyi)")

    plays = load_plays(args.data_dir)

    # Strip preseason — validated: matches published regular-season attempt counts
    pre_mask = plays["Week"].str.lower().str.contains("preseason|hall of fame", na=False)
    plays = plays[~pre_mask].copy()
    print(f"[INFO] Preseason removed. Remaining: {len(plays):,} plays.")

    plays = parse_play_start(plays)
    plays = parse_yards_gained(plays)

    qb_set = build_qb_set(plays)
    print(f"[INFO] {len(qb_set)} QBs identified from passing plays.")

    plays["is_tush_push"] = plays.apply(lambda r: is_tush_push(r, qb_set), axis=1)
    plays["play_type"]    = plays["PlayDescription"].apply(classify_play_type)
    plays["formation"]    = plays["PlayTimeFormation"].apply(extract_formation)
    plays["converted"]    = plays.apply(is_conversion, axis=1)

    recommend(plays, opponent_full, qb_set)


if __name__ == "__main__":
    main()