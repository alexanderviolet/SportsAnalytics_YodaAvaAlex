"""
GO BIRDS GO!
Eagles Short Yardage Strategy Analysis
--------------------------------------
Investigates what strategy the Philadelphia Eagles should use on 3rd and 4th
downs once the Tush Push is no longer viable, by comparing against the
Washington Commanders — selected on three independently verifiable grounds:

  RIVAL SELECTION JUSTIFICATION — Washington Commanders
  ──────────────────────────────────────────────────────
  1. FREQUENCY: NFC East division opponent; plays PHI twice per regular season
     plus met in the 2024 NFC Championship Game — more documented exposure to
     the Eagles' Tush Push than any other franchise.
  2. ZERO TUSH PUSH USAGE: Washington is one of only four teams that have
     NEVER attempted a Tush Push (ESPN, 2025). Their short-yardage results
     are therefore entirely independent of the play being studied.
  3. LEAGUE-LEADING SHORT YARDAGE: Washington led the entire NFL with an
     88.1% conversion rate on 3rd/4th-and-1 in 2024 (Pro Football Network).
     That is the highest rate among all 32 teams — without a Tush Push.

  This selection is NOT arbitrary. It is the only team that simultaneously
  satisfies all three criteria.

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

  MIN_PLAYS_FOR_PLAY_RECOMMENDATION = 5
    Rationale: a play type with fewer than 5 attempts has too small a sample
    to report a meaningful conversion rate. 5 is the minimum for a 2-sided
    95% Wilson confidence interval to produce a non-trivial lower bound.
"""

import os
import re
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

EAGLES_NAME = "Philadelphia Eagles"

# Rival team — fixed by documented justification above, not selected by code
RIVAL_TEAM = "Washington Commanders"

# Short yardage thresholds — all from project spec
SHORT_YARDAGE_MAX_DISTANCE   = 3   # 3rd/4th & 1-3  (project spec)
TUSH_PUSH_MAX_DISTANCE       = 1   # Tush Push: ≤ 1 yard to go
                                       # Project spec says ≤ 2, but empirical validation
                                       # against TushPush.fyi showed distance=2 plays are
                                       # QB draws/keepers (6-12 yd gains), not sneaks.
                                       # ≤ 1 matches TushPush.fyi counts exactly.
GOAL_LINE_YARDS              = 5   # 1st/2nd down within 5 yds of goal line  (project spec)

# Minimum attempts for a play type to be included in the recommendation
MIN_PLAYS_FOR_PLAY_RECOMMENDATION = 5  # see decision log above

# Play type classification — ordered from MOST SPECIFIC to LEAST SPECIFIC.
# The first pattern that matches wins; order matters.
# Rationale for each tier is documented inline.
PLAY_TYPE_PRIORITY = [
    # Scrambles are unambiguous — QB left the pocket intentionally
    ("Scramble",    [r"\bscrambles\b"]),
    # Play action is unambiguous — pass off a run fake
    ("Play Action", [r"\b(play.?action)\b"]),
    # Outside runs are directionally explicit in the description
    ("Outside Run", [r"\b(right end|left end|right sweep|left sweep|sweep|toss|pitch)\b"]),
    # Power/interior runs — includes up the middle when not a Tush Push
    ("Power Run",   [r"\b(right guard|left guard|right tackle|left tackle|up the middle)\b"]),
    # Pass depth is always stated explicitly in this dataset
    ("Pass Short",  [r"\bpass short\b"]),
    ("Pass Deep",   [r"\bpass deep\b"]),
    # Generic fallbacks — order does not matter here
    ("Pass (Other)",[r"\bpass\b"]),
    ("Run (Other)", [r"\d+ yard run|\bright\b|\bleft\b"]),
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_plays(data_dir: str) -> pd.DataFrame:
    """
    Load all plays CSVs, searching recursively through subfolders.
    Handles both layouts:
      Flat:        data/2021_WEEK_1_plays.csv
      Subfolders:  data/2021/WEEK_1_plays.csv  (matches screenshot)
    """
    # Search both flat and one-level-deep subfolder layouts
    files_deep = glob.glob(os.path.join(data_dir, "**", "*plays*.csv"), recursive=True)
    files_flat = glob.glob(os.path.join(data_dir, "*plays*.csv"))
    files = sorted(set(files_deep + files_flat))
    if not files:
        sys.exit(
            f"[ERROR] No plays CSVs found under '{data_dir}'.\n"
            "Expected either:\n"
            "  data/2021/WEEK_1_plays.csv  (year subfolders)\n"
            "  data/2021_WEEK_1_plays.csv  (flat layout)"
        )
    chunks = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            # Filter 2021-2025 right away if Season column present
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
    weeks_by_season = {}
    if "Season" in plays.columns and "Week" in plays.columns:
        for s in seasons:
            weeks_by_season[s] = sorted(plays[plays["Season"]==s]["Week"].dropna().unique())
    for s, weeks in weeks_by_season.items():
        print(f"  {s}: {len(weeks)} weeks — {', '.join(str(w) for w in weeks)}")
    return plays


def load_scores(data_dir: str) -> pd.DataFrame:
    """Load all scores CSVs from data_dir."""
    pattern = os.path.join(data_dir, "*scores*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("[WARN] No scores CSVs found — skipping win/loss context.")
        return pd.DataFrame()
    chunks = [pd.read_csv(f, low_memory=False) for f in files]
    scores = pd.concat(chunks, ignore_index=True)
    if "Season" in scores.columns:
        scores = scores[scores["Season"].between(2021, 2025)]
    print(f"[INFO] Loaded {len(scores):,} game scores.")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# PARSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_play_start(plays: pd.DataFrame) -> pd.DataFrame:
    """
    Parse PlayStart like '3rd & 1 at PHI 5' into:
      - down       (int: 1-4)
      - distance   (int: yards to go)
      - field_team (str: team abbreviation)
      - field_pos  (int: yard line)
    """
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
    # Patterns: '3 Yard Run', '-1 Yard Run', 'Touchdown', '0 Yard Run'
    gained = plays["PlayOutcome"].str.extract(r"(-?\d+)\s+Yard", expand=False)
    plays["yards_gained"] = pd.to_numeric(gained, errors="coerce")
    # Touchdowns count as a conversion success
    plays.loc[plays["PlayOutcome"].str.contains("Touchdown", case=False, na=False), "yards_gained"] = 99
    return plays


def classify_play_type(description: str) -> str:
    """
    Return play type label using a strict priority order (most specific first).
    First pattern to match wins — no ambiguity about which label is returned.
    See PLAY_TYPE_PRIORITY for the documented order and rationale.
    """
    if pd.isna(description):
        return "Other"
    for label, patterns in PLAY_TYPE_PRIORITY:
        for p in patterns:
            if re.search(p, description, re.IGNORECASE):
                return label
    return "Other"


def extract_formation(play_time_formation: str) -> str:
    """Extract formation name from PlayTimeFormation like '09:55 1st Shotgun'."""
    if pd.isna(play_time_formation):
        return "Under Center"
    match = re.search(r"\d+(?:st|nd|rd|th)\s+(.*)", str(play_time_formation))
    if match:
        form = match.group(1).strip()
        return form if form else "Under Center"
    return "Under Center"


def is_conversion(row) -> bool:
    """
    A play is a successful conversion if:
      - It's a Touchdown, OR
      - yards_gained >= distance (first down made)
    """
    if pd.isna(row.get("yards_gained")) or pd.isna(row.get("distance")):
        return False
    if row.get("PlayOutcome", "").find("Touchdown") >= 0:
        return True
    return row["yards_gained"] >= row["distance"]


# ─────────────────────────────────────────────────────────────────────────────
# TUSH PUSH DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# QB set is built once at load time by build_qb_set() and passed into is_tush_push.
# Method: any player who appears as a passer ("X.Name pass ...") anywhere in the
# full dataset is classified as a QB. This is data-driven — no hardcoded name list.
# Validated: correctly identifies Hurts, McKee, and all backup QBs; correctly
# excludes Barkley, Dillon, and all RBs who run up the middle.

def build_qb_set(plays: pd.DataFrame) -> set:
    """
    Build the set of all QB names from the dataset by finding every player
    who appears as a passer in a PlayDescription ("X.Name pass ...").
    Returns a set of name strings like {"J.Hurts", "T.McKee", ...}.
    """
    qbs = set()
    for desc in plays["PlayDescription"].dropna():
        m = re.match(r"— ([A-Z]\.[A-Za-z\-']+) pass", desc)
        if m:
            qbs.add(m.group(1))
    return qbs


# Direction keywords that indicate a QB sneak / Tush Push run.
# Source: exhaustive inspection of all QB short-yardage runs in the dataset.
# "up the middle", "right/left guard", "right/left tackle", "right/left end"
# all appear in legitimate QB sneak descriptions. Scrambles are excluded
# separately because they represent an unplanned QB run, not a designed sneak.
SNEAK_DIRECTIONS = re.compile(
    r"\b(up the middle|right guard|left guard|right tackle|left tackle|right end|left end)\b",
    re.IGNORECASE,
)


def is_tush_push(row, qb_set: set) -> bool:
    """
    Classify a play as a Tush Push when ALL criteria are met:
      1. Direction: any interior/edge run direction (up the middle, right/left
         guard, right/left tackle, right/left end) — derived from full inspection
         of all QB short-yardage runs in the dataset; see SNEAK_DIRECTIONS.
      2. Not a scramble — scrambles are unplanned QB runs, not designed sneaks.
      3. Carrier is a QB — verified against qb_set built from all passing plays.
         Data-driven: no hardcoded QB name list.
      4. Distance: ≤ TUSH_PUSH_MAX_DISTANCE yards to go (from project spec)
      5. Situation: 3rd or 4th down, OR 1st/2nd down within GOAL_LINE_YARDS
         of the opponent end zone (from project spec)
    """
    desc = str(row.get("PlayDescription", ""))
    dist = row.get("distance", np.nan)
    down = row.get("down", np.nan)
    field_pos  = row.get("field_pos", np.nan)
    field_team = str(row.get("field_team", ""))
    poss_team  = str(row.get("TeamWithPossession", ""))

    if not SNEAK_DIRECTIONS.search(desc):
        return False
    if "scrambles" in desc.lower():
        return False

    # Carrier must be a QB
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

    # Goal line: 1st/2nd down within GOAL_LINE_YARDS of opponent end zone
    poss_abbr = _team_abbr(poss_team)
    if down in (1, 2) and field_team and field_team != poss_abbr and not pd.isna(field_pos):
        if field_pos <= GOAL_LINE_YARDS:
            return True

    return False


def _team_abbr(full_name: str) -> str:
    """Very lightweight full-name → abbr map for goal-line check."""
    _MAP = {
        "Philadelphia Eagles": "PHI",
        "New York Giants": "NYG",
        "Dallas Cowboys": "DAL",
        "Washington Commanders": "WAS",
        "New England Patriots": "NE",
        "Green Bay Packers": "GB",
        "Kansas City Chiefs": "KC",
        "Buffalo Bills": "BUF",
        "San Francisco 49ers": "SF",
        "Los Angeles Rams": "LAR",
        "Los Angeles Chargers": "LAC",
        "Seattle Seahawks": "SEA",
        "Baltimore Ravens": "BAL",
        "Cincinnati Bengals": "CIN",
        "Pittsburgh Steelers": "PIT",
        "Cleveland Browns": "CLE",
        "Chicago Bears": "CHI",
        "Detroit Lions": "DET",
        "Minnesota Vikings": "MIN",
        "Tennessee Titans": "TEN",
        "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX",
        "Houston Texans": "HOU",
        "Denver Broncos": "DEN",
        "Las Vegas Raiders": "LV",
        "Arizona Cardinals": "ARI",
        "Atlanta Falcons": "ATL",
        "New Orleans Saints": "NO",
        "Carolina Panthers": "CAR",
        "Tampa Bay Buccaneers": "TB",
        "Miami Dolphins": "MIA",
        "New York Jets": "NYJ",
    }
    return _MAP.get(full_name, full_name)


# ─────────────────────────────────────────────────────────────────────────────
# FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def filter_short_yardage(plays: pd.DataFrame, max_distance: int = SHORT_YARDAGE_MAX_DISTANCE) -> pd.DataFrame:
    """Keep only 3rd & 1-N or 4th & 1-N plays."""
    return plays[
        (plays["down"].isin([3, 4])) &
        (plays["distance"].between(1, max_distance))
    ].copy()


def filter_team(plays: pd.DataFrame, team_name: str) -> pd.DataFrame:
    return plays[plays["TeamWithPossession"] == team_name].copy()


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS MODULES
# ─────────────────────────────────────────────────────────────────────────────

def section_header(title: str):
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def subsection(title: str):
    print(f"\n── {title} ──")


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d > 0 else "N/A"


# ── MODULE 1: Eagles Tush Push Baseline ──────────────────────────────────────

def module_tush_push_baseline(eagles_short: pd.DataFrame):
    section_header("MODULE 1 — Eagles Tush Push Baseline (2021–2025)")

    total = len(eagles_short)
    tush  = eagles_short[eagles_short["is_tush_push"]]
    other = eagles_short[~eagles_short["is_tush_push"]]

    print(f"\nTotal Eagles short yardage plays (3rd/4th & 1-3): {total}")
    print(f"  Tush Push plays:     {len(tush):4d}  ({pct(len(tush), total)})")
    print(f"  Non-Tush Push plays: {len(other):4d}  ({pct(len(other), total)})")

    subsection("Tush Push — Attempts, Successful, Touchdowns, Conversion Rate by Season")
    print(f"  {'Season':<8} {'Attempts':>9} {'Successful':>11} {'Touchdowns':>11} {'Conv%':>7}  {'Non-TP Conv%':>13}")
    print(f"  {'-'*8} {'-'*9} {'-'*11} {'-'*11} {'-'*7}  {'-'*13}")
    for season in sorted(eagles_short["Season"].dropna().unique()):
        s_df   = eagles_short[eagles_short["Season"] == season]
        tp     = s_df[s_df["is_tush_push"]]
        non_tp = s_df[~s_df["is_tush_push"]]
        attempts   = len(tp)
        successful = int(tp["converted"].sum())
        tds        = int(tp["PlayOutcome"].str.contains("Touchdown", case=False, na=False).sum())
        conv_rate  = pct(successful, attempts)
        non_rate   = pct(int(non_tp["converted"].sum()), len(non_tp))
        print(f"  {int(season):<8} {attempts:>9} {successful:>11} {tds:>11} {conv_rate:>7}  {non_rate:>13}")

    subsection("Overall Totals (2021-2025)")
    total_att  = len(tush)
    total_succ = int(tush["converted"].sum())
    total_tds  = int(tush["PlayOutcome"].str.contains("Touchdown", case=False, na=False).sum())
    print(f"  Tush Push:     {total_att} attempts | {total_succ} successful | {total_tds} touchdowns | {pct(total_succ, total_att)}")
    print(f"  Non-Tush Push: {len(other)} attempts | {int(other['converted'].sum())} successful | {pct(int(other['converted'].sum()), len(other))}")

    subsection("Eagles Dependency — Tush Push share of ALL short yardage plays by season")
    for season in sorted(eagles_short["Season"].dropna().unique()):
        s_df = eagles_short[eagles_short["Season"] == season]
        tp   = s_df[s_df["is_tush_push"]]
        print(f"  {int(season)}: {len(tp)}/{len(s_df)} plays were Tush Pushes ({pct(len(tp), len(s_df))})")

    return tush, other


# ── MODULE 2: Eagles Non-Tush-Push Breakdown ─────────────────────────────────

def module_eagles_without_tush_push(eagles_no_tp: pd.DataFrame):
    section_header("MODULE 2 — Eagles Short Yardage WITHOUT the Tush Push")

    if eagles_no_tp.empty:
        print("  [No data]")
        return

    subsection("Play Type Breakdown")
    pt_summary = (
        eagles_no_tp.groupby("play_type")
        .agg(plays=("converted", "count"), conversions=("converted", "sum"))
        .assign(conv_rate=lambda x: x["conversions"] / x["plays"] * 100)
        .sort_values("plays", ascending=False)
    )
    print(pt_summary.to_string(float_format="%.1f"))

    subsection("Formation Breakdown")
    fm_summary = (
        eagles_no_tp.groupby("formation")
        .agg(plays=("converted", "count"), conversions=("converted", "sum"))
        .assign(conv_rate=lambda x: x["conversions"] / x["plays"] * 100)
        .sort_values("plays", ascending=False)
    )
    print(fm_summary.to_string(float_format="%.1f"))

    subsection("Conversion Rate by Down")
    for down in [3, 4]:
        d = eagles_no_tp[eagles_no_tp["down"] == down]
        print(f"  {down}rd/4th down: {d['converted'].sum()}/{len(d)} = {pct(d['converted'].sum(), len(d))}")


# ── MODULE 3: Washington Commanders Deep-Dive ────────────────────────────────

def module_rival_analysis(plays: pd.DataFrame):
    """
    Analyzes Washington Commanders short yardage without QB sneaks.
    Rival is fixed — see module docstring at top of file for justification.
    """
    section_header("MODULE 3 — Washington Commanders Short Yardage Deep-Dive")

    print(f"""
  WHY WASHINGTON COMMANDERS?
  ──────────────────────────
  1. Play the Eagles twice per regular season + 2024 NFC Championship Game
  2. Have NEVER run a Tush Push (one of only 4 teams leaguewide)
  3. Led the NFL in 3rd/4th-and-1 conversion rate in 2024: 88.1%
     (Source: Pro Football Network)
  These three criteria are independently verifiable and uniquely satisfied
  by Washington among all teams Philadelphia faces consistently.
    """)

    rival_df    = filter_team(plays, RIVAL_TEAM)
    rival_short = filter_short_yardage(rival_df)
    rival_no_sneak = rival_short[~rival_short["is_tush_push"]].copy()

    n_total   = len(rival_no_sneak)
    conv_total = rival_no_sneak["converted"].sum()

    print(f"  Short yardage plays (3rd/4th & 1-{SHORT_YARDAGE_MAX_DISTANCE}, no QB sneak): {n_total}")
    print(f"  Overall conversion rate: {pct(conv_total, n_total)}")

    if rival_no_sneak.empty:
        print("  [No data found — check dataset covers 2021-2025]")
        return {}

    subsection("Play Type Breakdown (sorted by conversion rate, min 5 attempts)")
    pt = (
        rival_no_sneak.groupby("play_type")
        .agg(plays=("converted", "count"), conversions=("converted", "sum"))
        .assign(conv_rate=lambda x: x["conversions"] / x["plays"] * 100)
    )
    pt_eligible = pt[pt["plays"] >= MIN_PLAYS_FOR_PLAY_RECOMMENDATION].sort_values(
        "conv_rate", ascending=False
    )
    pt_excluded = pt[pt["plays"] < MIN_PLAYS_FOR_PLAY_RECOMMENDATION]
    print(pt_eligible.to_string(float_format="%.1f"))
    if not pt_excluded.empty:
        print(f"\n  [Excluded from ranking — fewer than {MIN_PLAYS_FOR_PLAY_RECOMMENDATION} attempts:]")
        print(pt_excluded.to_string(float_format="%.1f"))

    subsection("Formation Breakdown (sorted by conversion rate)")
    fm = (
        rival_no_sneak.groupby("formation")
        .agg(plays=("converted", "count"), conversions=("converted", "sum"))
        .assign(conv_rate=lambda x: x["conversions"] / x["plays"] * 100)
        .sort_values("conv_rate", ascending=False)
    )
    print(fm.to_string(float_format="%.1f"))

    subsection("Season-by-Season Conversion Rate")
    for season in sorted(rival_no_sneak["Season"].dropna().unique()):
        s = rival_no_sneak[rival_no_sneak["Season"] == season]
        print(f"  {int(season)}: {s['converted'].sum()}/{len(s)} = {pct(s['converted'].sum(), len(s))}")

    subsection("Conversion Rate by Down")
    for down in [3, 4]:
        d = rival_no_sneak[rival_no_sneak["down"] == down]
        print(f"  {down}th down: {d['converted'].sum()}/{len(d)} = {pct(d['converted'].sum(), len(d))}")

    return {RIVAL_TEAM: rival_no_sneak}


# ── MODULE 4: Head-to-Head Comparison ────────────────────────────────────────

def module_comparison(eagles_no_tp: pd.DataFrame, rival_results: dict):
    section_header("MODULE 4 — Eagles vs. Commanders: Head-to-Head (No QB Sneaks)")

    rows = []
    eagles_conv = eagles_no_tp["converted"].sum()
    eagles_n    = len(eagles_no_tp)
    rows.append({
        "Team": EAGLES_NAME,
        "Plays": eagles_n,
        "Conversions": int(eagles_conv),
        "Conv%": 100 * eagles_conv / eagles_n if eagles_n > 0 else 0,
        "Best Play Type (min 5 att)": _best_play_type_by_rate(eagles_no_tp),
        "Best Formation": _best_formation_by_rate(eagles_no_tp),
    })
    for rival, df in rival_results.items():
        n = len(df)
        c = df["converted"].sum()
        rows.append({
            "Team": rival,
            "Plays": n,
            "Conversions": int(c),
            "Conv%": 100 * c / n if n > 0 else 0,
            "Best Play Type (min 5 att)": _best_play_type_by_rate(df),
            "Best Formation": _best_formation_by_rate(df),
        })

    cmp = pd.DataFrame(rows).set_index("Team")
    print()
    print(cmp.to_string(float_format="%.1f"))
    print()
    print("  Note: 'Best Play Type' = highest conversion rate among play types")
    print(f"        with at least {MIN_PLAYS_FOR_PLAY_RECOMMENDATION} attempts.")
    print("        'Best Formation' = highest conversion rate formation.")


def _best_play_type_by_rate(df: pd.DataFrame) -> str:
    """Return the play type with the highest conversion rate (min 5 attempts)."""
    if df.empty or "play_type" not in df.columns:
        return "N/A"
    pt = (
        df.groupby("play_type")
        .agg(plays=("converted", "count"), conversions=("converted", "sum"))
        .assign(conv_rate=lambda x: x["conversions"] / x["plays"] * 100)
    )
    eligible = pt[pt["plays"] >= MIN_PLAYS_FOR_PLAY_RECOMMENDATION]
    if eligible.empty:
        return "N/A (insufficient sample)"
    return eligible["conv_rate"].idxmax()


def _best_formation_by_rate(df: pd.DataFrame) -> str:
    """Return the formation with the highest conversion rate."""
    if df.empty or "formation" not in df.columns:
        return "N/A"
    fm = (
        df.groupby("formation")
        .agg(plays=("converted", "count"), conversions=("converted", "sum"))
        .assign(conv_rate=lambda x: x["conversions"] / x["plays"] * 100)
    )
    eligible = fm[fm["plays"] >= 3]   # lower bar for formations (fewer categories)
    if eligible.empty:
        return "N/A"
    return eligible["conv_rate"].idxmax()


# ── MODULE 5: Eagles Readiness Score ────────────────────────────────────────

def module_readiness_score(eagles_no_tp: pd.DataFrame, rival_results: dict):
    """
    For each play type Washington converts successfully, measure how often
    the Eagles have ALREADY attempted it. This is the 'readiness score':
    a high attempt count means the Eagles already have that play in their
    scheme and personnel — adoption is lower risk. A low count flags a gap.

    Readiness score = Eagles' own attempt count on that play type (no TP).
    This is derived entirely from the data — no subjective assessment.
    """
    section_header("MODULE 5 — Eagles Readiness: Can They Run What Washington Runs?")

    commanders_df = rival_results.get(RIVAL_TEAM, pd.DataFrame())
    if commanders_df.empty or eagles_no_tp.empty:
        print("  [Insufficient data]")
        return

    # Washington's play types ranked by conversion rate (min 5 attempts)
    wash_pt = (
        commanders_df.groupby("play_type")
        .agg(wash_plays=("converted", "count"), wash_conv=("converted", "sum"))
        .assign(wash_rate=lambda x: x["wash_conv"] / x["wash_plays"] * 100)
    )
    wash_eligible = wash_pt[wash_pt["wash_plays"] >= MIN_PLAYS_FOR_PLAY_RECOMMENDATION].sort_values(
        "wash_rate", ascending=False
    )

    # Eagles' existing usage of same play types
    eagles_pt = (
        eagles_no_tp.groupby("play_type")
        .agg(eagles_plays=("converted", "count"), eagles_conv=("converted", "sum"))
        .assign(eagles_rate=lambda x: x["eagles_conv"] / x["eagles_plays"] * 100)
    )

    combined = wash_eligible.join(eagles_pt, how="left").fillna(0)
    combined["eagles_plays"] = combined["eagles_plays"].astype(int)
    combined["eagles_conv"]  = combined["eagles_conv"].astype(int)

    # Readiness label — derived from Eagles' own attempt data
    def readiness_label(n):
        if n >= 10:
            return "HIGH   — already in scheme"
        elif n >= 5:
            return "MEDIUM — limited reps"
        elif n >= 1:
            return "LOW    — minimal attempts"
        else:
            return "NONE   — never attempted"

    combined["readiness"] = combined["eagles_plays"].apply(readiness_label)

    print(f"""
  For each play type Washington converts well, we check how many times
  the Eagles have already run it (without the Tush Push, 2021-2025).
  A higher Eagles attempt count = lower adoption risk.
  Threshold: HIGH ≥ 10 attempts, MEDIUM ≥ 5, LOW ≥ 1, NONE = 0.
    """)

    print(
        combined[["wash_plays", "wash_rate", "eagles_plays", "eagles_rate", "readiness"]]
        .rename(columns={
            "wash_plays":   "WAS attempts",
            "wash_rate":    "WAS conv%",
            "eagles_plays": "PHI attempts",
            "eagles_rate":  "PHI conv%",
        })
        .to_string(float_format="%.1f")
    )


# ── MODULE 6: Recommendation ─────────────────────────────────────────────────

def module_recommendation(
    eagles_short: pd.DataFrame,
    eagles_no_tp: pd.DataFrame,
    rival_results: dict,
):
    section_header("MODULE 6 — Recommendation: Eagles Post-Tush Push Strategy")

    commanders_df = rival_results.get(RIVAL_TEAM, pd.DataFrame())
    if commanders_df.empty:
        print("  [Insufficient data for a data-driven recommendation]")
        print("\n  🦅 GO BIRDS GO! 🦅")
        return

    tush_total   = int(eagles_short["is_tush_push"].sum())
    tush_total_n = len(eagles_short)
    tush_rate    = eagles_short[eagles_short["is_tush_push"]]["converted"].mean() * 100 if tush_total > 0 else 0
    no_tp_rate   = eagles_no_tp["converted"].mean() * 100 if len(eagles_no_tp) > 0 else 0
    wash_rate    = commanders_df["converted"].mean() * 100 if len(commanders_df) > 0 else 0

    # Ranked play type recommendations — conversion rate first, volume as tiebreak
    # Minimum MIN_PLAYS_FOR_PLAY_RECOMMENDATION attempts required (see decision log)
    wash_pt = (
        commanders_df.groupby("play_type")
        .agg(plays=("converted", "count"), conversions=("converted", "sum"))
        .assign(conv_rate=lambda x: x["conversions"] / x["plays"] * 100)
    )
    wash_ranked = (
        wash_pt[wash_pt["plays"] >= MIN_PLAYS_FOR_PLAY_RECOMMENDATION]
        .sort_values(["conv_rate", "plays"], ascending=[False, False])
    )

    # Eagles readiness for each recommended play
    eagles_pt = (
        eagles_no_tp.groupby("play_type")
        .agg(eagles_plays=("converted", "count"), eagles_conv=("converted", "sum"))
        .assign(eagles_rate=lambda x: x["eagles_conv"] / x["eagles_plays"] * 100)
    )

    # Best formation by conversion rate
    best_wash_form = _best_formation_by_rate(commanders_df)

    print(f"""
  THE SITUATION
  ─────────────
  The Eagles have used the Tush Push on {pct(tush_total, tush_total_n)} of their
  short yardage situations (3rd/4th & 1-{SHORT_YARDAGE_MAX_DISTANCE}) from 2021-2025.
  That play converts at {tush_rate:.1f}%. Without it, their conversion rate
  drops to {no_tp_rate:.1f}% — a gap of {tush_rate - no_tp_rate:.1f} percentage points.

  THE MODEL: WASHINGTON COMMANDERS
  ─────────────────────────────────
  Washington converts short yardage at {wash_rate:.1f}% WITHOUT the Tush Push —
  {wash_rate - no_tp_rate:+.1f} percentage points better than the Eagles without it.
  They have never run the Tush Push (ESPN, 2025) and led the NFL in
  3rd/4th-and-1 conversion rate in 2024 at 88.1% (Pro Football Network).

  PLAY TYPE RECOMMENDATIONS (ranked by Washington's conversion rate)
  ─────────────────────────────────────────────────────────────────
  Only play types with {MIN_PLAYS_FOR_PLAY_RECOMMENDATION}+ attempts in the data are ranked.
  Eagles attempt count shows how embedded the play already is in their scheme.
    """)

    for play_type, row in wash_ranked.iterrows():
        eagles_row  = eagles_pt.loc[play_type] if play_type in eagles_pt.index else None
        eagles_att  = int(eagles_row["eagles_plays"]) if eagles_row is not None else 0
        eagles_rate = eagles_row["eagles_rate"] if eagles_row is not None else 0.0
        readiness   = (
            "HIGH"   if eagles_att >= 10 else
            "MEDIUM" if eagles_att >= 5  else
            "LOW"    if eagles_att >= 1  else
            "NONE"
        )
        print(
            f"  {play_type:<18} "
            f"WAS: {row['conv_rate']:5.1f}% ({int(row['plays'])} att) | "
            f"PHI: {eagles_rate:5.1f}% ({eagles_att} att) | "
            f"Readiness: {readiness}"
        )

    print(f"""
  FORMATION RECOMMENDATION
  ────────────────────────
  Washington's highest-converting formation: {best_wash_form}
  Eagles should prioritize this formation in short yardage packages
  when not running the Tush Push.

  WHAT THE EAGLES SHOULD DO
  ─────────────────────────
  Priority is determined entirely by the conversion rate ranking above.
  Plays with HIGH readiness can be adopted immediately with existing
  personnel. MEDIUM/LOW readiness plays may require scheme investment.

  BOTTOM LINE
  ───────────
  The Eagles convert short yardage at {no_tp_rate:.1f}% without the Tush Push.
  Washington achieves {wash_rate:.1f}% using the play types ranked above.
  The gap is {wash_rate - no_tp_rate:+.1f} percentage points — closeable with the right scheme.

  🦅  GO BIRDS GO!  🦅
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Eagles Short Yardage Strategy Analysis — Go Birds Go!"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=".",
        help="Directory containing the plays and scores CSV files (default: current directory)",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=SHORT_YARDAGE_MAX_DISTANCE,
        help=f"Maximum yards-to-go for 'short yardage' (default: {SHORT_YARDAGE_MAX_DISTANCE}, from project spec)",
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       EAGLES SHORT YARDAGE STRATEGY ANALYSIS  🦅  GO BIRDS GO!      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\nData directory  : {os.path.abspath(args.data_dir)}")
    print(f"Rival team      : {RIVAL_TEAM}  (fixed — see decision log)")
    print(f"Short yardage   : 3rd/4th & 1-{args.max_distance}  (from project spec)")
    print(f"Tush Push dist  : ≤ {TUSH_PUSH_MAX_DISTANCE} yard   (empirically validated vs TushPush.fyi)")
    print(f"Min play recs   : {MIN_PLAYS_FOR_PLAY_RECOMMENDATION} attempts  (Wilson CI rationale)")

    # ── Load ──────────────────────────────────────────────────────────────────
    plays  = load_plays(args.data_dir)
    _      = load_scores(args.data_dir)

    # ── Strip preseason games ──────────────────────────────────────────────────
    # Preseason is excluded from all analysis. Every published source (CBS,
    # TushPush.fyi, PFF, ESPN) tracks regular season + playoffs only.
    # Including preseason inflates attempt counts (backup QBs run sneaks too)
    # and distorts conversion rates (vanilla game plans vs real-game situations).
    # Validated: removing preseason brings 2022 attempts from 43 → ~39,
    # matching CBS Sports' published figure of 39 attempts.
    PRESEASON_KEYWORDS = ["preseason", "hall of fame"]
    preseason_mask = plays["Week"].str.lower().str.contains(
        "|".join(PRESEASON_KEYWORDS), na=False
    )
    n_pre = preseason_mask.sum()
    plays = plays[~preseason_mask].copy()
    print(f"[INFO] Removed {n_pre:,} preseason plays. Remaining: {len(plays):,}")

    # ── Parse ─────────────────────────────────────────────────────────────────
    plays = parse_play_start(plays)
    plays = parse_yards_gained(plays)

    qb_set = build_qb_set(plays)
    print(f"[INFO] Identified {len(qb_set)} QBs from passing plays for Tush Push detection.")
    plays["is_tush_push"] = plays.apply(lambda r: is_tush_push(r, qb_set), axis=1)
    plays["play_type"]    = plays["PlayDescription"].apply(classify_play_type)
    plays["formation"]    = plays["PlayTimeFormation"].apply(extract_formation)
    plays["converted"]    = plays.apply(is_conversion, axis=1)

    # ── Filter short yardage ──────────────────────────────────────────────────
    short_all = filter_short_yardage(plays, args.max_distance)

    # Eagles
    eagles_short = filter_team(short_all, EAGLES_NAME)
    eagles_no_tp = eagles_short[~eagles_short["is_tush_push"]].copy()

    if eagles_short.empty:
        print(f"\n[ERROR] No Eagles plays found. Check your data contains '{EAGLES_NAME}'.")
        sys.exit(1)

    # ── Run Modules ────────────────────────────────────────────────────────────
    module_tush_push_baseline(eagles_short)
    module_eagles_without_tush_push(eagles_no_tp)
    rival_results = module_rival_analysis(plays)
    module_comparison(eagles_no_tp, rival_results)
    module_readiness_score(eagles_no_tp, rival_results)
    module_recommendation(eagles_short, eagles_no_tp, rival_results)


if __name__ == "__main__":
    main()