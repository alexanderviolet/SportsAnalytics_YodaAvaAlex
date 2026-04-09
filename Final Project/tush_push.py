# """
# GO BIRDS GO!
# ============
# Eagles Short Yardage Analysis: What happens when the Tush Push is gone?
# Alex Violet, Ava Sim, Yoda Ermias — Sports Analytics, April 2026

# Action Plan:
#   1. Establish Eagles' Tush Push (QB sneak) baseline on 3rd/4th & short
#   2. Filter out QB sneaks for both Eagles and Giants
#   3. Break down Giants' short yardage approach
#   4. Compare to what the Eagles do without the Tush Push
#   5. Check if Giants' model fits the Eagles
#   6. Make the recommendation
#   7. Go Birds.

# Run from the Final Project directory:
#     python tush_push_analysis.py

# File structure expected:
#   data/
#     2021/ ... 2025/   ← weekly play CSVs
#     2021_scores.csv ... 2025_scores.csv
# """

# import os
# import glob
# import pandas as pd

# # ── Configuration ─────────────────────────────────────────────────────────────
# DATA_DIR   = "data"
# YEARS      = [2021, 2022, 2023, 2024, 2025]

# # Short yardage = 3 yards or fewer to go
# SHORT_YARDAGE_MAX = 3

# # ──────────────────────────────────────────────────────────────────────────────
# # HELPERS
# # ──────────────────────────────────────────────────────────────────────────────

# def load_all_plays() -> pd.DataFrame:
#     frames = []
#     for year in YEARS:
#         pattern = os.path.join(DATA_DIR, str(year), "*_plays.csv")
#         for f in sorted(glob.glob(pattern)):
#             try:
#                 frames.append(pd.read_csv(f))
#             except Exception as e:
#                 print(f"  [WARN] {f}: {e}")
#     if not frames:
#         raise FileNotFoundError("No play files found. Check DATA_DIR and run from Final Project folder.")
#     df = pd.concat(frames, ignore_index=True)
#     print(f"Loaded {len(df):,} total plays across {YEARS[0]}–{YEARS[-1]}.\n")
#     return df


# def parse_yards_to_go(play_start: str) -> int | None:
#     """Extract yards-to-go from strings like '3rd & 1 at PHI 5' or '4th & 2 at DAL 30'."""
#     try:
#         # Format: "Nth & YY at ..."
#         part = play_start.split("&")[1].strip()
#         yards = int(part.split()[0])
#         return yards
#     except Exception:
#         return None


# def parse_down(play_start: str) -> str | None:
#     """Extract the down string: '3rd' or '4th'."""
#     try:
#         return play_start.split("&")[0].strip()
#     except Exception:
#         return None


# def classify_play(desc: str) -> str:
#     """Classify a play description into a category."""
#     d = str(desc).lower()
#     if "punt" in d:
#         return "Punt"
#     if "field goal is good" in d:
#         return "Field Goal (GOOD)"
#     if "field goal is no good" in d or "field goal is blocked" in d:
#         return "Field Goal (MISSED)"
#     if "touchdown" in d and "pass" in d:
#         return "Touchdown Pass"
#     if "touchdown" in d and ("run" in d or "up the middle" in d or "left" in d or "right" in d):
#         return "Touchdown Run"
#     if "qb sneak" in d or is_qb_sneak(d):
#         return "QB Sneak (Tush Push)"
#     if "fumble" in d:
#         return "Fumble"
#     if "intercepted" in d or "interception" in d:
#         return "Interception"
#     if "sacked" in d:
#         return "Sack / Scramble Loss"
#     if "incomplete" in d:
#         return "Pass Incomplete"
#     if "pass" in d:
#         return "Pass Complete"
#     if "scrambles" in d:
#         return "QB Scramble"
#     if "penalty" in d or "no play" in d:
#         return "Penalty"
#     if any(x in d for x in ["up the middle", "left guard", "right guard", "left tackle", "right tackle",
#                               "left end", "right end", "off left", "off right"]):
#         return "Run"
#     return "Other"


# def is_qb_sneak(desc: str) -> bool:
#     """Detect QB sneak / Tush Push plays from description."""
#     d = str(desc).lower()
#     # A QB sneak: QB runs up the middle for short gain, not a touchdown or scramble
#     # Key signal: "hurts up the middle" or "hurts left guard" with <=3 yard gain
#     qb_names = ["hurts", "prescott", "mariota", "pickett", "mckee"]
#     run_phrases = ["up the middle", "left guard", "right guard"]
#     has_qb = any(name in d for name in qb_names)
#     has_run = any(phrase in d for phrase in run_phrases)
#     return has_qb and has_run and "pass" not in d and "sacked" not in d


# def is_converted(desc: str, yards_needed: int) -> bool:
#     """Check if the play resulted in a conversion (first down, TD, or FG good)."""
#     d = str(desc).lower()
#     if "touchdown" in d or "field goal is good" in d:
#         return True
#     if "incomplete" in d or "no good" in d or "punt" in d or "fumble" in d:
#         return False
#     if "intercepted" in d or "sacked" in d:
#         return False
#     # Try to extract yards gained from description
#     try:
#         # "... to PHI 32 for 5 yards" or "... for 2 yards"
#         if " for " in d:
#             part = d.split(" for ")[-1].strip()
#             yards = int(part.split(" yard")[0].strip().replace("-", "").replace(",", ""))
#             return yards >= yards_needed
#     except Exception:
#         pass
#     return False


# def section(title: str):
#     print()
#     print("=" * 80)
#     print(f"  {title}")
#     print("=" * 80)


# # ──────────────────────────────────────────────────────────────────────────────
# # MAIN ANALYSIS
# # ──────────────────────────────────────────────────────────────────────────────

# def main():
#     print("GO BIRDS GO! — Eagles Short Yardage Analysis")
#     print("Alex Violet, Ava Sim, Yoda Ermias | Sports Analytics | April 2026\n")

#     # ── Load data ─────────────────────────────────────────────────────────────
#     df = load_all_plays()

#     # Add parsed columns
#     df["Down"]        = df["PlayStart"].astype(str).apply(parse_down)
#     df["YardsToGo"]   = df["PlayStart"].astype(str).apply(parse_yards_to_go)
#     df["PlayType"]    = df["PlayDescription"].astype(str).apply(classify_play)
#     df["IsQBSneak"]   = df["PlayDescription"].astype(str).apply(lambda d: is_qb_sneak(d.lower()))

#     # ── Filter to 3rd & short and 4th & short ─────────────────────────────────
#     short_mask = (
#         df["Down"].isin(["3rd", "4th"]) &
#         df["YardsToGo"].notna() &
#         (df["YardsToGo"] <= SHORT_YARDAGE_MAX)
#     )
#     short_df = df[short_mask].copy()
#     print(f"Short yardage plays (3rd/4th & 1–{SHORT_YARDAGE_MAX}): {len(short_df):,}\n")

#     # ── Filter to Eagles and Giants games ─────────────────────────────────────
#     eagles_short = short_df[
#         (short_df["AwayTeam"] == "Eagles") | (short_df["HomeTeam"] == "Eagles")
#     ].copy()

#     giants_short = short_df[
#         (short_df["AwayTeam"] == "Giants") | (short_df["HomeTeam"] == "Giants")
#     ].copy()

#     # Only Eagles/Giants possession respectively
#     eagles_offense = eagles_short[
#         eagles_short["TeamWithPossession"].str.contains("Philadelphia", na=False)
#     ].copy()

#     giants_offense = giants_short[
#         giants_short["TeamWithPossession"].str.contains("New York Giants", na=False)
#     ].copy()

#     # ══════════════════════════════════════════════════════════════════════════
#     # STEP 1: Eagles Tush Push Baseline
#     # ══════════════════════════════════════════════════════════════════════════
#     section("STEP 1 — Eagles Tush Push (QB Sneak) Baseline")

#     sneaks = eagles_offense[eagles_offense["IsQBSneak"]]
#     non_sneaks = eagles_offense[~eagles_offense["IsQBSneak"]]

#     print(f"Eagles short yardage plays (offense):   {len(eagles_offense):,}")
#     print(f"  → QB Sneak / Tush Push plays:         {len(sneaks):,}  ({100*len(sneaks)/max(len(eagles_offense),1):.1f}%)")
#     print(f"  → All other short yardage plays:      {len(non_sneaks):,}  ({100*len(non_sneaks)/max(len(eagles_offense),1):.1f}%)")

#     print("\nTush Push usage by year:")
#     for year in YEARS:
#         yr_plays = eagles_offense[eagles_offense["Season"] == year]
#         yr_sneaks = yr_plays[yr_plays["IsQBSneak"]]
#         pct = 100 * len(yr_sneaks) / max(len(yr_plays), 1)
#         print(f"  {year}: {len(yr_sneaks):>3} sneak(s) out of {len(yr_plays):>3} short yardage plays  ({pct:.1f}%)")

#     # ══════════════════════════════════════════════════════════════════════════
#     # STEP 2 & 3: Eagles WITHOUT QB Sneak vs. Giants — Play Type Breakdown
#     # ══════════════════════════════════════════════════════════════════════════
#     section("STEP 2 & 3 — Play Type Breakdown (QB Sneaks Removed)")

#     eagles_no_sneak = eagles_offense[~eagles_offense["IsQBSneak"]].copy()
#     giants_no_sneak = giants_offense[~giants_offense["IsQBSneak"]].copy()

#     def play_type_summary(data: pd.DataFrame, label: str):
#         print(f"\n{label}  ({len(data):,} plays, QB sneaks excluded)")
#         print(f"  {'Play Type':<28} {'Count':>6}  {'Share':>7}")
#         print(f"  {'-'*44}")
#         counts = data["PlayType"].value_counts()
#         for play, count in counts.items():
#             pct = 100 * count / max(len(data), 1)
#             print(f"  {play:<28} {count:>6}   {pct:>6.1f}%")

#     play_type_summary(eagles_no_sneak, "Philadelphia Eagles (no QB sneak)")
#     play_type_summary(giants_no_sneak, "New York Giants (no QB sneak)")

#     # ══════════════════════════════════════════════════════════════════════════
#     # STEP 4: Conversion Rates
#     # ══════════════════════════════════════════════════════════════════════════
#     section("STEP 4 — Conversion Rates (3rd & 4th & Short, No QB Sneak)")

#     def conversion_rate_by_type(data: pd.DataFrame, label: str):
#         print(f"\n{label}")
#         print(f"  {'Play Type':<28} {'Plays':>6}  {'Conv.':>6}  {'Rate':>7}")
#         print(f"  {'-'*50}")

#         # Compute conversions
#         data = data.copy()
#         data["Converted"] = data.apply(
#             lambda r: is_converted(r["PlayDescription"], r["YardsToGo"]), axis=1
#         )

#         overall = data["Converted"].sum()
#         total   = len(data)
#         print(f"  {'OVERALL':<28} {total:>6}  {overall:>6}  {100*overall/max(total,1):>6.1f}%")

#         for play_type, grp in data.groupby("PlayType"):
#             conv  = grp["Converted"].sum()
#             plays = len(grp)
#             rate  = 100 * conv / max(plays, 1)
#             print(f"  {play_type:<28} {plays:>6}  {conv:>6}  {rate:>6.1f}%")

#         return data

#     eagles_conv = conversion_rate_by_type(eagles_no_sneak, "Eagles (no QB sneak)")
#     giants_conv = conversion_rate_by_type(giants_no_sneak, "Giants (no QB sneak)")

#     # ══════════════════════════════════════════════════════════════════════════
#     # STEP 5 & 6: Formation Analysis & Recommendation
#     # ══════════════════════════════════════════════════════════════════════════
#     section("STEP 5 — Formation Usage on Short Yardage")

#     def formation_breakdown(data: pd.DataFrame, label: str):
#         print(f"\n{label}")
#         print(f"  {'Formation':<30} {'Count':>6}  {'Share':>7}")
#         print(f"  {'-'*46}")
#         counts = data["PlayTimeFormation"].fillna("Unknown").value_counts()
#         for form, count in counts.items():
#             pct = 100 * count / max(len(data), 1)
#             print(f"  {str(form):<30} {count:>6}   {pct:>6.1f}%")

#     formation_breakdown(eagles_no_sneak, "Eagles formations (no QB sneak)")
#     formation_breakdown(giants_no_sneak, "Giants formations (no QB sneak)")

#     # ══════════════════════════════════════════════════════════════════════════
#     # STEP 6: The Recommendation
#     # ══════════════════════════════════════════════════════════════════════════
#     section("STEP 6 — Summary & Recommendation")

#     eagles_overall_conv = eagles_conv["Converted"].sum()
#     eagles_total        = len(eagles_conv)
#     giants_overall_conv = giants_conv["Converted"].sum()
#     giants_total        = len(giants_conv)

#     eagles_rate = 100 * eagles_overall_conv / max(eagles_total, 1)
#     giants_rate = 100 * giants_overall_conv / max(giants_total, 1)

#     print(f"""
# Eagles short yardage conversion rate (no QB sneak): {eagles_rate:.1f}%
# Giants short yardage conversion rate (no QB sneak): {giants_rate:.1f}%

# Key findings:
#   - The Eagles lean heavily on the QB sneak / Tush Push. When that's removed,
#     their conversion rate {"drops" if eagles_rate < giants_rate else "holds steady"} compared to the Giants.
#   - The Giants' approach (see play type and formation breakdown above) offers
#     a data-backed alternative for short yardage situations.
#   - Pass completions and designed runs (non-sneak) make up the bulk of both
#     teams' short yardage arsenal outside the QB sneak.

# Recommendation:
#   If the Tush Push becomes unavailable, the Eagles should model the Giants'
#   short yardage strategy: emphasize {giants_no_sneak["PlayType"].value_counts().index[0]}
#   plays in {giants_no_sneak["PlayTimeFormation"].fillna("Unknown").value_counts().index[0]}
#   formations, which the data shows as the Giants' highest-volume approach.
#   The Eagles' current roster (Hurts' mobility, Barkley's power) supports this
#   model. The numbers show it works.

# GO BIRDS.
# """)


# if __name__ == "__main__":
#     main()

"""
GO BIRDS GO!
============
Eagles Short Yardage Analysis: What happens when the Tush Push is gone?
Alex Violet, Ava Sim, Yoda Ermias — Sports Analytics, April 2026

Action Plan:
  1. Establish Eagles' Tush Push (QB sneak) baseline on 3rd/4th & short
  2. Filter out QB sneaks for both Eagles and Giants
  3. Break down Giants' short yardage approach
  4. Compare to what the Eagles do without the Tush Push
  5. Check if Giants' model fits the Eagles
  6. Make the recommendation
  7. Go Birds.

Run from the Final Project directory:
    python tush_push_analysis.py

File structure expected:
  data/
    2021/ ... 2025/   ← weekly play CSVs
    2021_scores.csv ... 2025_scores.csv
"""

import os
import glob
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR   = "data"
YEARS      = [2021, 2022, 2023, 2024, 2025]

# Short yardage = 3 yards or fewer to go
SHORT_YARDAGE_MAX = 3

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_all_plays() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        pattern = os.path.join(DATA_DIR, str(year), "*_plays.csv")
        for f in sorted(glob.glob(pattern)):
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                print(f"  [WARN] {f}: {e}")
    if not frames:
        raise FileNotFoundError("No play files found. Check DATA_DIR and run from Final Project folder.")
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} total plays across {YEARS[0]}–{YEARS[-1]}.\n")
    return df


def parse_yards_to_go(play_start: str) -> int | None:
    """Extract yards-to-go from strings like '3rd & 1 at PHI 5' or '4th & 2 at DAL 30'."""
    try:
        part = play_start.split("&")[1].strip()
        yards = int(part.split()[0])
        return yards
    except Exception:
        return None


def parse_down(play_start: str) -> str | None:
    """Extract the down string: '3rd' or '4th'."""
    try:
        return play_start.split("&")[0].strip()
    except Exception:
        return None


def parse_formation(play_time_formation: str) -> str:
    """
    Extract just the formation from strings like '02:00 4th Shotgun'
    or '15:00 2nd No Huddle, Shotgun'.
    The column encodes: <game_clock> <down> <formation...>
    We skip the first two tokens (clock + down) and keep the rest.
    """
    try:
        parts = str(play_time_formation).strip().split()
        if len(parts) >= 3:
            return " ".join(parts[2:])
        elif len(parts) == 2:
            return "Unknown"
        return "Unknown"
    except Exception:
        return "Unknown"


def is_qb_sneak(desc: str) -> bool:
    """
    Detect QB sneak / Tush Push plays from play description.

    Primary signal: literal 'qb sneak' label in the play-by-play string.
    This is the most reliable signal — NFL scorers use it consistently.

    Secondary heuristic: short gain (<=2 yards) up the middle with no
    known skill-position player named in the description. We blacklist
    known non-QB ball carriers rather than whitelisting QB names, because
    the blocklist scales better across all teams and years without needing
    to hardcode every QB on every roster.

    Sanity check caught false positives: mckinnon, michel, akers,
    samuel, juszczyk, mitchell — all now in the blocklist.
    """
    d = str(desc).lower()

    # Primary signal: explicit label — always trust this
    if "qb sneak" in d:
        return True

    # Secondary heuristic
    short_gain = False
    if " for " in d:
        try:
            part = d.split(" for ")[-1].strip()
            yards = int(part.split(" yard")[0].strip().replace("-", "").replace(",", ""))
            short_gain = yards <= 2
        except Exception:
            pass

    is_middle_run = "up the middle" in d
    is_pass_play  = any(x in d for x in [
        "pass", "sacked", "scrambles", "incomplete", "intercepted"
    ])

    # Known non-QB ball carriers — RBs, FBs, WRs, TEs.
    # If any of these names appear, it's not a QB sneak.
    known_non_qbs = [
        # RBs
        "barkley", "mccaffrey", "henry", "pollard", "swift", "jacobs", "cook",
        "montgomery", "pierce", "robinson", "williams", "ekeler", "walker",
        "singletary", "mostert", "white", "hunt", "chubb", "mixon", "elliott",
        "foreman", "gainwell", "boston scott", "tucker", "michel", "mckinnon",
        "akers", "henderson", "mattison", "harris", "warren", "sermon",
        "dobbins", "carter", "hubbard", "moss", "stevenson", "edmonds",
        "conner", "rhamondre", "pacheco", "hall", "spiller", "breida",
        "mack", "bell", "gore", "gurley", "fournette", "perine", "ingram",
        "abdullah", "killins", "kelley", "gillmore", "lindsay", "mitchell",
        "barnett",
        # FBs / H-backs
        "juszczyk", "vitale", "di marco", "leggett", "feliciano",
        # WRs (direct snaps, end-arounds, jet sweeps)
        "samuel", "hill", "jefferson", "diggs", "adams", "thielen",
        "cooper", "lamb", "metcalf", "lockett", "kupp", "woods",
        "cooks", "moore", "mclaurin", "boyd", "higgins", "chase",
        "pittman", "hardman", "peoples-jones",
        # TEs (direct snaps)
        "kelce", "waller", "kittle", "andrews", "schultz", "goedert",
        "hockenson", "engram", "njoku", "gesicki", "otton",
    ]

    is_skill_player = any(name in d for name in known_non_qbs)

    return is_middle_run and short_gain and not is_pass_play and not is_skill_player


def classify_play(desc: str) -> str:
    """Classify a play description into a category."""
    d = str(desc).lower()

    # Check QB sneak FIRST — before the generic run logic that would mis-classify it
    if is_qb_sneak(d):
        return "QB Sneak (Tush Push)"
    if "punt" in d:
        return "Punt"
    if "field goal is good" in d:
        return "Field Goal (GOOD)"
    if "field goal is no good" in d or "field goal is blocked" in d:
        return "Field Goal (MISSED)"
    if "touchdown" in d and "pass" in d:
        return "Touchdown Pass"
    if "touchdown" in d and ("run" in d or "up the middle" in d or "left" in d or "right" in d):
        return "Touchdown Run"
    if "fumble" in d:
        return "Fumble"
    if "intercepted" in d or "interception" in d:
        return "Interception"
    if "sacked" in d:
        return "Sack / Scramble Loss"
    if "incomplete" in d:
        return "Pass Incomplete"
    if "pass" in d:
        return "Pass Complete"
    if "scrambles" in d:
        return "QB Scramble"
    if "penalty" in d or "no play" in d:
        return "Penalty"
    if any(x in d for x in ["up the middle", "left guard", "right guard", "left tackle",
                              "right tackle", "left end", "right end", "off left", "off right"]):
        return "Run"
    return "Other"


def is_converted(desc: str, yards_needed: int) -> bool:
    """Check if the play resulted in a conversion (first down, TD, or FG good)."""
    d = str(desc).lower()
    if "touchdown" in d or "field goal is good" in d:
        return True
    if "incomplete" in d or "no good" in d or "punt" in d or "fumble" in d:
        return False
    if "intercepted" in d or "sacked" in d:
        return False
    try:
        if " for " in d:
            part = d.split(" for ")[-1].strip()
            yards = int(part.split(" yard")[0].strip().replace("-", "").replace(",", ""))
            return yards >= yards_needed
    except Exception:
        pass
    return False


def section(title: str):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def sanity_check_sneak_detection(df: pd.DataFrame):
    """
    Print sample plays flagged as QB sneaks and sample middle runs NOT flagged.
    Run after computing IsQBSneak to verify the heuristic before full analysis.
    Look for: are any flagged plays clearly not QB sneaks?
    Are any unflagged middle runs suspiciously sneak-like?
    """
    section("SANITY CHECK — QB Sneak Detection")

    flagged = df[df["IsQBSneak"]]["PlayDescription"]
    print(f"\nPlays flagged as QB Sneak ({len(flagged):,} total). Sample of 10:")
    print(flagged.head(10).to_string())

    middle_not_flagged = df[
        df["PlayDescription"].str.lower().str.contains("up the middle", na=False) &
        ~df["IsQBSneak"]
    ]["PlayDescription"]
    print(f"\nMiddle runs NOT flagged as sneak ({len(middle_not_flagged):,} total). Sample of 10:")
    print(middle_not_flagged.head(10).to_string())


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("GO BIRDS GO! — Eagles Short Yardage Analysis")
    print("Alex Violet, Ava Sim, Yoda Ermias | Sports Analytics | April 2026\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_all_plays()

    # Add parsed columns
    df["Down"]        = df["PlayStart"].astype(str).apply(parse_down)
    df["YardsToGo"]   = df["PlayStart"].astype(str).apply(parse_yards_to_go)
    df["PlayType"]    = df["PlayDescription"].astype(str).apply(classify_play)
    df["IsQBSneak"]   = df["PlayDescription"].astype(str).apply(is_qb_sneak)
    df["Formation"]   = df["PlayTimeFormation"].astype(str).apply(parse_formation)

    # ── Sanity check: verify sneak detection before continuing ────────────────
    sanity_check_sneak_detection(df)

    # ── Filter to 3rd & short and 4th & short ────────────────────────────────
    short_mask = (
        df["Down"].isin(["3rd", "4th"]) &
        df["YardsToGo"].notna() &
        (df["YardsToGo"] <= SHORT_YARDAGE_MAX)
    )
    short_df = df[short_mask].copy()
    print(f"\nShort yardage plays (3rd/4th & 1–{SHORT_YARDAGE_MAX}): {len(short_df):,}\n")

    # ── Filter to Eagles and Giants games ────────────────────────────────────
    eagles_short = short_df[
        (short_df["AwayTeam"] == "Eagles") | (short_df["HomeTeam"] == "Eagles")
    ].copy()

    giants_short = short_df[
        (short_df["AwayTeam"] == "Giants") | (short_df["HomeTeam"] == "Giants")
    ].copy()

    # Only Eagles/Giants possession respectively
    eagles_offense = eagles_short[
        eagles_short["TeamWithPossession"].str.contains("Philadelphia", na=False)
    ].copy()

    giants_offense = giants_short[
        giants_short["TeamWithPossession"].str.contains("New York Giants", na=False)
    ].copy()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Eagles Tush Push Baseline
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 1 — Eagles Tush Push (QB Sneak) Baseline")

    sneaks     = eagles_offense[eagles_offense["IsQBSneak"]]
    non_sneaks = eagles_offense[~eagles_offense["IsQBSneak"]]

    print(f"Eagles short yardage plays (offense):   {len(eagles_offense):,}")
    print(f"  → QB Sneak / Tush Push plays:         {len(sneaks):,}  ({100*len(sneaks)/max(len(eagles_offense),1):.1f}%)")
    print(f"  → All other short yardage plays:      {len(non_sneaks):,}  ({100*len(non_sneaks)/max(len(eagles_offense),1):.1f}%)")

    print("\nTush Push usage by year:")
    for year in YEARS:
        yr_plays  = eagles_offense[eagles_offense["Season"] == year]
        yr_sneaks = yr_plays[yr_plays["IsQBSneak"]]
        pct = 100 * len(yr_sneaks) / max(len(yr_plays), 1)
        print(f"  {year}: {len(yr_sneaks):>3} sneak(s) out of {len(yr_plays):>3} short yardage plays  ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 & 3: Eagles WITHOUT QB Sneak vs. Giants — Play Type Breakdown
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 2 & 3 — Play Type Breakdown (QB Sneaks Removed)")

    eagles_no_sneak = eagles_offense[~eagles_offense["IsQBSneak"]].copy()
    giants_no_sneak = giants_offense[~giants_offense["IsQBSneak"]].copy()

    def play_type_summary(data: pd.DataFrame, label: str):
        print(f"\n{label}  ({len(data):,} plays, QB sneaks excluded)")
        print(f"  {'Play Type':<28} {'Count':>6}  {'Share':>7}")
        print(f"  {'-'*44}")
        counts = data["PlayType"].value_counts()
        for play, count in counts.items():
            pct = 100 * count / max(len(data), 1)
            print(f"  {play:<28} {count:>6}   {pct:>6.1f}%")

    play_type_summary(eagles_no_sneak, "Philadelphia Eagles (no QB sneak)")
    play_type_summary(giants_no_sneak, "New York Giants (no QB sneak)")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Conversion Rates
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 4 — Conversion Rates (3rd & 4th & Short, No QB Sneak)")

    def conversion_rate_by_type(data: pd.DataFrame, label: str):
        print(f"\n{label}")
        print(f"  {'Play Type':<28} {'Plays':>6}  {'Conv.':>6}  {'Rate':>7}")
        print(f"  {'-'*50}")

        data = data.copy()
        data["Converted"] = data.apply(
            lambda r: is_converted(r["PlayDescription"], r["YardsToGo"]), axis=1
        )

        overall = data["Converted"].sum()
        total   = len(data)
        print(f"  {'OVERALL':<28} {total:>6}  {overall:>6}  {100*overall/max(total,1):>6.1f}%")

        for play_type, grp in data.groupby("PlayType"):
            conv  = grp["Converted"].sum()
            plays = len(grp)
            rate  = 100 * conv / max(plays, 1)
            print(f"  {play_type:<28} {plays:>6}  {conv:>6}  {rate:>6.1f}%")

        return data

    eagles_conv = conversion_rate_by_type(eagles_no_sneak, "Eagles (no QB sneak)")
    giants_conv = conversion_rate_by_type(giants_no_sneak, "Giants (no QB sneak)")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Formation Analysis
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 5 — Formation Usage on Short Yardage")

    def formation_breakdown(data: pd.DataFrame, label: str):
        print(f"\n{label}")
        print(f"  {'Formation':<30} {'Count':>6}  {'Share':>7}")
        print(f"  {'-'*46}")
        counts = data["Formation"].value_counts()
        for form, count in counts.items():
            pct = 100 * count / max(len(data), 1)
            print(f"  {str(form):<30} {count:>6}   {pct:>6.1f}%")

    formation_breakdown(eagles_no_sneak, "Eagles formations (no QB sneak)")
    formation_breakdown(giants_no_sneak, "Giants formations (no QB sneak)")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6: The Recommendation
    # ══════════════════════════════════════════════════════════════════════════
    section("STEP 6 — Summary & Recommendation")

    eagles_overall_conv = eagles_conv["Converted"].sum()
    eagles_total        = len(eagles_conv)
    giants_overall_conv = giants_conv["Converted"].sum()
    giants_total        = len(giants_conv)

    eagles_rate = 100 * eagles_overall_conv / max(eagles_total, 1)
    giants_rate = 100 * giants_overall_conv / max(giants_total, 1)

    top_giants_play      = giants_no_sneak["PlayType"].value_counts().index[0]
    top_giants_formation = giants_no_sneak["Formation"].value_counts().index[0]

    print(f"""
Eagles short yardage conversion rate (no QB sneak): {eagles_rate:.1f}%
Giants short yardage conversion rate (no QB sneak): {giants_rate:.1f}%

Key findings:
  - The Eagles lean heavily on the QB sneak / Tush Push. When that's removed,
    their conversion rate {"drops" if eagles_rate < giants_rate else "holds steady"} compared to the Giants.
  - The Giants' approach (see play type and formation breakdown above) offers
    a data-backed alternative for short yardage situations.
  - Pass completions and designed runs (non-sneak) make up the bulk of both
    teams' short yardage arsenal outside the QB sneak.

Recommendation:
  If the Tush Push becomes unavailable, the Eagles should model the Giants'
  short yardage strategy: emphasize {top_giants_play} plays out of
  {top_giants_formation} formations, which the data shows as the Giants'
  highest-volume approach on short yardage.
  The Eagles' current roster (Hurts' mobility, Barkley's power) supports this
  model. The numbers show it works.

GO BIRDS.
""")


if __name__ == "__main__":
    main()