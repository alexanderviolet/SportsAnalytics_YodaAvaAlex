"""
diagnostics.py  —  4-part shot detection investigation
Run from the same directory as 0021500495.json and 0021500495.csv
"""

import csv, json, math, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & PARAMETERS  (keep in sync with playground.py)
# ─────────────────────────────────────────────────────────────────────────────
RIM_HEIGHT        = 10
RIM_DIAMETER      = 1.5
BASKETBALL_DIAMETER = 1
BASKET_LEFT       = (5.25,  25, 10)
BASKET_RIGHT      = (88.75, 25, 10)

SHOT_THRESHOLD    = RIM_HEIGHT - 2 * BASKETBALL_DIAMETER   # 8
FRAMES_AFTER_PEAK = 64
DUPLICATE_WINDOW  = 2
H_TOLERANCE       = 2.5 * RIM_DIAMETER                # 3.75
V_ABOVE           = 2   * RIM_DIAMETER                # 3.0
V_BELOW           = 0.5 * BASKETBALL_DIAMETER         # 0.5

MATCH_WINDOW      = 3.0   # seconds — same as playground.py
PROXIMITY_WINDOW  = 2.0   # seconds either side for investigation #3

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data …")
with open('0021500495.json') as f:
    sportvu = json.load(f)

all_moments = []
for event in sportvu['events']:
    for moment in event['moments']:
        q  = moment[0]
        gc = moment[2]
        b  = moment[5][0]
        t  = (q - 1) * 720 + (720 - gc)
        all_moments.append({'time': t, 'quarter': q, 'clock': gc,
                             'x': b[2], 'y': b[3], 'z': b[4]})
all_moments.sort(key=lambda m: m['time'])
times_arr = np.array([m['time'] for m in all_moments])

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def passed_near_basket(window):
    for m in window:
        for basket in [BASKET_LEFT, BASKET_RIGHT]:
            h = math.sqrt((m['x']-basket[0])**2 + (m['y']-basket[1])**2)
            v = m['z'] - basket[2]
            if h <= H_TOLERANCE and -V_BELOW <= v <= V_ABOVE:
                return True
    return False

def moments_in_window(t_center, half=2.0):
    lo, hi = t_center - half, t_center + half
    return [all_moments[i] for i in np.where(
        (times_arr >= lo) & (times_arr <= hi))[0]]

def dist_to_nearest_basket(m):
    dl = math.sqrt((m['x']-BASKET_LEFT[0])**2  + (m['y']-BASKET_LEFT[1])**2)
    dr = math.sqrt((m['x']-BASKET_RIGHT[0])**2 + (m['y']-BASKET_RIGHT[1])**2)
    return min(dl, dr)

def shot_label(desc):
    """Return a short category string for a shot description."""
    d = desc.lower()
    m = re.search(r"(\d+)'", desc)
    dist = int(m.group(1)) if m else 0
    if dist >= 30:            return 'heave (≥30 ft)'
    if 'tip' in d or 'putback' in d: return 'putback/tip'
    if 'dunk' in d:           return 'dunk'
    if 'layup' in d:          return 'layup'
    if '3pt' in d or '3-pt' in d: return '3-pointer'
    return 'mid-range'

# ─────────────────────────────────────────────────────────────────────────────
# INSTRUMENTED DETECTION LOOP
# Records peak z and failure reason for every detected + missed shot
# ─────────────────────────────────────────────────────────────────────────────
print("Running instrumented detection loop …")

shots_detected = []          # list of peak moments
all_arc_peaks  = []          # every peak (detected or not), with metadata

in_shot = False
current_shot_peak = None
arc_passed_basket = False

for i in range(1, len(all_moments) - 1):
    prev, curr, nxt = all_moments[i-1], all_moments[i], all_moments[i+1]

    if not in_shot:
        if (curr['z'] > SHOT_THRESHOLD and
                curr['z'] >= prev['z'] and curr['z'] >= nxt['z']):
            in_shot = True
            current_shot_peak = curr
            arc_passed_basket = False
    else:
        if curr['z'] > current_shot_peak['z']:
            current_shot_peak = curr

        if curr['z'] < RIM_HEIGHT:
            window = all_moments[i: i + FRAMES_AFTER_PEAK]
            near   = passed_near_basket(window)
            is_dup = (shots_detected and
                      current_shot_peak['time'] - shots_detected[-1]['time']
                      <= DUPLICATE_WINDOW)

            all_arc_peaks.append({
                'peak':      current_shot_peak,
                'detected':  near and not is_dup,
                'near_basket': near,
                'duplicate': is_dup,
            })

            if near and not is_dup:
                shots_detected.append(current_shot_peak)

            in_shot = False
            current_shot_peak = None

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CSV  &  CLASSIFY EVERY MISSED SHOT  (investigation #4)
# ─────────────────────────────────────────────────────────────────────────────
csv_shots = []
with open('0021500495.csv') as f:
    for row in csv.DictReader(f):
        if row['EVENTMSGTYPE'] in ['1', '2']:
            p   = int(row['PERIOD'])
            mm, ss = row['PCTIMESTRING'].split(':')
            clk = int(mm)*60 + int(ss)
            t   = (p-1)*720 + (720-clk)
            csv_shots.append({
                'time':  t, 'period': p,
                'clock': row['PCTIMESTRING'],
                'made':  row['EVENTMSGTYPE'] == '1',
                'desc':  (row['HOMEDESCRIPTION'] or
                          row['VISITORDESCRIPTION']).strip()
            })

# Classify each CSV shot
failure_buckets = defaultdict(list)   # bucket_name → [shot]
matched_shots   = []

for shot in csv_shots:
    nearby_detected = [s for s in shots_detected
                       if abs(s['time'] - shot['time']) <= MATCH_WINDOW]
    if nearby_detected:
        matched_shots.append(shot)
        continue

    # ── Not matched — figure out WHY ─────────────────────────────────────
    seg = moments_in_window(shot['time'], half=MATCH_WINDOW)

    # Did the ball ever trigger arc detection?
    arc_triggered = any(
        p['peak']['time'] >= shot['time'] - MATCH_WINDOW and
        p['peak']['time'] <= shot['time'] + MATCH_WINDOW
        for p in all_arc_peaks
    )

    # Did any arc in that window pass the basket check?
    arc_near = any(
        p['near_basket'] and
        abs(p['peak']['time'] - shot['time']) <= MATCH_WINDOW
        for p in all_arc_peaks
    )

    # Was it swallowed by the duplicate filter?
    arc_dup = any(
        p['duplicate'] and
        abs(p['peak']['time'] - shot['time']) <= MATCH_WINDOW
        for p in all_arc_peaks
    )

    # What was the max z near this shot?
    max_z = max((m['z'] for m in seg), default=0)

    if max_z <= SHOT_THRESHOLD:
        bucket = 'never_above_threshold'
    elif not arc_triggered:
        bucket = 'arc_not_triggered'
    elif arc_dup:
        bucket = 'duplicate_filter'
    elif not arc_near:
        bucket = 'arc_triggered_but_not_near_basket'
    else:
        bucket = 'other'

    failure_buckets[bucket].append({**shot, 'max_z': max_z})

# ─────────────────────────────────────────────────────────────────────────────
# PRINT INVESTIGATION #1 — Coordinates around each missed shot
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("INVESTIGATION 1 — Ball coordinates around each missed shot")
print("═"*70)

all_missed = [s for bucket in failure_buckets.values() for s in bucket]
for shot in sorted(all_missed, key=lambda s: s['time']):
    seg = moments_in_window(shot['time'], half=2.0)
    if not seg:
        continue
    zs   = [m['z'] for m in seg]
    xs   = [m['x'] for m in seg]
    ys   = [m['y'] for m in seg]
    print(f"\n  ✗ Q{shot['period']} {shot['clock']}  {shot['desc']}")
    print(f"    max_z={max(zs):.2f}ft  "
          f"z_range=[{min(zs):.1f}, {max(zs):.1f}]  "
          f"x_range=[{min(xs):.1f}, {max(xs):.1f}]  "
          f"y_range=[{min(ys):.1f}, {max(ys):.1f}]")

# ─────────────────────────────────────────────────────────────────────────────
# INVESTIGATION #4 — Failure reason per missed shot (print summary)
# ─────────────────────────────────────────────────────────────────────────────
BUCKET_LABELS = {
    'never_above_threshold':             'Never above threshold (putbacks/heaves)',
    'arc_not_triggered':                 'Arc not triggered (unusual trajectory)',
    'duplicate_filter':                  'Caught by duplicate filter',
    'arc_triggered_but_not_near_basket': 'Arc triggered but failed basket check',
    'other':                             'Other / unclear',
}

print("\n" + "═"*70)
print("INVESTIGATION 4 — Why each missed shot failed (failure buckets)")
print("═"*70)
total_missed = sum(len(v) for v in failure_buckets.values())
for key, shots in sorted(failure_buckets.items(),
                         key=lambda kv: -len(kv[1])):
    label = BUCKET_LABELS.get(key, key)
    pct   = 100 * len(shots) / total_missed if total_missed else 0
    print(f"\n  [{len(shots):2d} shots | {pct:.0f}%]  {label}")
    for s in shots:
        print(f"    Q{s['period']} {s['clock']}  max_z={s['max_z']:.1f}ft  {s['desc']}")

# ─────────────────────────────────────────────────────────────────────────────
# INVESTIGATION #3 — How many missed shots are recoverable via proximity check
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("INVESTIGATION 3 — Proximity-only recovery (no arc required)")
print("═"*70)
recoverable = []
for shot in all_missed:
    seg = moments_in_window(shot['time'], half=PROXIMITY_WINDOW)
    if any(m['z'] >= RIM_HEIGHT and
           dist_to_nearest_basket(m) <= H_TOLERANCE
           for m in seg):
        recoverable.append(shot)

print(f"\n  {len(recoverable)} / {total_missed} missed shots would be caught")
print("  by a proximity-only detector (ball within H_TOLERANCE of basket")
print(f"  and at or above rim height within ±{PROXIMITY_WINDOW}s of shot time)\n")
for s in recoverable:
    print(f"    ✓ recoverable  Q{s['period']} {s['clock']}  {s['desc']}")

not_recoverable = [s for s in all_missed if s not in recoverable]
print(f"\n  {len(not_recoverable)} shots NOT recoverable even with proximity check:")
for s in not_recoverable:
    print(f"    ✗ Q{s['period']} {s['clock']}  max_z={s['max_z']:.1f}ft  {s['desc']}")

# ─────────────────────────────────────────────────────────────────────────────
# INVESTIGATION #2 — Peak-z histogram  +  failure bucket pie  +  shot categories
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("INVESTIGATION 2 — Generating plots …")
print("═"*70)

detected_peaks = [p['peak']['z'] for p in all_arc_peaks if p['detected']]
missed_peaks   = [p['peak']['z'] for p in all_arc_peaks if not p['detected']]

# Also gather peak-z for CSV-matched shots (best effort)
csv_matched_peaks = []
for shot in matched_shots:
    nearby = [p for p in all_arc_peaks
              if abs(p['peak']['time'] - shot['time']) <= MATCH_WINDOW
              and p['detected']]
    if nearby:
        best = min(nearby, key=lambda p: abs(p['peak']['time'] - shot['time']))
        csv_matched_peaks.append(best['peak']['z'])

fig = plt.figure(figsize=(16, 12), facecolor='#0e1118')
fig.suptitle('Shot Detection Diagnostics', fontsize=17, fontweight='bold',
             color='white', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.08, right=0.96, top=0.93, bottom=0.07)

DARK   = '#0e1118'
PANEL  = '#1a1d2e'
BLUE   = '#4A90D9'
ORANGE = '#E8834A'
GREEN  = '#52C97A'
RED    = '#E05C5C'
PURPLE = '#9B6DFF'
WHITE  = '#e8eaf0'
GREY   = '#6b7080'

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color('#2a2d3e')
    ax.tick_params(colors=GREY, labelsize=8)
    ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=8)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)

# ── Plot 1: Peak-z histogram ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'Investigation 2 — Peak Z-Height of All Arcs')

bins = np.arange(8, 22, 0.5)
ax1.hist(detected_peaks, bins=bins, color=BLUE,   alpha=0.75,
         label=f'detected ({len(detected_peaks)})', zorder=3)
ax1.hist(missed_peaks,   bins=bins, color=ORANGE, alpha=0.75,
         label=f'arc not detected ({len(missed_peaks)})', zorder=2)

ax1.axvline(SHOT_THRESHOLD, color='#ffcc00', lw=1.2, ls='--',
            label=f'threshold ({SHOT_THRESHOLD} ft)')
ax1.axvline(RIM_HEIGHT,     color=RED,       lw=1.2, ls=':',
            label=f'rim ({RIM_HEIGHT} ft)')
ax1.set_xlabel('peak z (ft)')
ax1.set_ylabel('count')
ax1.legend(fontsize=7, facecolor=PANEL, edgecolor='#2a2d3e',
           labelcolor=WHITE, framealpha=0.9)

# ── Plot 2: Failure bucket pie ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'Investigation 4 — Why Shots Were Missed')

bucket_sizes  = [len(v) for v in failure_buckets.values()]
bucket_names  = [BUCKET_LABELS.get(k, k) for k in failure_buckets]
pie_colors    = [ORANGE, PURPLE, GREEN, RED, BLUE][:len(bucket_sizes)]

if bucket_sizes:
    wedges, texts, autotexts = ax2.pie(
        bucket_sizes, labels=None,
        colors=pie_colors, autopct='%1.0f%%',
        startangle=140, pctdistance=0.75,
        wedgeprops={'edgecolor': DARK, 'linewidth': 1.5}
    )
    for at in autotexts:
        at.set_color(WHITE); at.set_fontsize(8)
    ax2.legend(wedges, [f'{n}\n({s})' for n, s in
                        zip(bucket_names, bucket_sizes)],
               loc='lower center', bbox_to_anchor=(0.5, -0.28),
               fontsize=7, facecolor=PANEL, edgecolor='#2a2d3e',
               labelcolor=WHITE, framealpha=0.9, ncol=1)

# ── Plot 3: Recovery bar (investigation #3) ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, 'Investigation 3 — Proximity Recovery Potential')

categories_ordered = ['Matched\n(current)', 'Proximity\nRecoverable',
                      'Not\nRecoverable']
values  = [len(matched_shots), len(recoverable), len(not_recoverable)]
colors  = [GREEN, BLUE, RED]
bars    = ax3.bar(categories_ordered, values, color=colors,
                  edgecolor=DARK, linewidth=1.2, width=0.5)

for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             str(val), ha='center', va='bottom', color=WHITE, fontsize=10,
             fontweight='bold')

ax3.set_ylabel('number of shots')
ax3.set_ylim(0, max(values) * 1.18)
total_csv = len(csv_shots)
ax3.set_title(
    f'Investigation 3 — Proximity Recovery  (total CSV shots: {total_csv})',
    color=WHITE, fontsize=10, fontweight='bold', pad=8)

# ── Plot 4: Missed shots by category ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, 'Investigation 4 — Missed Shots by Shot Type')

cat_counts = defaultdict(int)
for s in all_missed:
    cat_counts[shot_label(s['desc'])] += 1
cats   = sorted(cat_counts, key=lambda c: -cat_counts[c])
counts = [cat_counts[c] for c in cats]
palette = [ORANGE, PURPLE, BLUE, GREEN, RED, '#FFD166'][:len(cats)]

hbars = ax4.barh(cats[::-1], counts[::-1], color=palette[::-1],
                 edgecolor=DARK, linewidth=1.0, height=0.55)
for bar, val in zip(hbars, counts[::-1]):
    ax4.text(val + 0.15, bar.get_y() + bar.get_height()/2,
             str(val), va='center', color=WHITE, fontsize=9)
ax4.set_xlabel('missed shots not detected')
ax4.set_xlim(0, max(counts) * 1.25)

plt.savefig('diagnostics.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("\nPlot saved → diagnostics.png")
plt.show()

print("\n" + "═"*70)
print(f"FINAL SUMMARY")
print("═"*70)
print(f"  CSV total:          {len(csv_shots)}")
print(f"  Matched:            {len(matched_shots)}")
print(f"  Missed:             {total_missed}")
print(f"  False positives:    {len(shots_detected) - len(matched_shots)}")
print(f"  Proximity-recoverable missed: {len(recoverable)} / {total_missed}")