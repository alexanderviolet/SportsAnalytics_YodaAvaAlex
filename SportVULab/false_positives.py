import csv
import json
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import math

with open('0021500495.json') as f:
    sportvu = json.load(f)

# ── Constants ────────────────────────────────────────────────────────────────
BASKET_LEFT  = (5.25,  25, 10)
BASKET_RIGHT = (88.75, 25, 10)

SHOT_THRESHOLD    = 8.5    # min peak z for high-arc shots
DROP_THRESHOLD    = 8.0    # close window when ball falls below this (raised from 6.5)
                           # — matches starter code's "< 8" and keeps window open longer
BASKET_RADIUS     = 5.0    # 3D proximity to basket after peak
WINDOW_SIZE       = 50     # frames to look ahead for basket proximity (was 32, starter used 50)
MAX_SHOT_DISTANCE = 42.0   # no half-court heaves
LOOKBACK_PASS     = 20
PASS_TRAVEL_MIN   = 28.0

# Backboard lob: only reject peaks that are very high AND directly over the rim
BACKBOARD_EXCL_RADIUS = 7.0
BACKBOARD_EXCL_MIN_Z  = 12.0

# Hook shot: low arc close-range shots (layups, finger rolls, hooks)
HOOK_MIN_Z    = 7.0    # slightly lower than before to catch more real hooks
HOOK_MAX_DIST = 12.0   # back to original — hooks can happen from 12ft

DEDUP_GAP    = 3.0
MATCH_WINDOW = 5.5

# ── Helpers ──────────────────────────────────────────────────────────────────

def dist_to_nearest_basket_2d(x, y):
    d_left  = math.sqrt((x - BASKET_LEFT[0])**2  + (y - BASKET_LEFT[1])**2)
    d_right = math.sqrt((x - BASKET_RIGHT[0])**2 + (y - BASKET_RIGHT[1])**2)
    return min(d_left, d_right)

def dist_to_nearest_basket_3d(x, y, z):
    d_left  = math.sqrt((x - BASKET_LEFT[0])**2  + (y - BASKET_LEFT[1])**2  + (z - BASKET_LEFT[2])**2)
    d_right = math.sqrt((x - BASKET_RIGHT[0])**2 + (y - BASKET_RIGHT[1])**2 + (z - BASKET_RIGHT[2])**2)
    return min(d_left, d_right)

def is_valid_shot_origin(x, y):
    return dist_to_nearest_basket_2d(x, y) <= MAX_SHOT_DISTANCE

def is_backboard_lob(x, y, z):
    """
    Inbound/outlet lobs peak very high directly above the rim.
    Real close-range shots (putbacks, tips, hooks) peak below 12ft.
    """
    return dist_to_nearest_basket_2d(x, y) <= BACKBOARD_EXCL_RADIUS and z >= BACKBOARD_EXCL_MIN_Z

def is_long_pass(moments, peak_idx):
    """Ball traveled far horizontally before peaking = outlet pass."""
    start_idx = max(0, peak_idx - LOOKBACK_PASS)
    travel = math.sqrt(
        (moments[peak_idx]['x'] - moments[start_idx]['x'])**2 +
        (moments[peak_idx]['y'] - moments[start_idx]['y'])**2
    )
    return travel >= PASS_TRAVEL_MIN

def moving_toward_basket(moments, i, lookback=5):
    if i < lookback:
        return False
    curr = moments[i]
    prev = moments[i - lookback]
    return dist_to_nearest_basket_2d(curr['x'], curr['y']) < dist_to_nearest_basket_2d(prev['x'], prev['y'])

def passed_near_basket(moments_window):
    """
    Did the ball come within BASKET_RADIUS (3D) of either basket?
    Uses a larger window (WINDOW_SIZE=50) to give missed shots time
    to reach the rim even if the ball bounces away quickly.
    """
    return any(
        dist_to_nearest_basket_3d(m['x'], m['y'], m['z']) <= BASKET_RADIUS
        for m in moments_window
    )

def deduplicate_shots(shots, min_gap=DEDUP_GAP):
    if not shots:
        return shots
    shots.sort(key=lambda s: s['time'])
    deduped = [shots[0]]
    for shot in shots[1:]:
        if shot['time'] - deduped[-1]['time'] > min_gap:
            deduped.append(shot)
    return deduped

def shot_distance_scaled(moment):
    return min(10, dist_to_nearest_basket_2d(moment['x'], moment['y']) * (10 / 30))

def fmt_clock(t):
    q = int(t // 720) + 1
    remaining = 720 - (t % 720)
    return f"Q{q} {int(remaining//60)}:{int(remaining%60):02d}"

# ── Build moments ────────────────────────────────────────────────────────────
all_moments = []
for event in sportvu['events']:
    for moment in event['moments']:
        quarter    = moment[0]
        game_clock = moment[2]
        ball       = moment[5][0]
        t = (quarter - 1) * 720 + (720 - game_clock)
        all_moments.append({
            'time': t, 'quarter': quarter, 'clock': game_clock,
            'x': ball[2], 'y': ball[3], 'z': ball[4]
        })
all_moments.sort(key=lambda m: m['time'])

# ── Detection loop ────────────────────────────────────────────────────────────
shots_detected        = []
in_shot               = False
current_shot_peak     = None
current_shot_peak_idx = None
current_shot_type     = None

for i in range(1, len(all_moments) - 1):
    prev   = all_moments[i - 1]
    curr   = all_moments[i]
    next_m = all_moments[i + 1]

    if not in_shot:
        if not is_valid_shot_origin(curr['x'], curr['y']):
            continue

        # Standard jump shot — peaks above threshold anywhere in valid zone
        is_high_shot = (
            curr['z'] > SHOT_THRESHOLD and
            curr['z'] >= prev['z'] and
            curr['z'] >= next_m['z']
        )

        # Hook/layup/finger roll — low arc, close to basket, moving toward it
        is_hook_shot = (
            curr['z'] > HOOK_MIN_Z and
            curr['z'] >= prev['z'] and
            curr['z'] >= next_m['z'] and
            dist_to_nearest_basket_2d(curr['x'], curr['y']) <= HOOK_MAX_DIST and
            moving_toward_basket(all_moments, i)
        )

        if is_high_shot or is_hook_shot:
            in_shot               = True
            current_shot_peak     = curr
            current_shot_peak_idx = i
            current_shot_type     = 'high' if is_high_shot else 'hook'

    else:
        if curr['z'] > current_shot_peak['z']:
            current_shot_peak     = curr
            current_shot_peak_idx = i

        # Close the window when ball drops below DROP_THRESHOLD
        if curr['z'] < DROP_THRESHOLD:
            window_after = all_moments[i:i + WINDOW_SIZE]

            # Filter 1: long outlet/inbound pass
            if is_long_pass(all_moments, current_shot_peak_idx):
                in_shot = False; current_shot_peak = None
                current_shot_peak_idx = None; current_shot_type = None
                continue

            # Filter 2: inbound lob arcing high directly over the rim
            if is_backboard_lob(current_shot_peak['x'], current_shot_peak['y'], current_shot_peak['z']):
                in_shot = False; current_shot_peak = None
                current_shot_peak_idx = None; current_shot_type = None
                continue

            # Accept if ball passes near a basket in the window after the peak
            if passed_near_basket(window_after):
                peak = dict(current_shot_peak)
                peak['type'] = current_shot_type
                shots_detected.append(peak)

            in_shot = False; current_shot_peak = None
            current_shot_peak_idx = None; current_shot_type = None

shots_detected = deduplicate_shots(shots_detected)

# ── Load CSV ──────────────────────────────────────────────────────────────────
csv_shots = []
with open('0021500495.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['EVENTMSGTYPE'] in ['1', '2']:
            period = int(row['PERIOD'])
            tp = row['PCTIMESTRING'].split(':')
            t  = (period - 1) * 720 + (720 - int(tp[0]) * 60 - int(tp[1]))
            csv_shots.append({
                'time': t, 'period': period, 'clock': row['PCTIMESTRING'],
                'made': row['EVENTMSGTYPE'] == '1',
                'desc': row['HOMEDESCRIPTION'] or row['VISITORDESCRIPTION']
            })

# ── Match ─────────────────────────────────────────────────────────────────────
matched_json_times = set()
match_results = []

for csv_shot in csv_shots:
    close = [s for s in shots_detected if abs(s['time'] - csv_shot['time']) <= MATCH_WINDOW]
    if close:
        best = min(close, key=lambda s: abs(s['time'] - csv_shot['time']))
        matched_json_times.add(best['time'])
        match_results.append({
            'status': 'MATCH', 'period': csv_shot['period'], 'clock': csv_shot['clock'],
            'diff': best['time'] - csv_shot['time'], 'time': csv_shot['time'],
            'made': csv_shot['made'], 'desc': csv_shot['desc'], 'json_time': best['time']
        })
    else:
        nearest = min(shots_detected, key=lambda s: abs(s['time'] - csv_shot['time']))
        match_results.append({
            'status': 'MISSED', 'period': csv_shot['period'], 'clock': csv_shot['clock'],
            'diff': nearest['time'] - csv_shot['time'], 'time': csv_shot['time'],
            'made': csv_shot['made'], 'desc': csv_shot['desc'], 'json_time': None
        })

false_positives = [s for s in shots_detected if s['time'] not in matched_json_times]
true_positives  = [s for s in shots_detected if s['time'] in matched_json_times]
missed_csv      = [r for r in match_results if r['status'] == 'MISSED']
fp_high = [s for s in false_positives if s['type'] == 'high']
fp_hook = [s for s in false_positives if s['type'] == 'hook']

recall    = len(true_positives) / len(csv_shots) * 100 if csv_shots else 0
precision = len(true_positives) / len(shots_detected) * 100 if shots_detected else 0

# ── Console output ────────────────────────────────────────────────────────────
print("=" * 70)
print("  MATCHING CSV SHOTS TO JSON DETECTIONS")
print("=" * 70)
for r in match_results:
    icon = "✓" if r['status'] == 'MATCH' else "✗"
    made = "MADE  " if r['made'] else "MISSED"
    print(f"  {icon} {r['status']:6s} | Q{r['period']} {r['clock']} | {made} | diff={r['diff']:+.1f}s | {r['desc']}")

print()
print("=" * 70)
print("  FALSE POSITIVES (detected by JSON, not in CSV)")
print("=" * 70)
for s in sorted(false_positives, key=lambda x: x['time']):
    print(f"  ✗ FP | {fmt_clock(s['time'])} | t={s['time']:.1f}s | "
          f"z={s['z']:.2f}ft | x={s['x']:.1f} y={s['y']:.1f} | {s['type']}")

print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  CSV total shots:          {len(csv_shots)}")
print(f"  JSON detections:          {len(shots_detected)}")
print(f"  Matched (true positives): {len(true_positives)}")
print(f"  Missed by JSON:           {len(missed_csv)}")
print(f"  False positives:          {len(false_positives)}")
print(f"    └─ High arc:            {len(fp_high)}")
print(f"    └─ Hook/low arc:        {len(fp_hook)}")
print(f"  Recall:                   {recall:.1f}%")
print(f"  Precision:                {precision:.1f}%")
print("=" * 70)

# ── Required Shot Timeline ─────────────────────────────────────────────────────
shot_times = np.array([s['time'] for s in shots_detected])
shot_facts = np.array([shot_distance_scaled(s) for s in shots_detected])

fig1, ax1 = plt.subplots(figsize=(12, 3))
fig1.canvas.manager.set_window_title('Shot Timeline')
plt.scatter(shot_times, np.full_like(shot_times, 0), marker='o', s=50,
            color='royalblue', edgecolors='black', zorder=3, label='shot')
plt.bar(shot_times, shot_facts, bottom=2, color='royalblue', edgecolor='black',
        width=5, label='distance from basket (0–10)')
ax1.spines['bottom'].set_position('zero')
ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')
ax1.spines['left'].set_color('none')
ax1.tick_params(axis='x', length=20)
ax1.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 720, 1440, 2160, 2880]))
ax1.set_yticks([])
_, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
ax1.set_xlim(-15, xmax); ax1.set_ylim(ymin, ymax + 5)
ax1.text(xmax, 2, "time", ha='right', va='top', size=10)
plt.legend(ncol=5, loc='upper left')
plt.tight_layout()
#plt.savefig("Shot_Timeline.png")
print("\nSaved: Shot_Timeline.png")

# ── Analysis dashboard ─────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(18, 14))
fig2.suptitle('Shot Detection Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig2, hspace=0.42, wspace=0.35)

# Panel A: Court map
ax_court = fig2.add_subplot(gs[0:2, 0:2])
ax_court.set_facecolor('#f5e6c8')
ax_court.set_title('A — Court Map: Detections vs False Positives', fontweight='bold')
ax_court.add_patch(plt.Rectangle((0, 0), 94, 50, fill=False, color='black', lw=2))
for x_paint, x_basket in [(0, BASKET_LEFT[0]), (94, BASKET_RIGHT[0])]:
    paint_x = min(x_paint, x_basket + 8) if x_paint == 0 else max(x_paint, x_basket - 8)
    ax_court.add_patch(plt.Rectangle(
        (min(x_paint, paint_x), 17), abs(paint_x - x_paint), 16,
        fill=True, facecolor='#d4b896', edgecolor='black', lw=1))
for bx, by in [(BASKET_LEFT[0], 25), (BASKET_RIGHT[0], 25)]:
    ax_court.plot(bx, by, 'ro', markersize=10, zorder=5)
for bx, by in [(BASKET_LEFT[0], BASKET_LEFT[1]), (BASKET_RIGHT[0], BASKET_RIGHT[1])]:
    ax_court.add_patch(plt.Circle((bx, by), BACKBOARD_EXCL_RADIUS,
                                   color='purple', fill=False, linestyle='--', lw=1.2, alpha=0.5))
ax_court.plot([], [], '--', color='purple', lw=1.2, alpha=0.6,
              label=f'Backboard excl. (r={BACKBOARD_EXCL_RADIUS}ft, z≥{BACKBOARD_EXCL_MIN_Z}ft)')
ax_court.scatter([s['x'] for s in true_positives], [s['y'] for s in true_positives],
                 c='steelblue', s=55, alpha=0.5, label=f'True positive ({len(true_positives)})', zorder=3)
if fp_high:
    ax_court.scatter([s['x'] for s in fp_high], [s['y'] for s in fp_high],
                     c='red', marker='X', s=110, alpha=0.85, label=f'FP high arc ({len(fp_high)})', zorder=4)
if fp_hook:
    ax_court.scatter([s['x'] for s in fp_hook], [s['y'] for s in fp_hook],
                     c='orange', marker='^', s=110, alpha=0.85, label=f'FP hook ({len(fp_hook)})', zorder=4)
ax_court.set_xlim(-2, 96); ax_court.set_ylim(-2, 52)
ax_court.set_xlabel('Court X (feet)'); ax_court.set_ylabel('Court Y (feet)')
ax_court.legend(loc='upper center', ncol=2, fontsize=8)

# Panel B: Donut
ax_donut = fig2.add_subplot(gs[0, 2])
ax_donut.set_title('B — Detection Breakdown', fontweight='bold')
sizes  = [len(true_positives), len(false_positives), len(missed_csv)]
colors = ['steelblue', 'red', 'gold']
labels = [f'True Pos\n({len(true_positives)})', f'False Pos\n({len(false_positives)})', f'Missed\n({len(missed_csv)})']
wedges, texts, autotexts = ax_donut.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
    startangle=90, wedgeprops=dict(width=0.55), textprops={'fontsize': 8})
for at in autotexts: at.set_fontsize(8)
ax_donut.text(0, 0, f"P={precision:.0f}%\nR={recall:.0f}%",
              ha='center', va='center', fontsize=9, fontweight='bold')

# Panel C: Per-quarter
ax_q = fig2.add_subplot(gs[1, 2])
ax_q.set_title('C — Detections per Quarter', fontweight='bold')
x = np.arange(4); w = 0.25
ax_q.bar(x - w, [sum(1 for s in true_positives  if s['quarter'] == q) for q in [1,2,3,4]], width=w, color='steelblue', label='True pos')
ax_q.bar(x,     [sum(1 for s in false_positives if s['quarter'] == q) for q in [1,2,3,4]], width=w, color='red',       label='False pos')
ax_q.bar(x + w, [sum(1 for r in missed_csv      if r['period']  == q) for q in [1,2,3,4]], width=w, color='gold',      label='Missed')
ax_q.set_xticks(x); ax_q.set_xticklabels(['Q1','Q2','Q3','Q4'])
ax_q.set_ylabel('Count'); ax_q.legend(fontsize=7)

# Panel D: Peak height histogram
ax_hist = fig2.add_subplot(gs[2, 0])
ax_hist.set_title('D — Peak Height Distribution', fontweight='bold')
bins = np.linspace(5, 20, 28)
ax_hist.hist([s['z'] for s in true_positives],  bins=bins, alpha=0.6, color='steelblue', label='True pos')
ax_hist.hist([s['z'] for s in false_positives], bins=bins, alpha=0.7, color='red',       label='False pos')
ax_hist.axvline(SHOT_THRESHOLD,       color='black',  linestyle='--', label=f'Threshold {SHOT_THRESHOLD}ft')
ax_hist.axvline(HOOK_MIN_Z,           color='green',  linestyle=':',  label=f'Hook min z {HOOK_MIN_Z}ft')
ax_hist.axvline(BACKBOARD_EXCL_MIN_Z, color='purple', linestyle=':',  label=f'Backboard excl. z≥{BACKBOARD_EXCL_MIN_Z}ft')
ax_hist.set_xlabel('Peak z (feet)'); ax_hist.set_ylabel('Count')
ax_hist.legend(fontsize=7)

# Panel E: Distance vs peak height scatter
ax_scat = fig2.add_subplot(gs[2, 1])
ax_scat.set_title('E — Peak Height vs Shot Distance', fontweight='bold')
tp_dist = [dist_to_nearest_basket_2d(s['x'], s['y']) for s in true_positives]
fp_dist = [dist_to_nearest_basket_2d(s['x'], s['y']) for s in false_positives]
ax_scat.scatter(tp_dist, [s['z'] for s in true_positives],  c='steelblue', alpha=0.4, s=35, label='True pos')
ax_scat.scatter(fp_dist, [s['z'] for s in false_positives], c='red', alpha=0.75, s=60, marker='X', label='False pos')
ax_scat.axhline(SHOT_THRESHOLD,       color='black',  linestyle='--', label=f'z={SHOT_THRESHOLD}ft')
ax_scat.axhline(HOOK_MIN_Z,           color='green',  linestyle=':',  label=f'hook min={HOOK_MIN_Z}ft')
ax_scat.axhline(BACKBOARD_EXCL_MIN_Z, color='purple', linestyle=':',  label=f'excl.={BACKBOARD_EXCL_MIN_Z}ft')
ax_scat.axvline(BACKBOARD_EXCL_RADIUS,color='purple', linestyle='-.', alpha=0.5, label=f'excl. dist={BACKBOARD_EXCL_RADIUS}ft')
ax_scat.set_xlabel('Dist from basket (ft)'); ax_scat.set_ylabel('Peak z (ft)')
ax_scat.legend(fontsize=7)

# Panel F: Missed shots table
ax_tbl = fig2.add_subplot(gs[2, 2])
ax_tbl.axis('off')
ax_tbl.set_title('F — Missed Shots Detail', fontweight='bold')
if missed_csv:
    tbl_data = []
    for r in missed_csv[:12]:
        desc = (r['desc'][:28] + '…') if len(r['desc']) > 28 else r['desc']
        tbl_data.append([f"Q{r['period']}", r['clock'], desc])
    if len(missed_csv) > 12:
        tbl_data.append(['…', '…', f'(+{len(missed_csv)-12} more)'])
    tbl = ax_tbl.table(cellText=tbl_data, colLabels=['Q', 'Clock', 'Description'],
                       loc='center', cellLoc='left')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.25)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#d0d8e8'); cell.set_text_props(fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f4f4f4')
else:
    ax_tbl.text(0.5, 0.5, '🎉 No missed shots!', ha='center', va='center',
                fontsize=12, transform=ax_tbl.transAxes)

#plt.savefig('shot_analysis_dashboard.png', dpi=150, bbox_inches='tight')
print("Saved: shot_analysis_dashboard.png")
plt.show()