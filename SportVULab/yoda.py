import csv
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

with open('0021500495.json', mode='r') as sportvu_json:
    sportvu = json.load(sportvu_json)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
BASKET_LEFT    = (5.25, 25)
BASKET_RIGHT   = (88.75, 25)
RIM_THRESHOLD  = 1.5       # x/y feet from hoop center to count as rim zone

MIN_ARRIVAL_Z  = 9.0       # ball must arrive at rim at z >= 9ft
COOLDOWN_S     = 3.0       # min seconds between recorded shots
LOOKBACK_FRAMES = 75       # frames (~3s at 25Hz) to look back for arc

# FP reduction filters
MIN_ARC_HEIGHT   = 2.0     # ball must rise/fall at least this far in lookback
                           # — real shots have a clear arc; passes/dead balls don't
MAX_PEAK_AGE     = 70      # arc peak must occur within this many frames of rim arrival
                           # — if peak was long ago the ball has been drifting, not flying
MAX_PRE_RIM_DIST = 2.0     # first frame of lookback must be this far from rim
                           # — if ball was already near rim, it's lingering not arriving

# Nearest-defender scaling: 6ft away = wide open = score of 10
OPEN_DIST = 6.0


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def xy_dist(ball, basket):
    return ((ball[2] - basket[0])**2 + (ball[3] - basket[1])**2) ** 0.5

def to_game_time(quarter, game_clock):
    return (quarter - 1) * 720 + (720 - game_clock)

def nearest_defender_dist(moment):
    """
    Distance to the nearest non-shooter player at the release moment.
    moment[5][0]  = ball
    moment[5][1:] = all 10 players
    Sort by distance to ball, skip index 0 (shooter), return index 1 (nearest other).
    """
    ball_x = moment[5][0][2]
    ball_y = moment[5][0][3]
    players = moment[5][1:]
    if len(players) < 2:
        return 0.0
    dists = sorted(
        ((p[2] - ball_x)**2 + (p[3] - ball_y)**2) ** 0.5
        for p in players
    )
    return dists[1]   # nearest defender (skip shooter at dists[0])


# ---------------------------------------------------------------------------
# STEP 1: Flatten + deduplicate all moments into one chronological timeline
# ---------------------------------------------------------------------------
seen_timestamps = set()
all_moments = []
for event in sportvu['events']:
    for moment in event['moments']:
        ts = moment[1]
        if ts not in seen_timestamps:
            seen_timestamps.add(ts)
            all_moments.append(moment)

all_moments.sort(key=lambda m: m[1])


# ---------------------------------------------------------------------------
# STEP 2: Group consecutive rim-zone moments into clusters
# ---------------------------------------------------------------------------
clusters = []
current_cluster = []
for moment in all_moments:
    ball = moment[5][0]
    near_rim = (xy_dist(ball, BASKET_LEFT)  < RIM_THRESHOLD or
                xy_dist(ball, BASKET_RIGHT) < RIM_THRESHOLD)
    if near_rim:
        current_cluster.append(moment)
    else:
        if current_cluster:
            clusters.append(current_cluster)
            current_cluster = []
if current_cluster:
    clusters.append(current_cluster)


# ---------------------------------------------------------------------------
# STEP 3: Timestamp → index map for fast lookback
# ---------------------------------------------------------------------------
ts_to_idx = {m[1]: i for i, m in enumerate(all_moments)}


# ---------------------------------------------------------------------------
# STEP 4: Find release moment (lowest z before arc peak in lookback window)
# ---------------------------------------------------------------------------
def find_release_moment(pre_cluster):
    z_vals = [m[5][0][4] for m in pre_cluster]
    if not z_vals:
        return None
    peak_idx    = int(np.argmax(z_vals))
    release_idx = int(np.argmin(z_vals[:peak_idx + 1]))
    return pre_cluster[release_idx]


# ---------------------------------------------------------------------------
# STEP 5: Detect shots
#
# Original filters:
#   a) Ball arrives at rim already elevated (z >= MIN_ARRIVAL_Z)
#   b) Cooldown between shots (COOLDOWN_S)
#   c) Shot clock != 24.0 at arrival (suppresses net-swish from prior basket)
#
# New FP-reduction filters:
#   d) Ball must be descending on arrival (z falling in first cluster frames)
#      — real shots come DOWN into the basket; rising ball = alley-oop pass
#   e) Arc height in lookback >= MIN_ARC_HEIGHT
#      — real shots have a clear rise-and-fall; passes/dead balls don't
#   f) Arc peak must be recent (within MAX_PEAK_AGE frames of rim arrival)
#      — old peak means ball has been drifting, not freshly shot
#   g) Ball must have come from outside the rim zone (first lookback frame
#      is far from rim) — ball already near rim = lingering dead ball
# ---------------------------------------------------------------------------
detected_shots = []
detected_facts = []
last_shot_time = -999.0

for cluster in clusters:
    first_moment = cluster[0]
    first_z      = first_moment[5][0][4]

    # (a) Ball must arrive elevated
    if first_z < MIN_ARRIVAL_Z:
        continue

    # (d) Ball must be descending on arrival
    # Check that z drops from the first to second frame of the cluster
    if len(cluster) >= 2:
        second_z = cluster[1][5][0][4]
        if second_z > first_z:   # ball still rising = not a shot descending to rim
            continue

    first_ts  = first_moment[1]
    first_idx = ts_to_idx[first_ts]
    start_idx = max(0, first_idx - LOOKBACK_FRAMES)
    pre_cluster = all_moments[start_idx:first_idx]

    if not pre_cluster:
        continue

    game_time = to_game_time(first_moment[0], first_moment[2])

    # (b) Cooldown
    if game_time - last_shot_time < COOLDOWN_S:
        continue

    # (c) Suppress net-swish from prior made basket
    if first_moment[3] == 24.0:
        continue

    z_vals   = [m[5][0][4] for m in pre_cluster]
    peak_idx = int(np.argmax(z_vals))

    # (e) Arc must have meaningful height
    arc_height = max(z_vals) - min(z_vals)
    if arc_height < MIN_ARC_HEIGHT:
        continue

    # (f) Peak must be recent — not an old drift
    frames_since_peak = len(z_vals) - 1 - peak_idx
    if frames_since_peak > MAX_PEAK_AGE:
        continue

    # (g) Ball must have originated from outside the rim zone
    first_pre_ball = pre_cluster[0][5][0]
    if (xy_dist(first_pre_ball, BASKET_LEFT)  < MAX_PRE_RIM_DIST or
            xy_dist(first_pre_ball, BASKET_RIGHT) < MAX_PRE_RIM_DIST):
        continue

    last_shot_time = game_time

    release_m = find_release_moment(pre_cluster)
    if release_m is None:
        continue

    # Shot fact: nearest defender at release, scaled 0–10
    # Tall bar = open shot, short bar = heavily contested
    defender_dist = nearest_defender_dist(release_m)
    shot_fact = min(10.0, (defender_dist / OPEN_DIST) * 10.0)

    detected_shots.append(game_time)
    detected_facts.append(shot_fact)


# ---------------------------------------------------------------------------
# STEP 6: Populate required arrays
# ---------------------------------------------------------------------------
shot_times = np.array(detected_shots)
shot_facts = np.array(detected_facts)


# ---------------------------------------------------------------------------
# STEP 7: Validate against CSV ground truth
# ---------------------------------------------------------------------------
WINDOW_BEFORE = 3.0
WINDOW_AFTER  = 1.0

csv_shots_gt = []
with open('0021500495.csv', mode='r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['EVENTMSGTYPE'] not in ('1', '2'):
            continue
        desc = row['HOMEDESCRIPTION'] or row['VISITORDESCRIPTION']
        if 'BLOCK' in desc:
            continue
        mins, secs = row['PCTIMESTRING'].strip().split(':')
        clock_s = int(mins) * 60 + int(secs)
        gt = (int(row['PERIOD']) - 1) * 720 + (720 - clock_s)
        csv_shots_gt.append(gt)

candidates = []
for ci, csv_gt in enumerate(csv_shots_gt):
    for ji, json_gt in enumerate(detected_shots):
        diff = json_gt - csv_gt
        if -WINDOW_BEFORE <= diff <= WINDOW_AFTER:
            candidates.append((abs(diff), ci, ji))
candidates.sort()

used_csv  = set()
used_json = set()
for _, ci, ji in candidates:
    if ci in used_csv or ji in used_json:
        continue
    used_csv.add(ci)
    used_json.add(ji)

true_positives  = len(used_csv)
missed          = len(csv_shots_gt) - true_positives
false_positives = len(detected_shots) - len(used_json)
recall          = true_positives / len(csv_shots_gt) * 100
precision       = true_positives / len(detected_shots) * 100

print(f'CSV total shots:           {len(csv_shots_gt)}')
print(f'JSON detections:           {len(detected_shots)}')
print(f'Matched (true positives):  {true_positives}')
print(f'Missed by JSON:            {missed}')
print(f'False positives:           {false_positives}')
print(f'Recall:                    {recall:.1f}%')
print(f'Precision:                 {precision:.1f}%')


# ---------------------------------------------------------------------------
# STEP 8: Shot Timeline (DO NOT MODIFY apart from label)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 3))
fig.canvas.manager.set_window_title('Shot Timeline')

plt.scatter(shot_times, np.full_like(shot_times, 0), marker='o', s=50,
            color='royalblue', edgecolors='black', zorder=3, label='shot')
plt.bar(shot_times, shot_facts, bottom=2, color='royalblue', edgecolor='black',
        width=5, label='nearest defender distance (scaled 0–10)')

ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.tick_params(axis='x', length=20)
ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 720, 1440, 2160, 2880]))
ax.set_yticks([])

_, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(-15, xmax)
ax.set_ylim(ymin, ymax + 5)
ax.text(xmax, 2, "time", ha='right', va='top', size=10)
plt.legend(ncol=5, loc='upper left')

plt.tight_layout()
plt.show()

#plt.savefig("Shot_Timeline.png")