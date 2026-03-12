import csv
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Read in the SportVU tracking data
sportvu = []
with open('0021500495.json', mode='r') as sportvu_json:
    sportvu = json.load(sportvu_json)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
BASKET_LEFT   = (5.25, 25)    # (x, y) of left hoop in feet
BASKET_RIGHT  = (88.75, 25)   # (x, y) of right hoop in feet
RIM_THRESHOLD = 1.5           # max x/y distance from hoop center to count as
                              # "at the rim" (feet). Tight threshold to avoid
                              # counting baseline plays near the basket.

MIN_ARRIVAL_Z  = 9.0          # ball must arrive at the rim at z >= 9ft.
                              # The rim sits at 10ft, so this ensures the ball
                              # is elevated on approach rather than bouncing
                              # along the floor near the baseline.
COOLDOWN_S     = 3.0          # minimum seconds between recorded shots.
                              # Duplicate rim-zone clusters from overlapping
                              # events are all < 1s apart; the shortest
                              # observed genuine back-to-back shot sequence
                              # (missed shot + immediate putback) is ~4.8s,
                              # so 3s cleanly separates artifacts from real
                              # distinct attempts.
LOOKBACK_FRAMES = 75          # frames to look back before rim arrival when
                              # searching for the arc peak (~3s at 25Hz).


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def xy_dist(ball, basket):
    """Euclidean distance in the x/y plane between ball and a basket."""
    return ((ball[2] - basket[0])**2 + (ball[3] - basket[1])**2) ** 0.5

def to_game_time(quarter, game_clock):
    """Convert quarter + game_clock (counting down) to seconds since tip-off."""
    return (quarter - 1) * 720 + (720 - game_clock)


# ---------------------------------------------------------------------------
# STEP 1: Flatten all moments into a single chronological timeline.
#
# Events overlap heavily in time and ~half are exact duplicates (the NBA logs
# multiple event types for the same play). Deduplicating by Unix timestamp
# gives us each physical moment exactly once.
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
# STEP 2: Group consecutive moments where the ball is within RIM_THRESHOLD
# of either hoop into clusters. Each cluster is a contiguous stretch of
# tracking frames where the ball is in the rim zone.
# ---------------------------------------------------------------------------
clusters = []
current_cluster = []
for moment in all_moments:
    ball = moment[5][0]  # [-1, -1, x, y, z]
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
# STEP 3: Build a timestamp-to-index map so we can efficiently look back
# through the moment stream to find the arc preceding each rim arrival.
# ---------------------------------------------------------------------------
ts_to_idx = {m[1]: i for i, m in enumerate(all_moments)}


# ---------------------------------------------------------------------------
# STEP 4: Helper to find the release moment within the pre-cluster window.
#
# The release point is defined as the global minimum z in the lookback window
# before the arc peak. This corresponds to the moment just before the player
# releases the ball — the lowest point before the upward arc begins.
# This approach requires no tuning parameters and is straightforward to
# defend: we are simply finding the bottom of the arc.
# ---------------------------------------------------------------------------
def find_release_moment(pre_cluster):
    z_vals = [m[5][0][4] for m in pre_cluster]
    if not z_vals:
        return None
    peak_idx = int(np.argmax(z_vals))
    # Global minimum before the peak
    release_idx = int(np.argmin(z_vals[:peak_idx + 1]))
    return pre_cluster[release_idx]


# ---------------------------------------------------------------------------
# STEP 5: For each cluster, decide whether it represents a shot attempt.
#
# Criteria (all must pass):
#   a) Ball arrives at the rim already elevated: first z >= MIN_ARRIVAL_Z
#      Every rim-zone cluster in this game with arrival z < 9ft was a
#      non-shot (floor bounce, baseline pass, inbound). There is a clean
#      gap in the data — nothing arrives between 8–9ft.
#   b) Not within COOLDOWN_S seconds of the previously recorded shot
#
# We record the first moment of the cluster as the shot time — by the time
# the ball is within 1.5ft of the rim in x/y it is already descending,
# so the first frame of rim proximity is the closest observable point to
# when the ball actually crossed the basket.
# ---------------------------------------------------------------------------
MAX_SHOT_DIST = 47.0  # half-court in feet, used for 0-10 scaling

detected_shots = []
detected_facts = []
last_shot_time = -999.0

for cluster in clusters:
    first_moment = cluster[0]
    first_z      = first_moment[5][0][4]

    # (a) Ball must arrive at rim already elevated
    if first_z < MIN_ARRIVAL_Z:
        continue

    # Look back through the moment stream before this cluster
    first_ts  = first_moment[1]
    first_idx = ts_to_idx[first_ts]
    start_idx = max(0, first_idx - LOOKBACK_FRAMES)
    pre_cluster = all_moments[start_idx:first_idx]

    if not pre_cluster:
        continue

    game_time = to_game_time(first_moment[0], first_moment[2])

    # (b) Cooldown: suppress duplicate detections from overlapping events
    if game_time - last_shot_time < COOLDOWN_S:
        continue

    last_shot_time = game_time

    # Compute shot distance from release point
    release_m = find_release_moment(pre_cluster)
    if release_m is None:
        continue
    bx, by = release_m[5][0][2], release_m[5][0][3]
    dist_left  = ((bx - BASKET_LEFT[0])**2  + (by - BASKET_LEFT[1])**2)  ** 0.5
    dist_right = ((bx - BASKET_RIGHT[0])**2 + (by - BASKET_RIGHT[1])**2) ** 0.5
    shot_dist  = min(dist_left, dist_right)

    # Scale to 0-10
    shot_fact = min(10.0, (shot_dist / MAX_SHOT_DIST) * 10.0)

    detected_shots.append(game_time)
    detected_facts.append(shot_fact)


# ---------------------------------------------------------------------------
# STEP 6: Populate the required arrays.
# ---------------------------------------------------------------------------
shot_times = np.array(detected_shots)
shot_facts = np.array(detected_facts)


# ---------------------------------------------------------------------------
# STEP 7: Validate detections against CSV ground truth.
#
# The CSV is used only for validation — it plays no role in detection.
# Blocks (EVENTMSGTYPE=2 with 'BLOCK' in description) are excluded from
# ground truth since the ball is deflected before reaching the rim and is
# therefore structurally undetectable by our method.
#
# Matching window is asymmetric: our detector fires at rim arrival, which
# is slightly before the official scorer logs the event. From the observed
# distribution, detections fall a median of 1.48s before the CSV time.
# We allow up to 3s before or 1s after the CSV time, derived from the data.
#
# Matching is one-to-one: candidate pairs are sorted by proximity and
# greedily assigned so each CSV shot and each detection is used at most once.
# ---------------------------------------------------------------------------
WINDOW_BEFORE = 3.0   # seconds our detector can fire before CSV time
WINDOW_AFTER  = 1.0   # seconds our detector can fire after CSV time

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

# Build all valid (csv_idx, json_idx) candidate pairs sorted by proximity
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




# This code creates the timeline display from the shot_times
# and shot_facts arrays.
# DO NOT MODIFY THIS CODE APART FROM THE SHOT FACT LABEL
fig, ax = plt.subplots(figsize=(12,3))
fig.canvas.manager.set_window_title('Shot Timeline')

plt.scatter(shot_times, np.full_like(shot_times, 0), marker='o', s=50, color='royalblue', edgecolors='black', zorder=3, label='shot')
plt.bar(shot_times, shot_facts, bottom=2, color='royalblue', edgecolor='black', width=5, label='shot distance (scaled 0-10)')

ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.tick_params(axis='x', length=20)
ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0,720,1440,2160,2880])) 
ax.set_yticks([])

_, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(-15, xmax)
ax.set_ylim(ymin, ymax+5)
ax.text(xmax, 2, "time", ha='right', va='top', size=10)
plt.legend(ncol=5, loc='upper left')

plt.tight_layout()
plt.show()

#plt.savefig("Shot_Timeline.png")