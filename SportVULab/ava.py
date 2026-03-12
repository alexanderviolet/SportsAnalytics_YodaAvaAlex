import csv
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
# Feel free to add anything else you need here

# Read in the SportVU tracking data
sportvu = []
with open('0021500495.json', mode='r') as sportvu_json:
    sportvu = json.load(sportvu_json)

print(len(sportvu['events'][0]['moments']))


# YOUR SOLUTION GOES HERE
# These are the two arrays that you need to populate with actual data

BASKET_LEFT = (5.25, 25, 10)
BASKET_RIGHT = (88.75, 25, 10)
BASKET_RADIUS = 4  # feet - roughly two basketballs of distance from rim edge
SHOT_THRESHOLD = 8.5  # ball must peak above this height to count as a shot
MIN_GAP = 0.0  # minimum seconds between shots


def near_basket_horizontally(x, y, threshold=12):
    """Is the ball within threshold feet horizontally of either basket?"""
    d_left = math.sqrt((x - BASKET_LEFT[0])**2 + (y - BASKET_LEFT[1])**2)
    d_right = math.sqrt((x - BASKET_RIGHT[0])**2 + (y - BASKET_RIGHT[1])**2)
    return min(d_left, d_right) <= threshold

def moving_toward_basket(moments, i, lookback=5):
    """Is the ball getting closer to the nearest basket over recent frames?"""
    if i < lookback:
        return False
    curr = moments[i]
    prev = moments[i - lookback]

    d_left_curr = math.sqrt((curr['x'] - BASKET_LEFT[0])**2 + (curr['y'] - BASKET_LEFT[1])**2)
    d_right_curr = math.sqrt((curr['x'] - BASKET_RIGHT[0])**2 + (curr['y'] - BASKET_RIGHT[1])**2)
    d_left_prev = math.sqrt((prev['x'] - BASKET_LEFT[0])**2 + (prev['y'] - BASKET_LEFT[1])**2)
    d_right_prev = math.sqrt((prev['x'] - BASKET_RIGHT[0])**2 + (prev['y'] - BASKET_RIGHT[1])**2)

    curr_min = min(d_left_curr, d_right_curr)
    prev_min = min(d_left_prev, d_right_prev)
    return curr_min < prev_min  # getting closer to a basket

def passed_near_basket(moments_window):
    """Check if ball passed within BASKET_RADIUS feet of either basket during this window"""
    for m in moments_window:
        d_left = math.sqrt((m['x'] - BASKET_LEFT[0])**2 +
                          (m['y'] - BASKET_LEFT[1])**2 +
                          (m['z'] - BASKET_LEFT[2])**2)
        d_right = math.sqrt((m['x'] - BASKET_RIGHT[0])**2 +
                           (m['y'] - BASKET_RIGHT[1])**2 +
                           (m['z'] - BASKET_RIGHT[2])**2)
        if min(d_left, d_right) <= BASKET_RADIUS:
            return True
    return False

def deduplicate_shots(shots, min_gap=3.0):
    if not shots:
        return shots
    shots.sort(key=lambda s: s['time'])
    deduped = [shots[0]]
    for shot in shots[1:]:
        if shot['time'] - deduped[-1]['time'] > min_gap:
            deduped.append(shot)
    return deduped

# Build all moments
all_moments = []
for event in sportvu['events']:
    for moment in event['moments']:
        quarter = moment[0]
        game_clock = moment[2]
        ball = moment[5][0]
        seconds_since_start = (quarter - 1) * 720 + (720 - game_clock)
        all_moments.append({
            'time': seconds_since_start,
            'quarter': quarter,
            'clock': game_clock,
            'x': ball[2],
            'y': ball[3],
            'z': ball[4]
        })

all_moments.sort(key=lambda m: m['time'])

# Detect shots using a state machine
shots_detected = []
in_shot = False
current_shot_peak = None

for i in range(1, len(all_moments) - 1):
    prev = all_moments[i-1]
    curr = all_moments[i]
    next = all_moments[i+1]

    if not in_shot:
        # High arc shot: ball peaks above SHOT_THRESHOLD anywhere on court
        is_high_shot = (curr['z'] > SHOT_THRESHOLD and
                        curr['z'] >= prev['z'] and
                        curr['z'] >= next['z'])

        # Low arc shot: ball peaks above 6ft near the basket and is moving toward it
        # Catches hook shots, finger rolls, and short layups
        is_hook_shot = (curr['z'] > 6.0 and
                        curr['z'] >= prev['z'] and
                        curr['z'] >= next['z'] and
                        near_basket_horizontally(curr['x'], curr['y'], threshold=12) and
                        moving_toward_basket(all_moments, i))

        if is_high_shot or is_hook_shot:
            in_shot = True
            current_shot_peak = curr

    else:
        # Track the highest point of this shot arc
        if curr['z'] > current_shot_peak['z']:
            current_shot_peak = curr

        # Wait for ball to come back down below 8 feet
        if curr['z'] < 8:
            # Shot arc complete — check if ball passed near a basket
            window_after = all_moments[i:i+50]
            if passed_near_basket(window_after):
                shots_detected.append(current_shot_peak)
                print(f"Q{current_shot_peak['quarter']} {current_shot_peak['clock']:.1f}s left | "
                      f"time={current_shot_peak['time']:.1f}s | "
                      f"peak z={current_shot_peak['z']:.2f}ft | "
                      f"x={current_shot_peak['x']:.2f} y={current_shot_peak['y']:.2f}")
            # Reset regardless of whether it was a shot
            in_shot = False
            current_shot_peak = None

print(f"\nTotal shots detected from JSON: {len(shots_detected)}")

shots_detected = deduplicate_shots(shots_detected, min_gap=3.0)
print(f"After deduplication: {len(shots_detected)} shots")


# Load CSV shots for comparison
csv_shots = []
with open('0021500495.csv', mode='r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['EVENTMSGTYPE'] in ['1', '2']:
            period = int(row['PERIOD'])
            time_parts = row['PCTIMESTRING'].split(':')
            clock_remaining = int(time_parts[0]) * 60 + int(time_parts[1])
            seconds_since_start = (period - 1) * 720 + (720 - clock_remaining)
            csv_shots.append({
                'time': seconds_since_start,
                'period': period,
                'clock': row['PCTIMESTRING'],
                'made': row['EVENTMSGTYPE'] == '1',
                'desc': row['HOMEDESCRIPTION'] or row['VISITORDESCRIPTION']
            })

print("=== MATCHING CSV SHOTS TO JSON DETECTIONS ===\n")
matched = 0
unmatched_csv = []
MATCH_WINDOW = 5.5  # seconds — slightly wider to catch edge cases

for csv_shot in csv_shots:
    close = [s for s in shots_detected
             if abs(s['time'] - csv_shot['time']) <= MATCH_WINDOW]
    if close:
        matched += 1
        best = min(close, key=lambda s: abs(s['time'] - csv_shot['time']))
        diff = best['time'] - csv_shot['time']
        print(f"✓ MATCH  | Q{csv_shot['period']} {csv_shot['clock']} | "
              f"diff={diff:+.1f}s | {csv_shot['desc']}")
    else:
        nearest = min(shots_detected, key=lambda s: abs(s['time'] - csv_shot['time']))
        nearest_diff = nearest['time'] - csv_shot['time']
        unmatched_csv.append(csv_shot)
        print(f"✗ MISSED | Q{csv_shot['period']} {csv_shot['clock']} | "
              f"nearest JSON={nearest_diff:+.1f}s | {csv_shot['desc']}")

print(f"\n=== SUMMARY ===")
print(f"CSV total shots:         {len(csv_shots)}")
print(f"JSON total shots:        {len(shots_detected)}")
print(f"Matched:                 {matched}")
print(f"Missed by JSON:          {len(unmatched_csv)}")
print(f"Extra in JSON (false +): {len(shots_detected) - matched}")

# Populate shot_times and shot_facts for visualization
shot_times = np.array([s['time'] for s in shots_detected])

def shot_fact(moment):
    """Shot distance from basket, scaled 0-10"""
    d_left = math.sqrt((moment['x'] - BASKET_LEFT[0])**2 + (moment['y'] - BASKET_LEFT[1])**2)
    d_right = math.sqrt((moment['x'] - BASKET_RIGHT[0])**2 + (moment['y'] - BASKET_RIGHT[1])**2)
    dist = min(d_left, d_right)
    return min(10, dist * (10 / 30))

shot_facts = np.array([shot_fact(s) for s in shots_detected])

# This code creates the timeline display from the shot_times
# and shot_facts arrays.
# DO NOT MODIFY THIS CODE APART FROM THE SHOT FACT LABEL
fig, ax = plt.subplots(figsize=(12,3))
fig.canvas.manager.set_window_title('Shot Timeline')

plt.scatter(shot_times, np.full_like(shot_times, 0), marker='o', s=50, color='royalblue', edgecolors='black', zorder=3, label='shot')
plt.bar(shot_times, shot_facts, bottom=2, color='royalblue', edgecolor='black', width=5, label='shot fact') # <- This is the label you can modify

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
