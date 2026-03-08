import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Feel free to add anything else you need here
import math

# Read in the SportVU tracking data
sportvu = []
with open('0021500495.json', mode='r') as sportvu_json:  
    sportvu = json.load(sportvu_json)


# YOUR SOLUTION GOES HERE
# These are the two arrays that you need to populate with actual data
shot_times = np.array([30, 705, 1870, 2500]) # Between 0 and 2880
shot_facts = np.array([5, 10, 8, 2]) # Scaled between 0 and 10

# CONSTANTS
RIM_HEIGHT = 10
RIM_DIAMETER = 1.5
BASKETBALL_DIAMETER = 1
BASKET_LEFT = (5.25, 25, 10)
BASKET_RIGHT = (88.75, 25, 10)

# PARAMETERS
# SHOT_THRESHOLD = RIM_HEIGHT - (BASKETBALL_DIAMETER / 2)
SHOT_THRESHOLD = RIM_HEIGHT - BASKETBALL_DIAMETER
FRAMES_AFTER_PEAK = 64
DUPLICATE_WINDOW = 2
H_TOLERANCE = 2.5 * RIM_DIAMETER
V_ABOVE = 2 * RIM_DIAMETER
V_BELOW = 0.5 * BASKETBALL_DIAMETER

# --- DATA PREPROCESSING ---
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

# Ensure data is in chronological order
all_moments.sort(key=lambda m: m['time'])

# --- DETECTION UTILITIES ---
def passed_near_basket(moments_window):
    """Checks if ball coordinates in a window are near either rim."""

    for m in moments_window:
        for basket in [BASKET_LEFT, BASKET_RIGHT]:
            h_dist = math.sqrt((m['x'] - basket[0])**2 + (m['y'] - basket[1])**2)
            v_diff = m['z'] - basket[2]
            
            v_ok = (v_diff >= -V_BELOW and v_diff <= V_ABOVE)
            if h_dist <= H_TOLERANCE and v_ok:
                return True
    return False

def calculate_scaled_dist(moment):
    """Calculates distance to nearest basket and scales 0-10 (up to 35ft)."""
    dist_l = math.sqrt((moment['x'] - BASKET_LEFT[0])**2 + (moment['y'] - BASKET_LEFT[1])**2)
    dist_r = math.sqrt((moment['x'] - BASKET_RIGHT[0])**2 + (moment['y'] - BASKET_RIGHT[1])**2)
    raw_dist = min(dist_l, dist_r)
    # Scale: 35ft is a 10, 0ft is a 0.
    return min(max((raw_dist / 35.0) * 10, 0), 10)

# --- MAIN DETECTION LOOP ---
shots_detected = []
in_shot = False
current_shot_peak = None

for i in range(1, len(all_moments) - 1):
    prev = all_moments[i-1]
    curr = all_moments[i]
    nxt = all_moments[i+1]

    if not in_shot:
        # Detect start of downward arc above the rim
        if (curr['z'] > SHOT_THRESHOLD and curr['z'] >= prev['z'] and curr['z'] >= nxt['z']):
            in_shot = True
            current_shot_peak = curr
    else:
        # Update peak if ball goes higher
        if curr['z'] > current_shot_peak['z']:
            current_shot_peak = curr

        # Once ball drops below rim, check if it was near the basket
        if curr['z'] < RIM_HEIGHT:
            window_after = all_moments[i : i + FRAMES_AFTER_PEAK]
            if passed_near_basket(window_after):
                # Avoid duplicate triggers for the same shot event
                if not shots_detected or (current_shot_peak['time'] - shots_detected[-1]['time'] > DUPLICATE_WINDOW):
                    shots_detected.append(current_shot_peak)
            
            in_shot = False
            current_shot_peak = None

# --- POPULATE OUTPUT ARRAYS ---
shot_times = np.array([s['time'] for s in shots_detected])
shot_facts = np.array([calculate_scaled_dist(s) for s in shots_detected])

# --- TIMELINE VISUALIZATION ---
fig, ax = plt.subplots(figsize=(12,3))
fig.canvas.manager.set_window_title('Shot Timeline')

plt.scatter(shot_times, np.full_like(shot_times, 0), marker='o', s=50, color='royalblue', edgecolors='black', zorder=3, label='shot')
# Label changed to reflect 'Shot Distance (Scaled)'
plt.bar(shot_times, shot_facts, bottom=2, color='royalblue', edgecolor='black', width=5, label='shot distance (0-10)') 

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

# plt.savefig("Shot_Timeline.png")


# ============================== BEGIN TESTING  ============================== #

def load_shot_csv():
    # Load CSV shots
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
    MATCH_WINDOW = 3.0  # TODO changing this window affects accuracy as well

    for csv_shot in csv_shots:
        # Find any JSON detection within 3 seconds of this CSV shot
        close = [s for s in shots_detected 
                if abs(s['time'] - csv_shot['time']) <= MATCH_WINDOW]
        if close:
            matched += 1
            best = min(close, key=lambda s: abs(s['time'] - csv_shot['time']))
            diff = best['time'] - csv_shot['time']
            print(f"✓ MATCH  | Q{csv_shot['period']} {csv_shot['clock']} | "
                f"diff={diff:+.1f}s | {csv_shot['desc']}")
        else:
            unmatched_csv.append(csv_shot)
            print(f"✗ MISSED | Q{csv_shot['period']} {csv_shot['clock']} | {csv_shot['desc']}")

    print(f"\n=== SUMMARY ===")
    print(f"CSV total shots:         {len(csv_shots)}")
    print(f"JSON total shots:        {len(shots_detected)}")
    print(f"Matched:                 {matched}")
    print(f"False negatives:         {len(unmatched_csv)}")
    print(f"False positives:         {len(shots_detected) - matched}")

    return csv_shots

# --- Call functions --- #
csv_shots = load_shot_csv()
print(f"\nParameters:")
print("Frame Window After Peak: ", FRAMES_AFTER_PEAK)
print("Shot Height Threshold: ", SHOT_THRESHOLD)
print("Duplicate Shot Window: ", DUPLICATE_WINDOW)
print("Horizontal Tolerance: ", H_TOLERANCE)
print("Upper Vertical Tolerance: ", V_ABOVE)
print("Lower Vertical Tolerance: ", V_BELOW)


# =============================== END TESTING ================================ #