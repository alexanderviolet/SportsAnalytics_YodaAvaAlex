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




# YOUR SOLUTION GOES HERE
# These are the two arrays that you need to populate with actual data
shot_times = np.array([30, 705, 1870, 2500]) # Between 0 and 2880
shot_facts = np.array([5, 10, 8, 2]) # Scaled between 0 and 10

RIM_HEIGHT = 10
BASKETBALL_DIAMETER = 1
SHOT_PEAK_MIN = RIM_HEIGHT + BASKETBALL_DIAMETER
FRAMES_AFTER_PEAK = 32 # TODO assuming 32 frames per second 16 frames is ~0.5 seconds
SHOT_THRESHOLD = RIM_HEIGHT - (BASKETBALL_DIAMETER / 2)

BASKET_LEFT = (5.25, 25, 10)
BASKET_RIGHT = (88.75, 25, 10)

# Convert sportsvu events into the relevant data of the ball's location
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

# Helper function to calculate distance
# TODO: Impose stricter penalty on x and y rather than z. We're getting A LOT of 
# false positives. I think it would be better to try computing the horizontal 
# distance only to basket and impose less strictness on vertical distance
# especially if it's above the basket. 
def passed_near_basket(moments_window):
    """Check if ball passed within tolerance of either basket"""
    for m in moments_window:
        d_left = math.sqrt((m['x'] - BASKET_LEFT[0])**2 +
                          (m['y'] - BASKET_LEFT[1])**2 +
                          (m['z'] - BASKET_LEFT[2])**2)
        d_right = math.sqrt((m['x'] - BASKET_RIGHT[0])**2 +
                           (m['y'] - BASKET_RIGHT[1])**2 +
                           (m['z'] - BASKET_RIGHT[2])**2)
        if min(d_left, d_right) <= (BASKETBALL_DIAMETER / 2):
            return True
    return False


# Detect shots
shots_detected = []
in_shot = False  # Boolean to determine if ball is in arc
current_shot_peak = None

for i in range(1, len(all_moments) - 1):
    prev = all_moments[i-1]
    curr = all_moments[i]
    next = all_moments[i+1]

    if not in_shot:
        # Check if ball is passing within an arc for within a 3 frame window
        if (curr['z'] > SHOT_THRESHOLD and # if shot above minimum height
            curr['z'] >= prev['z'] and
            curr['z'] >= next['z']):
            # Current z is at its maximum, so the ball is going down now
            # Assume that we should check any ball movement above SHOT_THRESHOLD
            # could be a shot. What's left to do now is check if near basket
            in_shot = True
            current_shot_peak = curr
    else:
        # We are currently attempting a shot based of increasing arc
        if curr['z'] > current_shot_peak['z']:
            current_shot_peak = curr

        # wait for ball to come below rim
        if curr['z'] < RIM_HEIGHT:
            # shot arc complete, check how close it is to basket
            window_after = all_moments[i:i+FRAMES_AFTER_PEAK] # TODO: Adjust window of peak to determine if shot is valid. 
            if passed_near_basket(window_after):
                shots_detected.append(current_shot_peak)
                # TESTING: Print our guesses
                # print(f"Q{current_shot_peak['quarter']} {current_shot_peak['clock']:.1f}s left | "
                #       f"time={current_shot_peak['time']:.1f}s | "
                #       f"peak z={current_shot_peak['z']:.2f}ft | "
                #       f"x={current_shot_peak['x']:.2f} y={current_shot_peak['y']:.2f}")
                
                # reset to waiting shot state:
                in_shot = False
                current_shot_peak = None


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
    MATCH_WINDOW = 3.0  # seconds

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
    print(f"Missed by JSON:          {len(unmatched_csv)}")
    print(f"Extra in JSON (false +): {len(shots_detected) - matched}")

    return csv_shots

# --- Call functions --- #
csv_shots = load_shot_csv()

# =============================== END TESTING ================================ #