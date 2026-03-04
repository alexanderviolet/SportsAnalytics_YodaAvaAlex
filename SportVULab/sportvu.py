import csv
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# Feel free to add anything else you need here

# Read in the SportVU tracking data
sportvu = []
with open('0021500495.json', mode='r') as sportvu_json:
    sportvu = json.load(sportvu_json)

print(len(sportvu['events'][0]['moments']))

# for event in sportvu['events']:
#     for moment in event['moments']:
#         # The ball is always the first entry in the players list (index 5)
#         # Ball has team_id = -1, player_id = -1
#         ball = moment[5][0]
#         quarter = moment[0]
#         game_clock = moment[2]
#         x = ball[2]
#         y = ball[3]
#         z = ball[4]
#         print(f"Q{quarter} {game_clock:.1f}s remaining | Ball: x={x:.2f}, y={y:.2f}, z={z:.2f}")


import json
import csv

with open('0021500495.json', mode='r') as sportvu_json:
    sportvu = json.load(sportvu_json)

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
            'x': ball[2],
            'y': ball[3],
            'z': ball[4]
        })

with open('0021500495.csv', mode='r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['EVENTMSGTYPE'] in ['1', '2']:
            period = int(row['PERIOD'])
            time_parts = row['PCTIMESTRING'].split(':')
            clock_remaining = int(time_parts[0]) * 60 + int(time_parts[1])
            seconds_since_start = (period - 1) * 720 + (720 - clock_remaining)

            # Look at a 3 second window BEFORE the event timestamp
            # (shot happens before the event is logged)
            window = [m for m in all_moments 
                      if -3 <= m['time'] - seconds_since_start <= 0.5]

            if window:
                peak = max(window, key=lambda m: m['z'])
                made = row['EVENTMSGTYPE'] == '1'
                desc = row['HOMEDESCRIPTION'] or row['VISITORDESCRIPTION']
                print(f"Q{period} {row['PCTIMESTRING']} | {'SCORED' if made else 'MISS'} | peak z={peak['z']:.2f}ft | x={peak['x']:.2f} | {desc}")
# YOUR SOLUTION GOES HERE
# These are the two arrays that you need to populate with actual data
shot_times = np.array([30, 705, 1870, 2500]) # Between 0 and 2880
shot_facts = np.array([5, 10, 8, 2]) # Scaled between 0 and 10




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
