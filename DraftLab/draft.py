import csv
import numpy as np
# You may or may not want to use this package, or others like it
# this is just a starting point for you
from sklearn.linear_model import LinearRegression

# Read the player database into an array of dictionaries
players = []
with open('playerDB.csv', mode='r') as player_csv:
    player_reader = csv.DictReader(player_csv)
    line_count = 0
    for row in player_reader:
        players.append(dict(row))

# Read the draft database into an array of dictionaries
draftPicks = []
with open('draftDB.csv', mode='r') as draft_csv:
    draft_reader = csv.DictReader(draft_csv)
    line_count = 0
    for row in draft_reader:
        draftPicks.append(dict(row))


# Get the draft picks to give/receive from the user
# You can assume that this input will be entered as expected
# DO NOT CHANGE THESE PROMPTS
print("\nSelect the picks to be traded away and the picks to be received in return.")
print("For each entry, provide 1 or more pick numbers from 1-60 as a comma-separated list.")
print("As an example, to trade the 1st, 3rd, and 25th pick you would enter: 1, 3, 25.\n")
give_str = input("Picks to give away: ")
receive_str = input("Picks to receive: ")

# Convert user input to an array of ints
give_picks = list(map(int, give_str.split(',')))
receive_picks = list(map(int, receive_str.split(',')))

# Success indicator that you will need to update based on your trade analysis
success = True



# YOUR SOLUTION GOES HERE

player_seasons = {}

for row in players:
    name = row['Player']
    if name not in player_seasons:
        player_seasons[name] = []
    player_seasons[name].append(row)

def average_player_stats(name):
    if name not in player_seasons:
        return None

    seasons = player_seasons[name]

    per_list = []
    ws_list = []
    bpm_list = []

    for season in seasons:
        if season['PER'] != '':
            per_list.append(float(season['PER']))

        if season['WS'] != '':
            ws_list.append(float(season['WS']))

        if season['BPM'] != '':
            bpm_list.append(float(season['BPM']))

    if len(per_list) == 0 and len(ws_list) == 0 and len(bpm_list) == 0:
        return None

    return {
        "avg_PER": np.mean(per_list) if per_list else None,
        "avg_WS": np.mean(ws_list) if ws_list else None,
        "avg_BPM": np.mean(bpm_list) if bpm_list else None
    }



for pick in draftPicks:
    pick_number = int(pick['numberPickOverall'])

    if pick_number in give_picks:
        name = pick['namePlayer']
        print(f"\nGive Pick #{pick_number}: {name}")

        stats = average_player_stats(name)
        if stats:
            print("Average Stats:", stats)
        else:
            print("No stats found.")

    if pick_number in receive_picks:
        name = pick['namePlayer']
        print(f"\nReceive Pick #{pick_number}: {name}")

        stats = average_player_stats(name)
        if stats:
            print("Average Stats:", stats)
        else:
            print("No stats found.")


## End Chat Code ##

# Print feeback on trade
# DO NOT CHANGE THESE OUTPUT MESSAGES
if success:
    print("\nTrade result: Success! This trade receives more value than it gives away.\n")
    # Print additional metrics/reasoning here
else:
    print("\nTrade result: Don't do it! This trade gives away more value than it receives.\n")
    # Print additional metrics/reasoning here