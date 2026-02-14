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

# Pseudocode plan:

# Code above this comment gives us entired data as dictionaries and lists
# We also have the draft numbers to give away and to receive

# First Implementation: 
# Compare the average vorp of player(s) to give away vs player(s) to receive
# Query the draftDB database and record the names of the corresponding draft numbers
#       Collect the names to use as keys in the performance database
# Once we have a list of names to search for, collect each player's VORP stat
# With each VORP stat, average the value for the corresponding draft pick
# If
#       sum(players_to_give_away_VORP) > sum(players_to_receieve_VORP)
#           SHIT TRADE, WE'RE GIVING AWAY MORE VALUE
# else
#           GOOD TRADE, WE'VE GETTING MORE VALUE THAN WE'RE LOSING

# Check if any picks were entered
if not give_picks and not receive_picks:
    print("\nNo picks entered. Please run the program again with valid picks.")
    success = False
else:
    # Step 1: Build a dictionary to quickly lookup player seasons
    player_seasons = {}
    for row in players:
        name = row['Player']
        if name not in player_seasons:
            player_seasons[name] = []
        player_seasons[name].append(row)
    
    # Step 2: Function to get average VORP for a player
    def get_avg_vorp(player_name):
        if player_name not in player_seasons:
            return 0
        
        seasons = player_seasons[player_name]
        vorp_values = []
        
        for season in seasons:
            if season['VORP'] and season['VORP'].strip():  # Check if VORP exists and isn't empty
                try:
                    vorp_values.append(float(season['VORP']))
                except ValueError:
                    pass  # Skip if conversion fails
        
        if not vorp_values:
            return 0
        
        return np.mean(vorp_values)
    
    # Step 3: Get names and calculate VORP for picks to give away
    give_names = []
    give_vorp_values = []
    
    print("\n--- PICKS TO GIVE AWAY ---")
    for pick in draftPicks:
        pick_number = int(pick['numberPickOverall'])
        
        if pick_number in give_picks:
            name = pick['namePlayer']
            give_names.append(name)
            
            avg_vorp = get_avg_vorp(name)
            give_vorp_values.append(avg_vorp)
            
            print(f"Pick #{pick_number}: {name} - Avg VORP: {avg_vorp:.2f}")
    
    # Step 4: Get names and calculate VORP for picks to receive
    receive_names = []
    receive_vorp_values = []
    
    print("\n--- PICKS TO RECEIVE ---")
    for pick in draftPicks:
        pick_number = int(pick['numberPickOverall'])
        
        if pick_number in receive_picks:
            name = pick['namePlayer']
            receive_names.append(name)
            
            avg_vorp = get_avg_vorp(name)
            receive_vorp_values.append(avg_vorp)
            
            print(f"Pick #{pick_number}: {name} - Avg VORP: {avg_vorp:.2f}")
    
    # Step 5: Calculate totals and determine trade success
    total_give_vorp = sum(give_vorp_values)
    total_receive_vorp = sum(receive_vorp_values)
    
    print(f"\n--- SUMMARY ---")
    print(f"Total VORP given away: {total_give_vorp:.2f}")
    print(f"Total VORP received: {total_receive_vorp:.2f}")
    print(f"Net VORP change: {total_receive_vorp - total_give_vorp:.2f}")
    
    # Step 6: Apply the trade logic from pseudocode
    if total_give_vorp > total_receive_vorp:
        success = False  # SHIT TRADE, WE'RE GIVING AWAY MORE VALUE
    else:
        success = True   # GOOD TRADE, WE'RE GETTING MORE VALUE THAN WE'RE LOSING

# Print feeback on trade
# DO NOT CHANGE THESE OUTPUT MESSAGES
if success:
    print("\nTrade result: Success! This trade receives more value than it gives away.\n")
    # Print additional metrics/reasoning here
else:
    print("\nTrade result: Don't do it! This trade gives away more value than it receives.\n")
    # Print additional metrics/reasoning here