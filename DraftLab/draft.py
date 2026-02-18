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
    # Add these functions after your existing get_avg_vorp function

    def get_stat_average(player_name, stat_column):
        """Get average of a specific stat for a player across their career"""
        if player_name not in player_seasons:
            return 0
        
        seasons = player_seasons[player_name]
        stat_values = []
        
        for season in seasons:
            if season[stat_column] and season[stat_column].strip():
                try:
                    stat_values.append(float(season[stat_column]))
                except ValueError:
                    pass
        
        if not stat_values:
            return 0
        
        return np.mean(stat_values)

    def calculate_player_score(player_name, weights=None):
        """
        Calculate a composite score for a player based on multiple statistics
        Default weights can be adjusted based on what you value most
        """
        if weights is None:
            # You can adjust these weights based on your preferences
            weights = {
                'VORP': 0.50,
                'TS%': 0.10,
                'TRB%': 0.10,
                'AST%': 0.10,
                'TOV%': 0.10,  # Lower TOV% is better, so we'll handle that
                'BLK%': 0.10
            }
        
        scores = {}
        
        # Get each stat average
        for stat, weight in weights.items():
            if stat == 'TOV%':  # For TOV%, lower is better
                raw_value = get_stat_average(player_name, stat)
                # Convert so that lower TOV% gives higher score
                # Using max TOV% of 25 as baseline (typical max)
                normalized = max(0, 25 - raw_value) / 25
                scores[stat] = normalized * weight
            else:
                raw_value = get_stat_average(player_name, stat)
                # Simple normalization based on typical ranges
                # You might want to adjust these ranges based on your data
                if stat == 'VORP':
                    normalized = min(raw_value / 10, 1.0)  # Cap at 10 VORP
                elif stat == 'TS%':
                    normalized = raw_value / 0.7  # 70% TS is elite
                elif stat in ['TRB%', 'AST%', 'BLK%']:
                    normalized = raw_value / 30  # 30% is elite
                else:
                    normalized = raw_value / 20
                
                scores[stat] = min(normalized * weight, weight)
        
        return sum(scores.values())

    # In your main solution, replace the VORP calculations:

    # Step 3: Get composite scores for picks to give away
    give_names = []
    give_scores = []

    print("\n--- PICKS TO GIVE AWAY ---")
    for pick in draftPicks:
        pick_number = int(pick['numberPickOverall'])
        
        if pick_number in give_picks:
            name = pick['namePlayer']
            give_names.append(name)
            
            composite_score = calculate_player_score(name)
            give_scores.append(composite_score)
            
            print(f"Pick #{pick_number}: {name} - Composite Score: {composite_score:.3f}")

    # Step 4: Get composite scores for picks to receive
    receive_names = []
    receive_scores = []

    print("\n--- PICKS TO RECEIVE ---")
    for pick in draftPicks:
        pick_number = int(pick['numberPickOverall'])
        
        if pick_number in receive_picks:
            name = pick['namePlayer']
            receive_names.append(name)
            
            composite_score = calculate_player_score(name)
            receive_scores.append(composite_score)
            
            print(f"Pick #{pick_number}: {name} - Composite Score: {composite_score:.3f}")

    # Step 5: Calculate totals and determine trade success
    total_give_score = sum(give_scores)
    total_receive_score = sum(receive_scores)

    print(f"\n--- SUMMARY ---")
    print(f"Total composite score given away: {total_give_score:.3f}")
    print(f"Total composite score received: {total_receive_score:.3f}")
    print(f"Net composite score change: {total_receive_score - total_give_score:.3f}")

    # Step 6: Apply the trade logic
    if total_give_score > total_receive_score:
        success = False
    else:
        success = True

# Print feeback on trade
# DO NOT CHANGE THESE OUTPUT MESSAGES
if success:
    print("\nTrade result: Success! This trade receives more value than it gives away.\n")
    # Print additional metrics/reasoning here
else:
    print("\nTrade result: Don't do it! This trade gives away more value than it receives.\n")
    # Print additional metrics/reasoning here

### Graphing Section ###

import matplotlib.pyplot as plt
import numpy as np

# Step 1: Build dictionary mapping pick number -> list of player names
pick_to_players = {}

for pick in draftPicks:
    try:
        pick_number = int(pick['numberPickOverall'])
        name = pick['namePlayer']
        
        if pick_number not in pick_to_players:
            pick_to_players[pick_number] = []
        
        pick_to_players[pick_number].append(name)
        
    except ValueError:
        pass  # skip bad rows if any

# Step 2: Calculate average VORP and average Composite Score for each draft pick number
avg_vorp_by_pick = {}
avg_composite_by_pick = {}

for pick_number in range(1, 61):  # Picks 1 through 60
    if pick_number in pick_to_players:
        vorps = []
        composites = []
        
        for player_name in pick_to_players[pick_number]:
            # Get VORP
            avg_vorp = get_avg_vorp(player_name)
            vorps.append(avg_vorp)
            
            # Get Composite Score
            composite_score = calculate_player_score(player_name)
            composites.append(composite_score)
        
        # Calculate averages
        if vorps:
            avg_vorp_by_pick[pick_number] = np.mean(vorps)
        else:
            avg_vorp_by_pick[pick_number] = 0
            
        if composites:
            avg_composite_by_pick[pick_number] = np.mean(composites)
        else:
            avg_composite_by_pick[pick_number] = 0
    else:
        avg_vorp_by_pick[pick_number] = 0
        avg_composite_by_pick[pick_number] = 0

# Step 3: Prepare data for plotting
x_vals = list(range(1, 61))
vorp_y_vals = [avg_vorp_by_pick[i] for i in x_vals]
composite_y_vals = [avg_composite_by_pick[i] for i in x_vals]

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Average VORP by Draft Pick (Discrete)
ax1.scatter(x_vals, vorp_y_vals, color='blue', s=50, alpha=0.7, label='VORP')
# Optional: Add a line connecting points to show trend
ax1.plot(x_vals, vorp_y_vals, color='lightblue', linestyle='--', alpha=0.5)
ax1.set_xlabel("Draft Pick Number", fontsize=12)
ax1.set_ylabel("Average Career VORP", fontsize=12)
ax1.set_title("Average VORP by Draft Pick", fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, 61, 5))  # Show every 5th pick on x-axis
ax1.legend()

# Plot 2: Average Composite Score by Draft Pick (Discrete)
ax2.scatter(x_vals, composite_y_vals, color='green', s=50, alpha=0.7, label='Composite Score')
ax2.plot(x_vals, composite_y_vals, color='lightgreen', linestyle='--', alpha=0.5)
ax2.set_xlabel("Draft Pick Number", fontsize=12)
ax2.set_ylabel("Average Composite Score", fontsize=12)
ax2.set_title("Average Composite Score by Draft Pick", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, 61, 5))  # Show every 5th pick on x-axis
ax2.legend()

plt.tight_layout()
plt.show()

# Optional: Create a combined overlay graph
fig2, ax3 = plt.subplots(figsize=(12, 7))

# Normalize both metrics to 0-1 scale for comparison
max_vorp = max(vorp_y_vals) if max(vorp_y_vals) > 0 else 1
max_composite = max(composite_y_vals) if max(composite_y_vals) > 0 else 1

vorp_normalized = [v / max_vorp for v in vorp_y_vals]
composite_normalized = [c / max_composite for c in composite_y_vals]

# Plot normalized values
ax3.scatter(x_vals, vorp_normalized, color='blue', s=60, alpha=0.7, label='VORP (normalized)', marker='o')
ax3.scatter(x_vals, composite_normalized, color='green', s=60, alpha=0.7, label='Composite Score (normalized)', marker='s')
ax3.plot(x_vals, vorp_normalized, color='lightblue', linestyle='--', alpha=0.5)
ax3.plot(x_vals, composite_normalized, color='lightgreen', linestyle='--', alpha=0.5)

ax3.set_xlabel("Draft Pick Number", fontsize=12)
ax3.set_ylabel("Normalized Score (0-1 scale)", fontsize=12)
ax3.set_title("Comparison: VORP vs Composite Score by Draft Pick", fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, 61, 5))
ax3.legend()

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("DRAFT PICK VALUE ANALYSIS")
print("="*60)
print(f"\nTop 5 Picks by VORP:")
top5_vorp = sorted(avg_vorp_by_pick.items(), key=lambda x: x[1], reverse=True)[:5]
for pick, value in top5_vorp:
    print(f"  Pick #{pick}: {value:.2f} VORP")

print(f"\nTop 5 Picks by Composite Score:")
top5_composite = sorted(avg_composite_by_pick.items(), key=lambda x: x[1], reverse=True)[:5]
for pick, value in top5_composite:
    print(f"  Pick #{pick}: {value:.3f}")

print(f"\nCorrelation between VORP and Composite Score rankings: {np.corrcoef(vorp_y_vals, composite_y_vals)[0,1]:.3f}")

### end graphing section ###