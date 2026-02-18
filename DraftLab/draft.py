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
                'VORP': 0.8,
                'TS%': 0.05,
                'TRB%': 0.05,
                'AST%': 0.05,
                'TOV%': 0.05
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

### Enhanced Graphing Section with Best Fit Curves ###

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

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

# Step 3: Prepare data for curve fitting
x_vals = np.array(list(range(1, 61)))
vorp_y_vals = np.array([avg_vorp_by_pick[i] for i in x_vals])
composite_y_vals = np.array([avg_composite_by_pick[i] for i in x_vals])

# Step 4: Define exponential decay function for fitting
def exponential_decay(x, a, b, c):
    """Exponential decay function: a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c

def power_law(x, a, b, c):
    """Power law function: a * x^(-b) + c (alternative fit)"""
    return a * np.power(x, -b) + c

# Step 5: Fit exponential curves to both datasets
# Filter out zero values for better fitting
valid_vorp = (vorp_y_vals > 0)
valid_composite = (composite_y_vals > 0)

try:
    # Fit VORP data
    popt_vorp_exp, pcov_vorp_exp = curve_fit(exponential_decay, 
                                             x_vals[valid_vorp], 
                                             vorp_y_vals[valid_vorp],
                                             p0=[10, 0.1, 0],  # Initial guess: [amplitude, decay rate, offset]
                                             maxfev=5000)
    
    # Generate fitted curve
    vorp_fitted_exp = exponential_decay(x_vals, *popt_vorp_exp)
    r2_vorp_exp = r2_score(vorp_y_vals[valid_vorp], vorp_fitted_exp[valid_vorp])
    
    # Fit Composite data
    popt_comp_exp, pcov_comp_exp = curve_fit(exponential_decay, 
                                             x_vals[valid_composite], 
                                             composite_y_vals[valid_composite],
                                             p0=[0.5, 0.1, 0],
                                             maxfev=5000)
    
    # Generate fitted curve
    comp_fitted_exp = exponential_decay(x_vals, *popt_comp_exp)
    r2_comp_exp = r2_score(composite_y_vals[valid_composite], comp_fitted_exp[valid_composite])
    
    fit_successful = True
except:
    fit_successful = False
    print("Warning: Exponential curve fitting failed. Using alternative method.")

# Step 6: Create enhanced visualizations
fig = plt.figure(figsize=(16, 12))

# Plot 1: VORP with Exponential Fit
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(x_vals, vorp_y_vals, color='blue', s=50, alpha=0.6, label='Actual VORP')
if fit_successful:
    ax1.plot(x_vals, vorp_fitted_exp, color='red', linewidth=3, 
             label=f'Exponential Fit (R²={r2_vorp_exp:.3f})')
ax1.set_xlabel("Draft Pick Number", fontsize=12)
ax1.set_ylabel("Average Career VORP", fontsize=12)
ax1.set_title("VORP by Draft Pick with Best Fit Curve", fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, 61, 5))
ax1.legend()

# Plot 2: Composite Score with Exponential Fit
ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(x_vals, composite_y_vals, color='green', s=50, alpha=0.6, label='Actual Composite')
if fit_successful:
    ax2.plot(x_vals, comp_fitted_exp, color='red', linewidth=3,
             label=f'Exponential Fit (R²={r2_comp_exp:.3f})')
ax2.set_xlabel("Draft Pick Number", fontsize=12)
ax2.set_ylabel("Average Composite Score", fontsize=12)
ax2.set_title("Composite Score by Draft Pick with Best Fit Curve", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, 61, 5))
ax2.legend()

# Plot 3: Combined view with normalized values
ax3 = fig.add_subplot(2, 2, 3)

# Normalize both metrics
max_vorp = max(vorp_y_vals) if max(vorp_y_vals) > 0 else 1
max_composite = max(composite_y_vals) if max(composite_y_vals) > 0 else 1

vorp_normalized = vorp_y_vals / max_vorp
composite_normalized = composite_y_vals / max_composite

ax3.scatter(x_vals, vorp_normalized, color='blue', s=40, alpha=0.5, label='VORP (norm)', marker='o')
ax3.scatter(x_vals, composite_normalized, color='green', s=40, alpha=0.5, label='Composite (norm)', marker='s')

if fit_successful:
    # Normalize fitted curves
    vorp_fitted_norm = vorp_fitted_exp / max_vorp
    comp_fitted_norm = comp_fitted_exp / max_composite
    ax3.plot(x_vals, vorp_fitted_norm, color='darkblue', linewidth=2.5, linestyle='--', label='VORP Fit')
    ax3.plot(x_vals, comp_fitted_norm, color='darkgreen', linewidth=2.5, linestyle='--', label='Composite Fit')

ax3.set_xlabel("Draft Pick Number", fontsize=12)
ax3.set_ylabel("Normalized Score", fontsize=12)
ax3.set_title("Comparison: VORP vs Composite (Normalized)", fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, 61, 5))
ax3.legend()

# Plot 4: Value Decay Rate Comparison
ax4 = fig.add_subplot(2, 2, 4)

if fit_successful:
    # Extract decay rates
    decay_rate_vorp = popt_vorp_exp[1]
    decay_rate_comp = popt_comp_exp[1]
    half_life_vorp = np.log(2) / decay_rate_vorp
    half_life_comp = np.log(2) / decay_rate_comp
    
    # Create bar chart of decay rates
    metrics = ['VORP', 'Composite']
    decay_rates = [decay_rate_vorp, decay_rate_comp]
    half_lives = [half_life_vorp, half_life_comp]
    
    colors = ['blue', 'green']
    bars = ax4.bar(metrics, decay_rates, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, rate, half in zip(bars, decay_rates, half_lives):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'Rate: {rate:.3f}\nHalf-life: {half:.1f} picks',
                ha='center', va='bottom', fontsize=10)
    
    ax4.set_ylabel("Decay Rate", fontsize=12)
    ax4.set_title("Value Decay Rate Comparison", fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
else:
    ax4.text(0.5, 0.5, "Curve fitting failed\nCannot compute decay rates",
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title("Value Decay Analysis (Failed)", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Print enhanced summary statistics with curve parameters
print("\n" + "="*70)
print("DRAFT PICK VALUE ANALYSIS WITH EXPONENTIAL FITTING")
print("="*70)

if fit_successful:
    print("\n📈 EXPONENTIAL FIT PARAMETERS:")
    print(f"   VORP: Value = {popt_vorp_exp[0]:.2f} × exp(-{popt_vorp_exp[1]:.3f} × pick) + {popt_vorp_exp[2]:.2f}")
    print(f"   Composite: Value = {popt_comp_exp[0]:.2f} × exp(-{popt_comp_exp[1]:.3f} × pick) + {popt_comp_exp[2]:.2f}")
    
    print("\n⏱️  VALUE DECAY METRICS:")
    print(f"   VORP half-life: {half_life_vorp:.1f} draft picks")
    print(f"   Composite half-life: {half_life_comp:.1f} draft picks")
    
    print("\n🎯 PREDICTED VALUES AT KEY DRAFT POSITIONS:")
    positions = [1, 5, 10, 15, 20, 30, 40, 50, 60]
    print(f"\n{'Pick':>6} | {'VORP':>10} | {'Composite':>12} | {'VORP % of #1':>14} | {'Comp % of #1':>14}")
    print("-" * 70)
    
    base_vorp = exponential_decay(1, *popt_vorp_exp)
    base_comp = exponential_decay(1, *popt_comp_exp)
    
    for pos in positions:
        pred_vorp = exponential_decay(pos, *popt_vorp_exp)
        pred_comp = exponential_decay(pos, *popt_comp_exp)
        vorp_pct = (pred_vorp / base_vorp) * 100
        comp_pct = (pred_comp / base_comp) * 100
        print(f"{pos:6d} | {pred_vorp:10.2f} | {pred_comp:12.3f} | {vorp_pct:13.1f}% | {comp_pct:13.1f}%")

print(f"\n📊 CORRELATION METRICS:")
print(f"   VORP R²: {r2_vorp_exp:.3f}" if fit_successful else "   VORP R²: N/A")
print(f"   Composite R²: {r2_comp_exp:.3f}" if fit_successful else "   Composite R²: N/A")
print(f"   Correlation between VORP and Composite: {np.corrcoef(vorp_y_vals, composite_y_vals)[0,1]:.3f}")

print("\n💡 INTERPRETATION:")
if fit_successful:
    if decay_rate_vorp > decay_rate_comp:
        print("   • VORP decays faster than Composite score")
        print("   • Your composite metric maintains value longer into the draft")
    else:
        print("   • Composite score decays faster than VORP")
        print("   • VORP maintains value longer into the draft")
    
    print(f"   • After {int(half_life_vorp)} picks, VORP value drops by 50%")
    print(f"   • After {int(half_life_comp)} picks, Composite value drops by 50%")

### end enhanced graphing section ###