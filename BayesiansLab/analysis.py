import pandas as pd

# Load the data
df = pd.read_csv('nba_team_data.csv')

# Calculate basic counts
total_teams = len(df)
teams_above_avg = df['Above_Average_Record'].sum() if df['Above_Average_Record'].dtype == 'bool' else (df['Above_Average_Record'] == 'Yes').sum()
teams_more_3pa = df['More_3PA_Than_Opponent'].sum() if df['More_3PA_Than_Opponent'].dtype == 'bool' else (df['More_3PA_Than_Opponent'] == 'Yes').sum()

# Calculate joint probabilities
both_conditions = ((df['Above_Average_Record'] == 'Yes') & (df['More_3PA_Than_Opponent'] == 'Yes')).sum()
only_above_avg = ((df['Above_Average_Record'] == 'Yes') & (df['More_3PA_Than_Opponent'] == 'No')).sum()
only_more_3pa = ((df['Above_Average_Record'] == 'No') & (df['More_3PA_Than_Opponent'] == 'Yes')).sum()
neither = ((df['Above_Average_Record'] == 'No') & (df['More_3PA_Than_Opponent'] == 'No')).sum()

# Calculate probabilities
P_A = teams_above_avg / total_teams  # P(Above Average Record)
P_B = teams_more_3pa / total_teams   # P(More 3PA than Opponent)
P_A_and_B = both_conditions / total_teams  # P(Above Average AND More 3PA)
P_A_given_B = P_A_and_B / P_B if P_B > 0 else 0  # P(A|B)
P_B_given_A = P_A_and_B / P_A if P_A > 0 else 0  # P(B|A)

# Bayes' Theorem: P(A|B) = [P(B|A) * P(A)] / P(B)
# This should equal our calculated P_A_given_B
bayes_P_A_given_B = (P_B_given_A * P_A) / P_B if P_B > 0 else 0

# Calculate odds ratios
odds_ratio = (P_A_and_B * neither) / (only_above_avg * only_more_3pa) if (only_above_avg * only_more_3pa) > 0 else 0

# Create contingency table
contingency_table = pd.DataFrame({
    'More 3PA = Yes': [both_conditions, only_more_3pa, teams_more_3pa],
    'More 3PA = No': [only_above_avg, neither, total_teams - teams_more_3pa],
    'Total': [teams_above_avg, total_teams - teams_above_avg, total_teams]
}, index=['Above Avg = Yes', 'Above Avg = No', 'Total'])

# Print results
print("=" * 60)
print("NBA TEAM ANALYSIS: 3PA DIFFERENTIAL vs WINNING RECORDS")
print("=" * 60)
print(f"\nTotal Teams Analyzed: {total_teams}")
print(f"Teams with Above Average Records: {teams_above_avg}")
print(f"Teams with More 3PA than Opponent: {teams_more_3pa}")
print(f"Teams with BOTH conditions: {both_conditions}")

print("\n" + "=" * 60)
print("CONTINGENCY TABLE")
print("=" * 60)
print(contingency_table)

print("\n" + "=" * 60)
print("PROBABILITY CALCULATIONS")
print("=" * 60)
print(f"P(Above Average Record): {P_A:.3f} ({teams_above_avg}/{total_teams})")
print(f"P(More 3PA than Opponent): {P_B:.3f} ({teams_more_3pa}/{total_teams})")
print(f"P(Above Average AND More 3PA): {P_A_and_B:.3f} ({both_conditions}/{total_teams})")

print(f"\nP(Above Average | More 3PA): {P_A_given_B:.3f}")
print(f"P(More 3PA | Above Average): {P_B_given_A:.3f}")

print(f"\nBayes' Theorem Verification:")
print(f"Calculated P(A|B): {P_A_given_B:.3f}")
print(f"Bayes' Theorem P(A|B): {bayes_P_A_given_B:.3f}")
print(f"Match: {'YES' if abs(P_A_given_B - bayes_P_A_given_B) < 0.001 else 'NO'}")

print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)

# Determine if more 3PA correlates with winning
if P_A_given_B > P_A:
    print("✓ EVIDENCE OF POSITIVE CORRELATION")
    print(f"  Teams that shoot more 3PA have a {P_A_given_B:.1%} chance of having an above average record,")
    print(f"  compared to the baseline of {P_A:.1%} for all teams.")
    print(f"  This is {P_A_given_B/P_A:.2f}x more likely.")
else:
    print("✗ NO EVIDENCE OF POSITIVE CORRELATION")
    print(f"  Teams that shoot more 3PA have a {P_A_given_B:.1%} chance of having an above average record,")
    print(f"  compared to the baseline of {P_A:.1%} for all teams.")

print(f"\nOdds Ratio: {odds_ratio:.2f}")
if odds_ratio > 1:
    print(f"Teams with more 3PA are {odds_ratio:.2f} times more likely to have above average records.")
elif odds_ratio < 1:
    print(f"Teams with more 3PA are {1/odds_ratio:.2f} times LESS likely to have above average records.")
else:
    print("No association between 3PA differential and winning records.")

# Calculate additional statistics
print("\n" + "=" * 60)
print("ADDITIONAL STATISTICS")
print("=" * 60)

# Success rates for each group
success_rate_more_3pa = both_conditions / teams_more_3pa if teams_more_3pa > 0 else 0
success_rate_less_3pa = only_above_avg / (total_teams - teams_more_3pa) if (total_teams - teams_more_3pa) > 0 else 0

print(f"Success rate for teams with MORE 3PA than opponent: {success_rate_more_3pa:.1%}")
print(f"Success rate for teams with FEWER 3PA than opponent: {success_rate_less_3pa:.1%}")

# List teams in each category
print("\nTeams with BOTH Above Average Record AND More 3PA:")
for team in df[(df['Above_Average_Record'] == 'Yes') & (df['More_3PA_Than_Opponent'] == 'Yes')]['Team']:
    print(f"  - {team}")

print("\nTeams with Above Average Record but FEWER 3PA:")
for team in df[(df['Above_Average_Record'] == 'Yes') & (df['More_3PA_Than_Opponent'] == 'No')]['Team']:
    print(f"  - {team}")

# Save detailed results to CSV
results_df = pd.DataFrame({
    'Metric': [
        'Total Teams',
        'Teams with Above Average Record',
        'Teams with More 3PA than Opponent',
        'Teams with Both Conditions',
        'P(Above Average)',
        'P(More 3PA)',
        'P(Above Average ∩ More 3PA)',
        'P(Above Average | More 3PA)',
        'P(More 3PA | Above Average)',
        'Odds Ratio',
        'Success Rate (More 3PA)',
        'Success Rate (Fewer 3PA)'
    ],
    'Value': [
        total_teams,
        teams_above_avg,
        teams_more_3pa,
        both_conditions,
        P_A,
        P_B,
        P_A_and_B,
        P_A_given_B,
        P_B_given_A,
        odds_ratio,
        success_rate_more_3pa,
        success_rate_less_3pa
    ]
})

results_df.to_csv('bayesian_analysis_results.csv', index=False)
print("\nDetailed results saved to 'bayesian_analysis_results.csv'")