from lxml import html
import requests
import re
import os
import time

# Configuration - set to True to use local HTML files, False to scrape live
USE_LOCAL_FILES = False

# Site to scrape - NBA season stats from 2015 to 2025
base_url = "https://www.basketball-reference.com"

# Years to scrape (2015 through 2025)
years = range(2015, 2026)  # 2015, 2016, ..., 2025

print("Scraping NBA season data from 2015-2025...")
print("=" * 60)

# Store all teams data across all seasons
all_teams_data = {}

for year in years:
    page = f"{base_url}/leagues/NBA_{year}.html"
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {year-1}-{str(year)[2:]} SEASON")
    print(f"{'='*60}")
    
    if USE_LOCAL_FILES:
        filename = f'nba_{year}_main.html'
        print(f"\nUsing locally saved HTML file: {filename}")
        
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = html.fromstring(content.encode('utf-8'))
        else:
            print(f"WARNING: {filename} not found! Skipping {year} season.")
            print(f"To include this season, save the page from {page}")
            print(f"as '{filename}' in this directory.")
            continue
    else:
        print(f"\nScraping live from website: {page}")
        try:
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            result = requests.get(page, headers=headers)
            result.raise_for_status()
            tree = html.fromstring(result.content)
            
            # Be respectful - wait between requests
            if year < years[-1]:  # Don't wait after the last one
                time.sleep(3)
                
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Failed to fetch page for {year}: {e}")
            print(f"\nTip: Save the page manually:")
            print(f"1. Visit {page} in your browser")
            print(f"2. Save as 'nba_{year}_main.html'")
            print(f"3. Set USE_LOCAL_FILES = True in this script")
            continue
    
    # Get team standings for this season
    print(f"  Scraping team standings for {year}...")
    
    season_teams = 0
    
    # Get both conference standings
    for conf in ['E', 'W']:
        table_id = f'confs_standings_{conf}'
        standings_table = tree.xpath(f'//table[@id="{table_id}"]')
        
        if standings_table:
            rows = standings_table[0].xpath('.//tbody/tr[@class="full_table"]')
            
            for row in rows:
                # Get team name
                team_name_elem = row.xpath('.//th[@data-stat="team_name"]/a')
                if team_name_elem:
                    team_name = team_name_elem[0].text_content().strip()
                    
                    # Create unique key: "Team Name - Year"
                    team_key = f"{team_name} - {year}"
                    
                    # Get wins and losses
                    wins = float(row.xpath('.//td[@data-stat="wins"]')[0].text_content())
                    losses = float(row.xpath('.//td[@data-stat="losses"]')[0].text_content())
                    win_pct = float(row.xpath('.//td[@data-stat="win_loss_pct"]')[0].text_content())
                    
                    # Calculate win rate from wins and losses
                    total_games = wins + losses
                    calculated_win_rate = wins / total_games if total_games > 0 else 0
                    
                    all_teams_data[team_key] = {
                        'team_name': team_name,
                        'year': year,
                        'wins': wins,
                        'losses': losses,
                        'win_pct': win_pct,
                        'calculated_win_rate': calculated_win_rate,
                        'team_3pa': None,
                        'opp_3pa': None
                    }
                    season_teams += 1
    
    print(f"  Found {season_teams} teams")
    
    # Get 3-point attempt data from TOTALS tables
    print(f"  Scraping team shooting statistics (TOTALS)...")
    
    # Get the totals team stats table
    stats_table = tree.xpath('//table[@id="totals-team"]')
    if stats_table:
        rows = stats_table[0].xpath('.//tbody/tr')
        
        for row in rows:
            # Skip header rows
            if row.get('class') and 'thead' in row.get('class'):
                continue
            
            # Get team name from any link to /teams/ and strip asterisk
            team_name_elem = row.xpath('.//a[contains(@href, "/teams/")]')
            if team_name_elem:
                team_name = team_name_elem[0].text_content().strip().rstrip('*')
                team_key = f"{team_name} - {year}"
                
                # Get total 3-point attempts
                three_pa_elem = row.xpath('.//td[@data-stat="fg3a"]')
                if three_pa_elem and team_key in all_teams_data:
                    all_teams_data[team_key]['team_3pa'] = float(three_pa_elem[0].text_content())
    
    # Get opponent totals table
    opp_stats_table = tree.xpath('//table[@id="totals-opponent"]')
    if opp_stats_table:
        rows = opp_stats_table[0].xpath('.//tbody/tr')
        
        for row in rows:
            # Skip header rows
            if row.get('class') and 'thead' in row.get('class'):
                continue
            
            # Get team name from any link to /teams/ and strip asterisk
            team_name_elem = row.xpath('.//a[contains(@href, "/teams/")]')
            if team_name_elem:
                team_name = team_name_elem[0].text_content().strip().rstrip('*')
                team_key = f"{team_name} - {year}"
                
                # Get total opponent 3-point attempts
                opp_three_pa_elem = row.xpath('.//td[@data-stat="opp_fg3a"]')
                if opp_three_pa_elem and team_key in all_teams_data:
                    all_teams_data[team_key]['opp_3pa'] = float(opp_three_pa_elem[0].text_content())
    
    print(f"  ✓ Completed {year} season")

# Filter out teams with incomplete data
complete_teams = {key: data for key, data in all_teams_data.items() 
                  if data['team_3pa'] is not None and data['opp_3pa'] is not None}

print("\n" + "=" * 60)
print("DATA COLLECTION SUMMARY")
print("=" * 60)
print(f"Total team-seasons collected: {len(all_teams_data)}")
print(f"Complete data available: {len(complete_teams)}")
print(f"Years covered: {min([d['year'] for d in complete_teams.values()])} - {max([d['year'] for d in complete_teams.values()])}")

# Check if we have data before proceeding
if len(complete_teams) == 0:
    print("\n⚠️  ERROR: No complete team data found!")
    print("\nMake sure you have the HTML files saved or can scrape live.")
    exit(1)

# Show sample of data
print("\nSample of collected data (first 5):")
print("-" * 100)
print(f"{'Team-Season':<40} {'Wins':>5} {'Losses':>5} {'Win %':>8} {'Total 3PA':>11} {'Opp 3PA':>10}")
print("-" * 100)
for i, (team_key, data) in enumerate(sorted(complete_teams.items())):
    if i >= 5:
        break
    print(f"{team_key:<40} {data['wins']:>5.0f} {data['losses']:>5.0f} "
          f"{data['win_pct']:>8.3f} {data['team_3pa']:>11.0f} {data['opp_3pa']:>10.0f}")
print(f"... and {len(complete_teams) - 5} more team-seasons")

# BAYESIAN ANALYSIS
# Event A: Team has above-average win rate (win_pct > 0.500)
# Event B: Team attempts more 3-pointers than opponents (team_3pa > opp_3pa)

print("\n" + "=" * 60)
print("BAYESIAN ANALYSIS (2015-2025 COMBINED)")
print("=" * 60)

# Calculate counts
total_teams = len(complete_teams)
above_avg_teams = sum(1 for data in complete_teams.values() if data['win_pct'] > 0.500)
more_3pa_teams = sum(1 for data in complete_teams.values() if data['team_3pa'] > data['opp_3pa'])
both_conditions = sum(1 for data in complete_teams.values() 
                     if data['win_pct'] > 0.500 and data['team_3pa'] > data['opp_3pa'])

# Calculate probabilities
# P(A) = Prior probability of above-average win rate
p_a = above_avg_teams / total_teams

# P(B) = Probability of attempting more 3-pointers than opponents
p_b = more_3pa_teams / total_teams

# P(B|A) = Likelihood: Probability of more 3PA given above-average record
if above_avg_teams > 0:
    p_b_given_a = both_conditions / above_avg_teams
else:
    p_b_given_a = 0

# P(A|B) = Posterior: Probability of above-average record given more 3PA
# Using Bayes Theorem: P(A|B) = P(B|A) * P(A) / P(B)
if p_b > 0:
    p_a_given_b = (p_b_given_a * p_a) / p_b
else:
    p_a_given_b = 0

# Print results
print(f"\nEvent A: Team has above-average win rate (W/L% > 0.500)")
print(f"Event B: Team attempts more 3-pointers than opponents (TOTAL season)")
print()
print(f"Total team-seasons analyzed: {total_teams}")
print(f"Team-seasons with above-average record: {above_avg_teams}")
print(f"Team-seasons attempting more 3PA than opponents: {more_3pa_teams}")
print(f"Team-seasons meeting both conditions: {both_conditions}")
print()
print("-" * 60)
print("BAYESIAN THEOREM CALCULATION:")
print("-" * 60)
print(f"P(A) [Prior]:           {p_a:.4f} ({above_avg_teams}/{total_teams})")
print(f"P(B):                   {p_b:.4f} ({more_3pa_teams}/{total_teams})")
print(f"P(B|A) [Likelihood]:    {p_b_given_a:.4f} ({both_conditions}/{above_avg_teams})")
print(f"P(A|B) [Posterior]:     {p_a_given_b:.4f}")
print()
print("Bayes Theorem: P(A|B) = P(B|A) × P(A) / P(B)")
print(f"             = {p_b_given_a:.4f} × {p_a:.4f} / {p_b:.4f}")
print(f"             = {p_a_given_b:.4f}")
print()

# Interpretation
print("=" * 60)
print("INTERPRETATION:")
print("=" * 60)
if p_a_given_b > p_a:
    difference = p_a_given_b - p_a
    print(f"The posterior probability ({p_a_given_b:.4f}) is HIGHER than the prior ({p_a:.4f})")
    print(f"by {difference:.4f} ({difference/p_a*100:.1f}% increase).")
    print()
    print("Over the 2015-2025 seasons, attempting more 3-pointers than opponents")
    print("is positively associated with having an above-average win rate.")
elif p_a_given_b < p_a:
    difference = p_a - p_a_given_b
    print(f"The posterior probability ({p_a_given_b:.4f}) is LOWER than the prior ({p_a:.4f})")
    print(f"by {difference:.4f} ({difference/p_a*100:.1f}% decrease).")
    print()
    print("Over the 2015-2025 seasons, attempting more 3-pointers than opponents")
    print("is negatively associated with having an above-average win rate.")
else:
    print(f"The posterior probability equals the prior probability ({p_a:.4f}).")
    print()
    print("Over the 2015-2025 seasons, 3-point attempt differential has no")
    print("relationship with having an above-average win rate.")

print()
print("ACTIONABLE INSIGHT:")
print("-" * 60)
if p_a_given_b > p_a:
    print("Based on 11 seasons of NBA data (2015-2025), teams looking to improve")
    print("their record should consider increasing their 3-point attempt rate")
    print("relative to their opponents. This data suggests a consistent correlation")
    print("between offensive aggressiveness from beyond the arc and winning basketball.")
else:
    print("Based on 11 seasons of NBA data (2015-2025), simply attempting more")
    print("3-pointers than opponents does not appear to be a strong predictor of")
    print("team success. Teams should focus on other factors like defensive")
    print("efficiency, shot selection quality, or overall team composition.")

# Year-by-year breakdown
print("\n" + "=" * 60)
print("YEAR-BY-YEAR BREAKDOWN")
print("=" * 60)

year_stats = {}
for team_key, data in complete_teams.items():
    year = data['year']
    if year not in year_stats:
        year_stats[year] = {
            'total': 0,
            'above_avg': 0,
            'more_3pa': 0,
            'both': 0
        }
    
    year_stats[year]['total'] += 1
    if data['win_pct'] > 0.500:
        year_stats[year]['above_avg'] += 1
    if data['team_3pa'] > data['opp_3pa']:
        year_stats[year]['more_3pa'] += 1
    if data['win_pct'] > 0.500 and data['team_3pa'] > data['opp_3pa']:
        year_stats[year]['both'] += 1

print()
print(f"{'Year':<10} {'Teams':>8} {'Above Avg':>12} {'More 3PA':>12} {'Both':>8} {'P(A|B)':>10}")
print("-" * 70)
for year in sorted(year_stats.keys()):
    stats = year_stats[year]
    p_b_year = stats['more_3pa'] / stats['total'] if stats['total'] > 0 else 0
    p_a_year = stats['above_avg'] / stats['total'] if stats['total'] > 0 else 0
    p_b_given_a_year = stats['both'] / stats['above_avg'] if stats['above_avg'] > 0 else 0
    p_a_given_b_year = (p_b_given_a_year * p_a_year) / p_b_year if p_b_year > 0 else 0
    
    print(f"{year:<10} {stats['total']:>8} {stats['above_avg']:>12} {stats['more_3pa']:>12} "
          f"{stats['both']:>8} {p_a_given_b_year:>10.4f}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nTotal team-seasons analyzed: {total_teams}")
print(f"Years covered: {min(year_stats.keys())} - {max(year_stats.keys())}")
print(f"\nOverall P(A|B): {p_a_given_b:.4f}")

# Export to CSV
import csv

print(f"\n{'='*60}")
print(f"EXPORTING DATA TO CSV")
print(f"{'='*60}")

# 1. Export detailed team data
csv_filename = 'nba_teams_2015_2025.csv'
print(f"\n1. Saving detailed team data to: {csv_filename}")

with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['team_name', 'year', 'season', 'wins', 'losses', 'win_pct', 
                  'calculated_win_rate', 'team_3pa', 'opp_3pa', 'three_pa_differential',
                  'above_avg_record', 'more_3pa_than_opp', 'both_conditions']
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for team_key in sorted(complete_teams.keys()):
        data = complete_teams[team_key]
        
        # Calculate derived fields
        three_pa_diff = data['team_3pa'] - data['opp_3pa']
        above_avg = 1 if data['win_pct'] > 0.500 else 0
        more_3pa = 1 if data['team_3pa'] > data['opp_3pa'] else 0
        both = 1 if (data['win_pct'] > 0.500 and data['team_3pa'] > data['opp_3pa']) else 0
        
        writer.writerow({
            'team_name': data['team_name'],
            'year': data['year'],
            'season': f"{data['year']-1}-{str(data['year'])[2:]}",
            'wins': int(data['wins']),
            'losses': int(data['losses']),
            'win_pct': data['win_pct'],
            'calculated_win_rate': data['calculated_win_rate'],
            'team_3pa': int(data['team_3pa']),
            'opp_3pa': int(data['opp_3pa']),
            'three_pa_differential': int(three_pa_diff),
            'above_avg_record': above_avg,
            'more_3pa_than_opp': more_3pa,
            'both_conditions': both
        })

print(f"   ✓ Exported {len(complete_teams)} team-seasons")

# 2. Export Bayesian analysis summary (ALL YEARS COMBINED)
summary_filename = 'bayesian_analysis_summary.csv'
print(f"\n2. Saving Bayesian analysis summary to: {summary_filename}")

with open(summary_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['analysis_period', 'event_a', 'event_b', 'total_teams', 'above_avg_teams', 
                  'more_3pa_teams', 'both_conditions', 'p_a_prior', 'p_b', 
                  'p_b_given_a_likelihood', 'p_a_given_b_posterior', 
                  'posterior_vs_prior_difference', 'posterior_vs_prior_pct_change']
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Define events
    event_a_definition = "Team has above-average win rate (W/L% > 0.500)"
    event_b_definition = "Team attempts more 3-pointers than opponents"
    
    # Overall analysis (2015-2025 combined)
    posterior_prior_diff = p_a_given_b - p_a
    posterior_prior_pct = (posterior_prior_diff / p_a * 100) if p_a > 0 else 0
    
    writer.writerow({
        'analysis_period': '2015-2025 (All Years Combined)',
        'event_a': event_a_definition,
        'event_b': event_b_definition,
        'total_teams': total_teams,
        'above_avg_teams': above_avg_teams,
        'more_3pa_teams': more_3pa_teams,
        'both_conditions': both_conditions,
        'p_a_prior': round(p_a, 4),
        'p_b': round(p_b, 4),
        'p_b_given_a_likelihood': round(p_b_given_a, 4),
        'p_a_given_b_posterior': round(p_a_given_b, 4),
        'posterior_vs_prior_difference': round(posterior_prior_diff, 4),
        'posterior_vs_prior_pct_change': round(posterior_prior_pct, 2)
    })
    
    # Year-by-year breakdown
    for year in sorted(year_stats.keys()):
        stats = year_stats[year]
        p_b_year = stats['more_3pa'] / stats['total'] if stats['total'] > 0 else 0
        p_a_year = stats['above_avg'] / stats['total'] if stats['total'] > 0 else 0
        p_b_given_a_year = stats['both'] / stats['above_avg'] if stats['above_avg'] > 0 else 0
        p_a_given_b_year = (p_b_given_a_year * p_a_year) / p_b_year if p_b_year > 0 else 0
        
        posterior_prior_diff_year = p_a_given_b_year - p_a_year
        posterior_prior_pct_year = (posterior_prior_diff_year / p_a_year * 100) if p_a_year > 0 else 0
        
        writer.writerow({
            'analysis_period': f'{year}',
            'event_a': event_a_definition,
            'event_b': event_b_definition,
            'total_teams': stats['total'],
            'above_avg_teams': stats['above_avg'],
            'more_3pa_teams': stats['more_3pa'],
            'both_conditions': stats['both'],
            'p_a_prior': round(p_a_year, 4),
            'p_b': round(p_b_year, 4),
            'p_b_given_a_likelihood': round(p_b_given_a_year, 4),
            'p_a_given_b_posterior': round(p_a_given_b_year, 4),
            'posterior_vs_prior_difference': round(posterior_prior_diff_year, 4),
            'posterior_vs_prior_pct_change': round(posterior_prior_pct_year, 2)
        })

print(f"   ✓ Exported overall analysis + {len(year_stats)} year-by-year breakdowns")

print(f"\n{'='*60}")
print("CSV EXPORT COMPLETE")
print(f"{'='*60}")
print(f"\nFiles created:")
print(f"  1. {csv_filename} - Detailed team data ({len(complete_teams)} rows)")
print(f"  2. {summary_filename} - Bayesian analysis summary ({1 + len(year_stats)} rows)")
print(f"\n{'='*60}")