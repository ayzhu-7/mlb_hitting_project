import pandas as pd
import numpy as np
import os
from pybaseball import statcast, pitching_stats, batting_stats, playerid_lookup, statcast_batter, statcast_pitcher
import statsapi
from datetime import date
import requests
from bs4 import BeautifulSoup

# kershaw_id = playerid_lookup('kershaw', 'clayton')
# kershaw_stats = statcast_pitcher('2017-06-01', '2017-07-01', kershaw_id['key_mlbam'][0])
# print(kershaw_stats)
# print(kershaw_stats.columns)
# print("here")

def squared_up(launch_velocity, bat_speed, pitch_speed):
    pitch_speed_home = 0.92 * pitch_speed
    max_theo = 1.23 * bat_speed + 0.23 * pitch_speed_home
    return launch_velocity / max_theo if max_theo > 0 else 0

def get_team_mapping():
    standings = statsapi.standings_data()
    team_mapping = {}
    for k in standings:
        for team in standings[k]['teams']:
            name = team['name']
            split_name = name.split(' ')
            if name in ['Chicago White Sox','Boston Red Sox','Toronto Blue Jays']:
                team_mapping[' '.join(split_name[-2:])] = team['team_id']
            else:
                team_mapping[split_name[-1]] = team['team_id']
    return team_mapping


def get_blast(launch_velocity, bat_speed, pitch_speed):
    """
    Calculate the blast score based on launch velocity, bat speed, and pitch speed.
    
    Args:
    launch_velocity (float): The speed of the ball off the bat in mph.
    bat_speed (float): The speed of the bat at impact in mph.
    pitch_speed (float): The speed of the pitch in mph.
    
    Returns:
    int: The blast score (1 if the condition is met, 0 otherwise).
    """
    result = 100 * squared_up(launch_velocity, bat_speed, pitch_speed) + bat_speed >= 164
    return int(result)

def is_barrel(exit_velocity, launch_angle):
    """
    Determines if a batted ball is classified as a "barrel" based on exit velocity and launch angle.

    Args:
    exit_velocity (float): The speed of the ball off the bat in miles per hour.
    launch_angle (float): The vertical angle at which the ball leaves the bat in degrees.

    Returns:
    int: 1 if the batted ball is a barrel, 0 otherwise.
    """
    barrel_dict = {98: [26, 30],
               99: [25, 31],
               100: [24, 33],
               101: [23, 34],
               102: [22, 35],
               103: [21, 36],
               104: [20, 37],
               105: [19, 38],
               106: [18, 39],
               107: [17, 40],
               108: [16, 41],
               109: [15, 42],
               110: [14, 43],
               111: [13, 44],
               112: [12, 45],
               113: [11, 46],
               114: [10, 47],
               115: [9, 48],
               116: [8, 50]}
    
    if pd.isna(exit_velocity) or pd.isna(launch_angle):
        return 0

    if exit_velocity < 98:
        print("a")
        return 0

    if exit_velocity >= 116:
        print("b")
        return 1 if (8 <= launch_angle <= 50) else 0

    exit_velocity = int(exit_velocity)
    lower_bound, upper_bound = barrel_dict[exit_velocity]

    if (lower_bound <= launch_angle) and (launch_angle <= upper_bound):
        print("c")
        return 1
    else:
        print("d")
        return 0


def barrels_by_side(df, side):
    """
    Filters the DataFrame for batters and calculates barrels.
    
    Args:
    df (DataFrame): The DataFrame containing batted ball data.
    
    Returns:
    Barrels and total attempts for the specified side.
    """
    side_df = df[df['stand'] == side].copy()
    # side_df['barrel'] = side_df.apply(lambda row: is_barrel(row['launch_speed'], row['launch_angle']), axis=1)
    return side_df['is_barrel'].sum(), side_df.shape[0]

def code_barrel(df):
    condition = (
        (df['launch_angle'] <= 50) &
        (df['launch_speed'] >= 98) &
        (df['launch_speed'] * 1.5 - df['launch_angle'] >= 117) &
        (df['launch_speed'] + df['launch_angle'] >= 123)
    )
    df['is_barrel'] = condition.astype(int)
    return df

def check_barrel(angle, speed):
    condition = (
        (angle <= 50) &
        (speed >= 98) &
        (speed * 1.5 - angle >= 117) &
        (speed + angle >= 123)
    )
    if condition: return 1
    else: return 0

def new_barrel(angle, speed):
    condition = (
        (angle <= 50) &
        (speed >= 98) &
        (speed * 1.5 - angle >= 117) &
        (speed + angle >= 123)
    )
    if condition: return 1
    else: return 0

def get_starting_pitchers(days_offset=0):
    today = (date.today() + pd.DateOffset(days=days_offset)).strftime('%Y-%m-%d')

    games = statsapi.schedule(start_date=today, end_date=today)

    pitchers = []

    # Loop through games and extract probable pitchers
    for game in games:
        home_team = game['home_name']
        away_team = game['away_name']
        home_pitcher = game.get('home_probable_pitcher', 'TBD')
        away_pitcher = game.get('away_probable_pitcher', 'TBD')
        
        pitchers.append([home_pitcher, 'away', game['game_id']])
        pitchers.append([away_pitcher, 'home', game['game_id']])
    return pitchers

def get_starting_ids(days_offset=0):
    todays_pitchers = get_starting_pitchers(days_offset=days_offset)
    print([p[0] for p in todays_pitchers])
    pitcher_ids = []

    for p in todays_pitchers:
        # print("a")
        if p[0] != '':
            # print(p[0])
            first_name, last_name = p[0].split(' ')[0], ' '.join(p[0].split(' ')[1:])
            # print('b')
            player_id = playerid_lookup(last_name, first_name, fuzzy=True)
            # player_id = player_id[player_id['mlb_played_last']==2025.0]
            # print(player_id)
            player_id = player_id['key_mlbam'].values[0]
            pitcher_ids.append([player_id, p[1], p[2]])  # Append player_id, team, and game_id
    return pitcher_ids

def get_barrel_percentage(player_id, start_date, end_date):
    player_barrels = statcast_pitcher(start_dt=start_date, end_dt=end_date,player_id=player_id)

    if player_barrels.empty:
        return 0
    # print(player_barrels['player_name'].iloc[0])
    player_bbe = player_barrels[(player_barrels['description'].isin(['hit_into_play']))].copy()
    player_bbe = code_barrel(player_bbe)
    left_barrels, left_bbes = barrels_by_side(player_bbe, 'L')
    right_barrels, right_bbes = barrels_by_side(player_bbe, 'R')
    # return (left_barrels + right_barrels) / (left_bbes + right_bbes) if (left_bbes + right_bbes) > 0 else 0
    total_barrels = left_barrels + right_barrels
    total_bbes = left_bbes + right_bbes
    barrel_dict = {'left_barrels': left_barrels,
                   'right_barrels': right_barrels,
                   'total_barrels': total_barrels,
                   'left_bbes': left_bbes,
                   'right_bbes': right_bbes,
                   'total_bbes': total_bbes,
                   'left_pct': left_barrels / left_bbes if left_bbes > 0 else 0,
                   'right_pct': right_barrels / right_bbes if right_bbes > 0 else 0,
                   'total_pct': total_barrels / total_bbes if total_bbes > 0 else 0}
    # return [total_barrels, total_bbes, total_barrels / total_bbes if total_bbes > 0 else 0]
    return barrel_dict

def hitter_barrel_percentage(player_id, start_date, end_date, pitch_mix, sides):
    player_barrels = statcast_batter(start_dt=start_date, end_dt=end_date,player_id=player_id)
    barrel_dict = {}

    for side in sides:
        if player_barrels.empty:
            temp_barrel_dict = {'total_barrels': 0, 'total_bbes': 0, 'barrel_pct': 0}
        else:
            player_bbe = player_barrels[(player_barrels['description'].isin(['hit_into_play'])) & (player_barrels['stand'] == side)].copy()
            player_bbe = code_barrel(player_bbe)
            player_bbe['is_blast'] = player_bbe.apply(lambda row: get_blast(row['launch_speed'], row['bat_speed'],row['release_speed']), axis=1)
            # player_bbe = player_bbe[player_bbe['pitch_name'].isin(pitch_mix[side].keys())]
            total_barrels = player_bbe['is_barrel'].sum()
            total_bbes = player_bbe.shape[0]
            temp_barrel_dict = {'total_barrels': total_barrels,
                        'total_bbes': total_bbes,
                        'barrel_pct': total_barrels / total_bbes if total_bbes > 0 else 0,
                        'blast_pct': player_bbe['is_blast'].sum() / total_bbes if total_bbes > 0 else 0}
        barrel_dict[side] = temp_barrel_dict
    return barrel_dict

def get_barrel_pcts(barrel_dict):
    barrel_pcts = []
    for side in barrel_dict:
        if barrel_dict[side]['total_bbes'] > 0:
            barrel_pcts.append([barrel_dict[side]['barrel_pct'], barrel_dict[side]['blast_pct']])
    return barrel_pcts

def check_barrel_pcts(barrel_pcts):
    """
    Check if barrel percentages are above a certain threshold.
    
    Args:
    barrel_pcts (list): List of barrel percentages for each side.
    
    Returns:
    bool: True if any barrel percentage is above 0.07, False otherwise.
    """
    # print(type(barrel_pcts))
    if len(barrel_pcts) == 0:
        return [0, 0]
    elif len(barrel_pcts) == 1:
        if barrel_pcts[0][0] > 0.1:
            return barrel_pcts[0]
        else:
            return [0, 0]
    if barrel_pcts[0][0] < 0.1 and barrel_pcts[1][0] < 0.1:
        return [0, 0]
    elif barrel_pcts[0][0] > barrel_pcts[1][0]:
        return barrel_pcts[0]
    else:
        return barrel_pcts[1]
    

def get_pitch_arsenal(pitcher_data, stand):
    if pitcher_data.empty:
        return {}

    # Filter for pitches thrown
    pitches = pitcher_data[pitcher_data['stand'] == stand]

    arsenal_vs_side = pitches['pitch_type'].value_counts().reset_index()
    arsenal_vs_side.columns = ['pitch_type', 'count']

    # Optional: Add % usage
    arsenal_vs_side['percent'] = 100 * arsenal_vs_side['count'] / arsenal_vs_side['count'].sum()

    arsenal_vs_side = arsenal_vs_side[arsenal_vs_side['percent'] > 18]

    pitch_percent_dict = dict(zip(arsenal_vs_side['pitch_type'], arsenal_vs_side['percent']))

    return pitch_percent_dict

def get_sides(left, right):
    sides = []
    if left > 0.07:
        sides.append('L')
    if right > 0.07:
        sides.append('R')
    return sides

def get_opposing_lineup(game_id, opponent):
    """
    Fetch the opposing team's lineup for a given game ID.
    """
    try:
        boxscore = statsapi.boxscore_data(game_id)
        lineup_ids = []
        for i in range(1, 10):
            player = boxscore[opponent + 'Batters'][i]
            if player:
                lineup_ids.append(player['personId'])
        return lineup_ids
    except Exception as e:
        print(f"Error fetching lineup for game {game_id}: {e}")
        return None

def get_team(game_id, team):
    game_data = statsapi.boxscore_data(game_id)
    if team == 'home':
        return game_data['teamInfo']['home']['teamName']
    else:
        return game_data['teamInfo']['away']['teamName']

def get_start_time(game_id):
    game_data = statsapi.boxscore_data(game_id)
    return game_data['gameBoxInfo'][-3]['value']

def get_rotowire_lineups():
    url = "https://www.rotowire.com/baseball/daily-lineups.php"
    soup = BeautifulSoup(requests.get(url).content, "html.parser")

    data_pitching = []
    data_batter = []
    team_type = ''

    for e in soup.select('.lineup__box ul li'):
        if team_type != e.parent.get('class')[-1]:
            order_count = 1
            team_type = e.parent.get('class')[-1]

        if e.get('class') and 'lineup__player-highlight' in e.get('class'):
            data_pitching.append({
                'date': e.find_previous('main').get('data-gamedate'),
                'game_time': e.find_previous('div', attrs={'class':'lineup__time'}).get_text(strip=True),
                'pitcher_name':e.a.get_text(strip=True),
                'team':e.find_previous('div', attrs={'class':team_type}).next.strip(),
                'lineup_throws':e.span.get_text(strip=True)
            })
        elif e.get('class') and 'lineup__player' in e.get('class'):
            data_batter.append({
                'date': e.find_previous('main').get('data-gamedate'),
                'game_time': e.find_previous('div', attrs={'class':'lineup__time'}).get_text(strip=True),
                'pitcher_name':e.a.get_text(strip=True),
                'team':e.find_previous('div', attrs={'class':team_type}).next.strip(),
                'pos': e.div.get_text(strip=True),
                'batting_order':order_count,
                'lineup_bats':e.span.get_text(strip=True)
            })
            order_count+=1

    df_pitching = pd.DataFrame(data_pitching)
    df_batter = pd.DataFrame(data_batter)

    return df_pitching, df_batter
    