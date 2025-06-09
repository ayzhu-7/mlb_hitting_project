from this import d
import pandas as pd
import numpy as np
import os
from pybaseball import statcast, pitching_stats, batting_stats, playerid_lookup, statcast_batter, statcast_pitcher
import statsapi
from datetime import date
import requests
from bs4 import BeautifulSoup
import unicodedata

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

def get_starting_pitchers_date(date):
    today = date.strftime('%Y-%m-%d')

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


def get_starting_ids_date(date):
    todays_pitchers = get_starting_pitchers_date(date)
    pitcher_ids = []

    for p in todays_pitchers:
        if p[0] != '':
            first_name, last_name = p[0].split(' ')[0], ' '.join(p[0].split(' ')[1:])
            player_id = playerid_lookup(last_name, first_name, fuzzy=True)
            player_id = player_id['key_mlbam'].values[0]
            pitcher_ids.append([player_id, p[1], p[2]])  # Append player_id, team, and game_id
    return pitcher_ids

def get_barrel_percentage(player_id, start_date, end_date):
    player_barrels = statcast_pitcher(start_dt=start_date, end_dt=end_date,player_id=player_id)

    if player_barrels.empty:
        return {'left_pct': 0, 'right_pct': 0, 'total_pct': 0}
    # print(player_barrels['player_name'].iloc[0])
    player_bbe = player_barrels[player_barrels['events'].notna()].copy()
    player_bbe = player_bbe[~player_bbe['events'].isin(['walk','hit_by_pitch','sac_fly','sac_bunt', 'truncated_pa'])]
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

def spray_angle(hc_x, hc_y, stand):
    mult = 1
    if stand == 'L':
        mult = -1
    return mult * np.degrees(np.arctan2((hc_x - 125.42), (198.27 - hc_y)))

def hitter_barrel_percentage(player_id, start_date, end_date, pitch_mix, sides):
    mapping = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
    player_barrels = statcast_batter(start_dt=start_date, end_dt=end_date,player_id=player_id)
    barrel_dict = {}

    for side in sides:
        if player_barrels.empty:
            temp_barrel_dict = {'total_barrels': 0, 'total_bbes': 0, 'barrel_pct': 0}
        else:
            player_abs = player_barrels[(player_barrels['events'].notna()) & (player_barrels['stand'] == side)].copy()
            total_pas = player_abs.shape[0]
            player_abs = player_abs[~player_abs['events'].isin(['walk','hit_by_pitch','field_error','sac_fly','sac_bunt','truncated_pa'])]
            player_abs['bases'] = player_abs['events'].map(lambda x: mapping.get(x, 0))
            player_hits = len(player_abs[player_abs['bases'] != 0]  )
            player_bbe = player_barrels[(player_barrels['description'].isin(['hit_into_play'])) & (player_barrels['stand'] == side)].copy()
            player_bbe = code_barrel(player_bbe)
            player_bbe['is_blast'] = player_bbe.apply(lambda row: get_blast(row['launch_speed'], row['bat_speed'],row['release_speed']), axis=1)
            player_bbe['pulled_air'] = player_bbe.apply(lambda row: 1 if (spray_angle(row['hc_x'], row['hc_y'], row['stand']) < -15 and row['launch_angle'] >= 10) else 0, axis=1)
            # player_bbe = player_bbe[player_bbe['pitch_name'].isin(pitch_mix[side].keys())]
            total_barrels = player_bbe['is_barrel'].sum()
            total_bbes = player_bbe.shape[0]
            total_abs = player_abs.shape[0]
            temp_barrel_dict = {'total_barrels': total_barrels,
                        'total_bbes': total_bbes,
                        'total_abs': total_abs,
                        'total_pas': total_pas,
                        'barrel_pct (BBE)': total_barrels / total_bbes if total_bbes > 0 else 0,
                        'barrel_pct (PA)': total_barrels / total_pas if total_abs > 0 else 0,
                        'blast_pct': player_bbe['is_blast'].sum() / total_bbes if total_bbes > 0 else 0,
                        'pulled_pct': player_bbe['pulled_air'].sum() / total_bbes if total_bbes > 0 else 0,
                        'iso': (player_abs['bases'].sum() - player_hits)/ total_abs if total_abs > 0 else 0,
                        'stand': side}
        barrel_dict[side] = temp_barrel_dict
    return barrel_dict


def hitter_barrel_percentage_date(player_id, start_date, end_date, pitch_mix, sides, hitter_df):
    mapping = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
    player_barrels = hitter_df[player_id][(hitter_df[player_id]['game_date'] <= end_date) & (hitter_df[player_id]['game_date'] >= start_date)]
    barrel_dict = {}

    for side in sides:
        if player_barrels.empty:
            temp_barrel_dict = {'total_barrels': 0, 'total_bbes': 0, 'barrel_pct': 0}
        else:
            player_abs = player_barrels[(player_barrels['events'].notna()) & (player_barrels['stand'] == side)].copy()
            total_pas = player_abs.shape[0]
            player_abs = player_abs[~player_abs['events'].isin(['walk','hit_by_pitch','field_error','sac_fly','sac_bunt','truncated_pa'])]
            player_abs['bases'] = player_abs['events'].map(lambda x: mapping.get(x, 0))
            player_hits = len(player_abs[player_abs['bases'] != 0]  )
            player_bbe = player_barrels[(player_barrels['description'].isin(['hit_into_play'])) & (player_barrels['stand'] == side)].copy()
            player_bbe = code_barrel(player_bbe)
            player_bbe['is_blast'] = player_bbe.apply(lambda row: get_blast(row['launch_speed'], row['bat_speed'],row['release_speed']), axis=1)
            player_bbe['pulled_air'] = player_bbe.apply(lambda row: 1 if (spray_angle(row['hc_x'], row['hc_y'], row['stand']) < -15 and row['launch_angle'] >= 10) else 0, axis=1)
            # player_bbe = player_bbe[player_bbe['pitch_name'].isin(pitch_mix[side].keys())]
            total_barrels = player_bbe['is_barrel'].sum()
            total_bbes = player_bbe.shape[0]
            total_abs = player_abs.shape[0]
            temp_barrel_dict = {'total_barrels': total_barrels,
                        'total_bbes': total_bbes,
                        'total_abs': total_abs,
                        'total_pas': total_pas,
                        'barrel_pct (BBE)': total_barrels / total_bbes if total_bbes > 0 else 0,
                        'barrel_pct (PA)': total_barrels / total_pas if total_abs > 0 else 0,
                        'blast_pct': player_bbe['is_blast'].sum() / total_bbes if total_bbes > 0 else 0,
                        'pulled_pct': player_bbe['pulled_air'].sum() / total_bbes if total_bbes > 0 else 0,
                        'iso': (player_abs['bases'].sum() - player_hits)/ total_abs if total_abs > 0 else 0,
                        'stand': side}
        barrel_dict[side] = temp_barrel_dict
    return barrel_dict

def get_barrel_pcts(barrel_dict):
    barrel_pcts = []
    for side in barrel_dict:
        if barrel_dict[side]['total_bbes'] > 0:
            barrel_pcts.append([barrel_dict[side]['barrel_pct (BBE)'], barrel_dict[side]['barrel_pct (PA)'], barrel_dict[side]['blast_pct'], barrel_dict[side]['pulled_pct'], barrel_dict[side]['iso'], barrel_dict[side]['stand'], barrel_dict[side]['total_bbes']])
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
        return [0, 0, 0,0,0,0, 0]
    elif len(barrel_pcts) == 1:
        if barrel_pcts[0][0] > 0.1:
            return barrel_pcts[0]
        else:
            return [0, 0, 0,0,0,0, 0]
    if barrel_pcts[0][0] < 0.1 and barrel_pcts[1][0] < 0.1:
        return [0, 0, 0,0,0,0, 0]
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
    

def get_specific_lineup(team, player_lineup):
    specific_lineup = player_lineup[player_lineup['team'] == team].copy()
    players = specific_lineup['pitcher_name'].tolist()
    return players


def remove_accents(input_str):
    normalized = unicodedata.normalize('NFKD', input_str)
    return ''.join(c for c in normalized if not unicodedata.combining(c))


def map_player_ids(team_ids):
    """
    Maps player names to MLB player IDs.
    
    Parameters
    ----------
    team_ids : list
        List of team IDs to map.
    
    Returns
    -------
    player_id_mapping : pandas.DataFrame
        DataFrame containing player name, team name, player ID, abbreviated name, clean name, and clean abbreviated name.
    """
    player_names = []
    player_ids = []
    team_names = []
    for team_id in team_ids:
        roster = statsapi.roster(team_id)
        team = statsapi.lookup_team(team_id)
        roster_player_names = [player.split()[2:] for player in roster.split('\n')]
        for name in roster_player_names:
            if len(name) >= 1:
                if len(name) > 2:
                    if name[2] == 'Jr.' or name[2] == 'Sr.' or name[2] == 'II' or name[2] == 'III' or name[2] == 'IV':
                        name[1] = ' '.join(name[1:-1])
                    elif name[1] == 'A.':
                        name[1] = ' '.join(name[2:])
                    else:
                        name[1] = ' '.join(name[1:])
                    print(name[0] + ' ' + name[1])
                player_id = playerid_lookup(name[1], name[0], fuzzy=True)['key_mlbam'].iloc[0]
                player_names.append(name[0] + ' ' + name[1])
                player_ids.append(player_id)
                team_names.append(team[0]['teamName'])

    player_id_mapping = pd.DataFrame({'name': player_names, 'team': team_names, 'player_id': player_ids})
    player_id_mapping['abbrev'] = player_id_mapping['name'].apply(lambda x: x.split(' ')[0][0] + '. ' + ' '.join(x.split(' ')[1:]))
    player_id_mapping['clean_name'] = player_id_mapping['name'].apply(remove_accents)
    player_id_mapping['clean_abbrev'] = player_id_mapping['clean_name'].apply(lambda x: x.split(' ')[0][0] + '. ' + ' '.join(x.split(' ')[1:]))
    player_id_mapping = player_id_mapping[['name','abbrev','clean_name','clean_abbrev','team','player_id']]
    player_id_mapping.to_csv('player_id_mapping.csv', index=False)
    return player_id_mapping
    


def get_player_ids(player_names, player_map_df, team_name):
    filtered_df = player_map_df[player_map_df['team'] == team_name]
    player_ids = []
    for name in player_names:
        new_df = filtered_df[filtered_df['name'] == name]['player_id'].values
        if len(new_df) == 0:
            new_df = filtered_df[filtered_df['abbrev'] == name]['player_id'].values
        if len(new_df) == 0:
            new_df = filtered_df[filtered_df['clean_name'] == name]['player_id'].values
        if len(new_df) == 0:
            new_df = filtered_df[filtered_df['clean_abbrev'] == name]['player_id'].values
        if len(new_df) == 0:
            raise ValueError(f"Player {name} not found in player_id_mapping.csv")
        player_ids.append(new_df[0])
    return player_ids


def get_hitter_outcomes(player_id, start_date, end_date, side):
    mapping = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
    player_outcomes = statcast_batter(start_dt=start_date, end_dt=end_date,player_id=player_id)
    player_outcomes = player_outcomes[(player_outcomes['events'].notna()) & (player_outcomes['stand'] == side)]
    player_outcomes['bases'] = player_outcomes['events'].map(lambda x: mapping.get(x, 0))
    if len(player_outcomes):
        print(player_outcomes['player_name'].iloc[0])
    return player_outcomes['bases'].sum()

def get_single_hitter_history(player_id, start_date, end_date):
    mapping = {'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}
    hitting_history = statcast_batter(start_dt=start_date, end_dt=end_date,player_id=player_id)
    hitting_history = hitting_history[hitting_history['events'].notna()]
    hitting_history['bases'] = hitting_history['events'].map(lambda x: mapping.get(x, 0))
    game_history = hitting_history.groupby('game_date')['bases'].sum().reset_index()
    game_history = game_history.rename(columns={'game_date': 'date', 'bases': player_id})
    game_history['date'] = pd.to_datetime(game_history['date'])
    return game_history

def get_all_hitter_history(player_ids, start_date, end_date):
    all_outcome = pd.DataFrame()
    dates = pd.date_range(start_date, end_date)
    all_outcome['date'] = dates
    for player_id in player_ids:
        game_history = get_single_hitter_history(player_id, start_date, end_date)
        all_outcome = pd.merge(all_outcome, game_history, on='date', how='left')
    return all_outcome

def update_hitter_history(player_hitting_history, start_date, end_date):
    player_ids = player_hitting_history.columns.tolist()[1:]
    for player_id in player_ids:
        new_data = get_single_hitter_history(player_id, start_date, end_date)
        player_hitting_history = pd.merge(player_hitting_history, new_data, on='date', how='left')
        player_hitting_history[player_id] = player_hitting_history[player_id].fillna(0)
    return player_hitting_history

def get_hitter_barrels(hitter_barrels, hitter_ids, start_date, end_date, pitch_mix, sides):
    if hitter_ids is None or len(hitter_ids) == 0:
        return {}
    hitter_barrels_copy = hitter_barrels[hitter_barrels['player_id'].isin(hitter_ids)].copy()
    hitter_barrels_copy['barrel_data'] = hitter_barrels_copy['player_id'].apply(hitter_barrel_percentage, start_date=start_date, end_date=end_date, pitch_mix=pitch_mix, sides=sides)
    hitter_barrels_copy['barrel_percentage'] = hitter_barrels_copy.apply(lambda row: get_barrel_pcts(row['barrel_data']), axis=1)
    hitter_barrels_copy['barrel_qual'] = hitter_barrels_copy['barrel_percentage'].apply(lambda x: check_barrel_pcts(x))
    # print(hitter_barrels_copy['barrel_qual'])
    filtered = hitter_barrels_copy[hitter_barrels_copy['barrel_qual'].apply(lambda x: x != [0,0,0,0,0,0,0])].copy()
    # print(filtered)
    if len(filtered) > 3:
        # Create a new column with just the first element
        filtered['barrel_qual_first'] = filtered['barrel_qual'].apply(lambda x: x[0])
        # Sort by that column and get top 3
        top_3 = filtered.nlargest(3, 'barrel_qual_first')
        # Drop the temporary column if you don't need it
        top_3 = top_3.drop('barrel_qual_first', axis=1)
        return dict(zip(zip(top_3['last_name, first_name'], top_3['player_id']), top_3['barrel_qual']))
    else:
        return dict(zip(zip(filtered['last_name, first_name'], filtered['player_id']), filtered['barrel_qual']))

def get_hitter_barrels_date(hitter_barrels, hitter_ids, start_date, end_date, pitch_mix, sides, hitter_df):
    if hitter_ids is None or len(hitter_ids) == 0:
        return {}
    hitter_barrels_copy = hitter_barrels[hitter_barrels['player_id'].isin(hitter_ids)].copy()
    hitter_barrels_copy['barrel_data'] = hitter_barrels_copy['player_id'].apply(hitter_barrel_percentage_date, start_date=start_date, end_date=end_date, pitch_mix=pitch_mix, sides=sides, hitter_df=hitter_df)
    hitter_barrels_copy['barrel_percentage'] = hitter_barrels_copy.apply(lambda row: get_barrel_pcts(row['barrel_data']), axis=1)
    hitter_barrels_copy['barrel_qual'] = hitter_barrels_copy['barrel_percentage'].apply(lambda x: check_barrel_pcts(x))
    # print(hitter_barrels_copy['barrel_qual'])
    filtered = hitter_barrels_copy[hitter_barrels_copy['barrel_qual'].apply(lambda x: x != [0,0,0,0,0,0,0])].copy()
    # print(filtered)
    if len(filtered) > 3:
        # Create a new column with just the first element
        filtered['barrel_qual_first'] = filtered['barrel_qual'].apply(lambda x: x[0])
        # Sort by that column and get top 3
        top_3 = filtered.nlargest(3, 'barrel_qual_first')
        # Drop the temporary column if you don't need it
        top_3 = top_3.drop('barrel_qual_first', axis=1)
        return dict(zip(zip(top_3['last_name, first_name'], top_3['player_id']), top_3['barrel_qual']))
    else:
        return dict(zip(zip(filtered['last_name, first_name'], filtered['player_id']), filtered['barrel_qual']))


def get_specific_day_hitters(date, pitcher_df, hitter_df):
    start_date = date.strftime('%Y-%m-%d')
    end_date = (date + pd.DateOffset(days=-30)).strftime('%Y-%m-%d')    
    pitcher_barrels = statcast_pitcher_exitvelo_barrels(2025)
    
    pitcher_ids = get_starting_ids_date(date)
    pitcher_id_nos = [p[0] for p in pitcher_ids]
    pitchers_arsenals = {}
    pitcher_opponents = {}
    pitcher_games = {}
    for id, opposing_team, game_id in pitcher_ids:
        pitcher_data = pitcher_df[id][(pitcher_df[id]['game_date'] <= end_date) & (pitcher_df[id]['game_date'] >= start_date)]
        pitcher_opponents[id] = opposing_team
        pitcher_games[id] = game_id
        pitchers_arsenals[id] = {'R': get_pitch_arsenal(pitcher_data, 'R'), 'L': get_pitch_arsenal(pitcher_data, 'L')}

    pitcher_barrels_copy = pitcher_barrels[pitcher_barrels['player_id'].isin(pitcher_id_nos)].copy()
    pitcher_barrels_copy['barrel_percentage'] = pitcher_barrels_copy['player_id'].apply(get_barrel_percentage, start_date=start_date, end_date=end_date)
    pitcher_barrels_copy['left_barrel_pct'] = pitcher_barrels_copy['barrel_percentage'].apply(lambda x: x['left_pct'])
    pitcher_barrels_copy['right_barrel_pct'] = pitcher_barrels_copy['barrel_percentage'].apply(lambda x: x['right_pct'])
    pitcher_barrels_copy['opponent'] = pitcher_barrels_copy['player_id'].apply(lambda x: pitcher_opponents[x])
    pitcher_barrels_copy['game_id'] = pitcher_barrels_copy['player_id'].apply(lambda x: pitcher_games.get(x))
    for index, row in pitcher_barrels_copy.iterrows():
        if row['barrel_percentage']['total_pct'] == 0:
            pitcher_barrels_copy.at[index, 'left_barrel_pct'] = row['brl_percent'] / 100
            pitcher_barrels_copy.at[index, 'right_barrel_pct'] = row['brl_percent'] / 100
        
    filtered_pitcher_barrels_copy = pitcher_barrels_copy[(pitcher_barrels_copy['left_barrel_pct'] > 0.07) | (pitcher_barrels_copy['right_barrel_pct'] > 0.07)].copy()
    filtered_pitcher_barrels_copy['sides'] = filtered_pitcher_barrels_copy.apply(lambda row: get_sides(row['left_barrel_pct'], row['right_barrel_pct']),axis=1)
    filtered_pitcher_barrels_copy['arsenal'] = filtered_pitcher_barrels_copy.apply(lambda row: pitchers_arsenals[row['player_id']], axis=1)


    pitcher_df = filtered_pitcher_barrels_copy.copy()
    pitcher_df['hitter_team'] = pitcher_df.apply(lambda row: get_team(row['game_id'], row['opponent']), axis=1)
    pitcher_df['opposing_lineup'] = pitcher_df.apply(lambda row: get_opposing_lineup(row['game_id'], row['opponent']), axis=1)

    hitter_barrels = statcast_batter_exitvelo_barrels(2025)

    hitter_pitcher_df = pitcher_df.copy()
    hitter_pitcher_df['hitter_barrels'] = hitter_pitcher_df.apply(lambda row: get_hitter_barrels_date(hitter_barrels, row['opposing_lineup'], start_date, end_date, row['arsenal'], row['sides'], hitter_df), axis=1)

    new_batter_ids = []
    for index, row in hitter_pitcher_df.iterrows():
        for key in row['hitter_barrels']:
            new_batter_ids.append(row['hitter_barrels'][key][1])
    
    return new_batter_ids