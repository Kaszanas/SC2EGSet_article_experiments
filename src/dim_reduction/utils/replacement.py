
class Replacement:
    def race_name_into_number_value(df):
        races = {'Terr': 1, 'Zerg': 2, 'Prot': 3}
        df.race = [races[item] for item in df.race]


    def outcome_into_number_value(df):
        outcome = {'Undecided': 0, 'Loss': 1, 'Win': 2}
        df.outcome = [outcome[item] for item in df.outcome]



    def map_into_number_value(df):
        csv_maps = df['map_name'].unique()
        unique_maps = {}
        number = 0
        for i in csv_maps:
            unique_maps[i] = number
            number += 1
        df.map_name = [unique_maps[item] for item in df.map_name]

    def player_name_into_number_value(df):
        csv_player_name = df['player_name'].unique()
        unique_player_name = {}
        number = 0
        for i in csv_player_name:
            unique_player_name[i] = number
            number += 1
        df.player_name = [unique_player_name[item] for item in df.player_name]

    def player_toon_into_number_value(df):
        csv_player_toon = df['player_toon'].unique()
        unique_player_toon = {}
        number = 0
        for i in csv_player_toon:
            unique_player_toon[i] = number
            number += 1
        df.player_toon = [unique_player_toon[item] for item in df.player_toon]