def select_games():
    """We want games that:
    1. Only use the original map
    2. Only have 4 players
    3. Have original tiles selected ...?
    """
    import pandas as pd

    folderlocation = "C:/Users/User1/PycharmProjects/TerraBot/terra-mystica"

    stats = pd.read_csv(f'{folderlocation}/games.csv')

    """
    input: 
    >> stats.base_map.value_counts(ii)
    Output:
    126fe960806d587c78546b30f1a90853b1ada468    0.667387  << this guy's our guy
    95a66999127893f5925a5f591d54f8bcb9a670e6    0.182092
    be8f6ebf549404d015547152d5f2a1906ae8dd90    0.139985
    fdb13a13cd48b7a3c3525f27e4628ff6905aa5b1    0.010466
    224736500d20520f195970eb0fd4c41df040c08c    0.000026
    735b073fd7161268bb2796c1275abda92acd8b1a    0.000017
    b109f78907d2cbd5699ced16572be46043558e41    0.000009
    c07f36f9e050992d2daf6d44af2bc51dca719c46    0.000009
    30b6ded823e53670624981abdb2c5b8568a44091    0.000009
    Name: base_map, dtype: float64"""

    # only use games that have the original map
    processedstats = stats[stats['base_map'] == '126fe960806d587c78546b30f1a90853b1ada468']

    # only use games that have 4 players
    processedstats = processedstats[processedstats['player_count'] == 4]

    # remove nas
    selectedgames = processedstats[~processedstats.isnull().any(axis=1)]

    return selectedgames
