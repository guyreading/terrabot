import pandas as pd
import numpy as np
import math
import time

def main():
    """This is the script version of creatingVPdata.ipynb for the dvc pipeline."""
    # load data in
    folderlocation = "D:/PycharmProjects/TerraBot/terra-mystica"
    gameevents = pd.read_csv(f'{folderlocation}/game_events.csv')
    games = pd.read_csv(f'{folderlocation}/games.csv')
    gameslist = list(pd.unique(gameevents['game']))
    allfactions = pd.unique(gameevents['faction'])
    gamescoringtiles = pd.read_csv(f'{folderlocation}/game_scoring_tiles.csv')
    gameoptions = pd.read_csv(f'{folderlocation}/game_options.csv')
    stats = pd.read_csv(f'{folderlocation}/stats.csv')

    # two vp dataset functions
    def makenewdf():
        """make an empty dataframe, organised in the way we want the target data, ready to be populated"""
        validfactions = ['witches', 'auren', 'swarmlings', 'mermaids', 'cultists', 'halflings', 'dwarves', 'engineers', 'chaosmagicians', 'giants', 'fakirs', 'nomads', 'darklings', 'alchemists']
        dfcols = ['game'] + validfactions
        vpdf = pd.DataFrame(columns=dfcols)

        return vpdf, dfcols, validfactions

    vpdf, dfcols, validfactions = makenewdf()

    def get_vp_from_game(singleGameEvents):
        """Input game events for a single game. This is a pd.DataFrame.
        Output a row where each faction in the game has its vp populated (the rest are nans)
        """
        newdf = pd.DataFrame([[np.nan] * 15], columns=dfcols)

        # assign the game number
        gameno = list(pd.unique(singleGameEvents['game']))

        # assert len(gameno) == 1, 'More than 1 unique game was found'
        try:
            newdf['game'].replace({np.nan: gameno[0]}, inplace=True)
        except:
            print(f'DEBUGGING: len of table is {len(singleGameEvents)}')
            print(f'DEBUGGING: gamnos list: {gameno}')
            print(singleGameEvents)
            raise

        # find factions - there are some artifacts in the data. E.g. the "faction", "all". We need to filter them out.
        rawfactions = list(pd.unique(singleGameEvents['faction']))
        verifiedfactions = [rawfaction for rawfaction in rawfactions if rawfaction in validfactions]

        for faction in verifiedfactions:
            vpfaction = sum(singleGameEvents[(singleGameEvents['event'] == 'vp') & (singleGameEvents['faction'] == faction)]['num'])
            newdf[faction].replace({np.nan: vpfaction}, inplace=True)

        return newdf

    # two features dataset functions
    def emptyfeaturesdf():
        """make an empty dataframe, organised in the way we want the feature data, ready to be populated"""
        colnames = ['game']
        uniqueScoreTiles = np.sort(pd.unique(gamescoringtiles['tile']))

        # One-hot of round tiles, for each round
        for gameround in range(1, 7):
            roundstr = f'r{gameround}'
            for tile in uniqueScoreTiles:
                colnames.append(roundstr + '_' + tile)

        # Boolean of bonus tiles
        for bon in range(1, 11):
            colnames.append(f'BON{bon}')

        # One-hot player count (from 2, 3, 4 or 5 players)
        for player in range(2, 6):
            colnames.append(f'{player}players')

        # one hot of the map used
        """126fe960806d587c78546b30f1a90853b1ada468 - map1
           95a66999127893f5925a5f591d54f8bcb9a670e6 - map2
           be8f6ebf549404d015547152d5f2a1906ae8dd90 - map3
        """
        colnames = colnames + ['map1', 'map2', 'map3']

        featuresdf = pd.DataFrame(columns=colnames)

        return featuresdf, colnames

    featuresdf, featcolnames = emptyfeaturesdf()

    def get_features_from_game(singlegameevents, singlegamemeta, singlegameST, singleendplayers=None):
        """
        Inputs:
            singlegameevents <pd.DataFrame>   - is game events for a single game
            singlegamemeta   <pd.DataFrame>   - is a single row from `games` that gives map & player count
            singlegameST     <pd.DataFrame>   - is a single row from `gamescoringtiles` that gives... score tile (suprisingly)
            singleendplayers <pd.DataFrame>   - is a single row from `end players` that gives the amount of players at end of game, after dropouts
        Return:          <pd.DataFrame>   - a row where features have been found (will be sparse)
        """
        newdf = pd.DataFrame([[0] * len(featcolnames)], columns=featcolnames)

        # assign game string
        singlegamemeta.iloc[0]['game']
        newdf['game'].replace({0: singlegamemeta.iloc[0]['game']}, inplace=True)

        # find the round tiles for each round
        for gameround in range(1, 7):
            roundstr = f'r{gameround}'
            scoretile = roundstr + '_' + singlegameST[singlegameST['round'] == gameround]['tile'].values[0]
            newdf[scoretile].replace({0: 1}, inplace=True)

        # Boolean of bonus tiles
        uniqueevents = list(pd.unique(singlegameevents['event']))
        bonustiles = [event[5:] for event in uniqueevents if event.startswith('pass:BON')]
        for bontile in bonustiles:
            newdf[bontile].replace({0: 1}, inplace=True)

        # One-hot player count (from 2, 3, 4 or 5 players)
        if singleendplayers is None:
            noplayers = singlegamemeta.iloc[0]['player_count']
            print('gamemeta used for player count')
        else:
            noplayers = singleendplayers.iloc[0]['endplayers']

        players = f'{noplayers}players'
        newdf[players].replace({0: 1}, inplace=True)

        # one hot of the map used
        mapdict = {'126fe960806d587c78546b30f1a90853b1ada468': 'map1',
                   '95a66999127893f5925a5f591d54f8bcb9a670e6': 'map2',
                   'be8f6ebf549404d015547152d5f2a1906ae8dd90': 'map3'
                   }
        basemap = singlegamemeta.iloc[0]['base_map']
        gamemap = mapdict[basemap]
        newdf[gamemap].replace({0: 1}, inplace=True)

        return newdf

    # filtering
    # making a dataset for ease
    data = dict()
    data['gameevents'] = gameevents
    data['games'] = games
    data['gamescoringtiles'] = gamescoringtiles

    def filteringByBadgames(data, badgames):
        """ Data is a dict containing gameevents, games, gamescoringtiles
            badgames is a pd.dataframe that contains ['game'] to filter by
        """
        gameeventsfil = data['gameevents']
        gamesfil = data['games']
        gamescoringtilesfil = data['gamescoringtiles']

        badgameslist = badgames['game']
        gameeventsfilbefore = len(gameeventsfil)
        gamesbefore = len(gamesfil)
        gameSTbefore = len(gamescoringtilesfil)

        gameeventsfil = gameeventsfil[~gameeventsfil['game'].isin(badgameslist)]
        gamesfil = gamesfil[~gamesfil['game'].isin(badgameslist)]
        gamescoringtilesfil = gamescoringtilesfil[~gamescoringtilesfil['game'].isin(badgameslist)]

        print(f'game events before: {gameeventsfilbefore}, game events after: {len(gameeventsfil)}, game events removed: {gameeventsfilbefore-len(gameeventsfil)}')
        print(f'games before: {gamesbefore}, games after: {len(gamesfil)}, games removed: {gamesbefore-len(gamesfil)}')
        print(f'gameST before: {gameSTbefore}, gameST after: {len(gamescoringtilesfil)}, games removed: {gameSTbefore-len(gamescoringtilesfil)}')

        data['gameevents'] = gameeventsfil
        data['games'] = gamesfil
        data['gamescoringtiles'] = gamescoringtilesfil

        return data

    # player count
    badgames = games[games["player_count"].isin([1, 6, 7])]
    data = filteringByBadgames(data, badgames)


if __name__ == "__main__":
    main()