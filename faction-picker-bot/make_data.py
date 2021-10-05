import pandas as pd
import numpy as np
import argparse
import math
import time
import yaml


def main(folderlocation, vpdfdir, featdfdir, playerdropdir):
    """This is the script version of creatingVPdata.ipynb for the dvc pipeline."""
    # load data in
    gameevents = pd.read_csv(f'{folderlocation}game_events.csv')
    games = pd.read_csv(f'{folderlocation}games.csv')
    gameslist = list(pd.unique(gameevents['game']))
    allfactions = pd.unique(gameevents['faction'])
    gamescoringtiles = pd.read_csv(f'{folderlocation}game_scoring_tiles.csv')
    gameoptions = pd.read_csv(f'{folderlocation}game_options.csv')
    stats = pd.read_csv(f'{folderlocation}stats.csv')

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
            colnames.append(f'round{gameround}tile')


        # Boolean of bonus tiles
        for bon in range(1, 11):
            colnames.append(f'BON{bon}')

        # One-hot player count
        colnames.append('no_players')

        # Boolean of other colours present on the board
        colnames = colnames + ['red', 'blue', 'green', 'black', 'grey', 'yellow', 'brown']


        # acceptable maps
        """126fe960806d587c78546b30f1a90853b1ada468 - map1
           95a66999127893f5925a5f591d54f8bcb9a670e6 - map2
           be8f6ebf549404d015547152d5f2a1906ae8dd90 - map3
        """
        colnames.append('map')

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
        newdf['game'].replace({0: singlegamemeta.iloc[0]['game']}, inplace=True)

        # find the round tiles for each round
        for gameround in range(1, 7):
            roundstr = f'round{gameround}tile'
            scoretile = singlegameST[singlegameST['round'] == gameround]['tile'].values[0]
            newdf[roundstr].replace({0: scoretile}, inplace=True)

        # Boolean of bonus tiles
        uniqueevents = list(pd.unique(singlegameevents['event']))
        bonustiles = [event[5:] for event in uniqueevents if event.startswith('pass:BON')]
        for bontile in bonustiles:
            newdf[bontile].replace({0: 1}, inplace=True)

        # get faction colour
        factions = pd.unique(singlegameevents['faction'])
        if 'auren' in factions or 'witches' in factions:
            newdf['green'].replace({0: 1}, inplace=True)
        if 'swarmlings' in factions or 'mermaids' in factions:
            newdf['blue'].replace({0: 1}, inplace=True)
        if 'cultists' in factions or 'halflings' in factions:
            newdf['brown'].replace({0: 1}, inplace=True)
        if 'darklings' in factions or 'alchemists' in factions:
            newdf['black'].replace({0: 1}, inplace=True)
        if 'nomads' in factions or 'fakirs' in factions:
            newdf['yellow'].replace({0: 1}, inplace=True)
        if 'giants' in factions or 'chaos magicians' in factions:
            newdf['red'].replace({0: 1}, inplace=True)
        if 'engineers' in factions or 'dwarves' in factions:
            newdf['grey'].replace({0: 1}, inplace=True)

        # Number player count (from 2, 3, 4 or 5 players)
        if singleendplayers is None:
            noplayers = singlegamemeta.iloc[0]['player_count']
            print('gamemeta used for player count')
        else:
            noplayers = singleendplayers.iloc[0]['endplayers']

        newdf['no_players'].replace({0: noplayers}, inplace=True)

        # one hot of the map used
        mapdict = {'126fe960806d587c78546b30f1a90853b1ada468': 'map1',
                   '95a66999127893f5925a5f591d54f8bcb9a670e6': 'map2',
                   'be8f6ebf549404d015547152d5f2a1906ae8dd90': 'map3'
                   }
        basemap = singlegamemeta.iloc[0]['base_map']
        gamemap = mapdict[basemap]
        newdf['map'].replace({0: gamemap}, inplace=True)

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

    extendedfactions = validfactions + ['dragonlords', 'riverwalkers', 'yetis', 'icemaidens', 'shapeshifters', 'acolytes']

    def check_game_ended(singlegame, verbose=False):
        r5 = singlegame[singlegame['round'] == 5]
        rawfactions = pd.unique(singlegame['faction'])
        verifiedfactions = [rawfaction for rawfaction in rawfactions if rawfaction in extendedfactions]
        allbool = []

        for faction in verifiedfactions:
            factionbool = len(r5[(r5['faction'] == faction)  & (r5['event'].str.startswith('pass'))]) == 1
            allbool.append(factionbool)

            if verbose:
                print(f'faction: {faction} ended their turn?: {factionbool}')

        isgood = all(allbool)
        startplayers = len(verifiedfactions)
        boolsum = sum(allbool)
        playersdropped = startplayers - boolsum

        return isgood, startplayers, playersdropped

    gameevents = data['gameevents']
    gameslist = list(pd.unique(gameevents['game']))

    gamelengthlen = len(gameslist)
    gamesroundup = math.ceil(gamelengthlen / 100.0) * 100
    jj = 0
    playerdropdf = pd.DataFrame(columns=['game', 'nodrops', 'startplayers', 'playersdropped'])

    for ii in range(100, gamesroundup+1, 100):
        ii = min(ii, gamelengthlen)
        if (ii % 10000) == 0:  # update every 10000 games
            print(f'Progressed to {ii}th game')

        next100games = gameslist[jj:ii]
        jj = ii  # so that we don't get any repetitions at the very end, where our set will be smaller

        gameevents100 = gameevents[gameevents['game'].isin(next100games)]

        for game in next100games:
            singlegame = gameevents100[gameevents100['game'] == game]

            if not len(singlegame) == 0:
                isgood, startplayers, playersdropped = check_game_ended(singlegame)
                newdf = pd.DataFrame([[game, isgood, startplayers, playersdropped]], columns=['game', 'nodrops', 'startplayers', 'playersdropped'])
                playerdropdf = playerdropdf.append(newdf, ignore_index=True)

    playerdropdf['endplayers'] = playerdropdf['startplayers'] - playerdropdf['playersdropped']
    playerdropdf.to_csv(playerdropdir)

    badgames = playerdropdf[playerdropdf['endplayers'].isin([0, 1])]
    data = filteringByBadgames(data, badgames)

    # removing unwanted maps
    acceptablemaps = ['126fe960806d587c78546b30f1a90853b1ada468',
                      '95a66999127893f5925a5f591d54f8bcb9a670e6',
                      'be8f6ebf549404d015547152d5f2a1906ae8dd90']

    badgames = games[~games["base_map"].isin(acceptablemaps)]
    data = filteringByBadgames(data, badgames)

    # creating full dataset
    vpdf, _, _ = makenewdf()
    featdf, _ = emptyfeaturesdf()

    gameevents = data['gameevents']
    games = data['games']
    gamescoringtiles = data['gamescoringtiles']

    gameslist = list(pd.unique(gameevents['game']))
    gamelengthlen = len(gameslist)
    gamesroundup = math.ceil(gamelengthlen / 100.0) * 100
    jj = 0

    for ii in range(100, gamesroundup+1, 100):
        ii = min(ii, gamelengthlen)
        if (ii % 10000) == 0:  # update every 10000 games
            print(f'Progressed to {ii}th game')

        next100games = gameslist[jj:ii]
        jj = ii  # so that we don't get any repetitions at the very end, where our set will be smaller

        gameevents100 = gameevents[gameevents['game'].isin(next100games)]
        gamemeta100 = games[games['game'].isin(next100games)]
        gameST100 = gamescoringtiles[gamescoringtiles['game'].isin(next100games)]
        endplayers100 = playerdropdf[playerdropdf['game'].isin(next100games)]  # use this for player count

        for game in next100games:
            singlegame = gameevents100[gameevents100['game'] == game]
            singlegamemeta = gamemeta100[gamemeta100['game'] == game]
            singlegameST = gameST100[gameST100['game'] == game]
            singleendplayers = endplayers100[endplayers100['game'] == game]

            if not len(singlegame) == 0:
                vpforgame = get_vp_from_game(singlegame)
                featsforgame = get_features_from_game(singlegame, singlegamemeta, singlegameST, singleendplayers=singleendplayers)

                vpdf = vpdf.append(vpforgame, ignore_index=True)
                featdf = featdf.append(featsforgame, ignore_index=True)

    print(f"no of unique games in table is: {len(list(pd.unique(vpdf['game'])))}")

    vpdf.to_csv(vpdfdir)
    featdf.to_csv(featdfdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input DVC params.')
    parser.add_argument('--params', type=str)
    args = parser.parse_args()
    paramsdir = args.params

    with open(paramsdir, 'r') as fd:
        params = yaml.safe_load(fd)

    vpdfdir = params['prepare']['vp-data-dir']
    featdfdir = params['prepare']['feature-data-dir']
    playerdropdir = params['prepare']['player-drop-dir']
    folderlocation = params['prepare']['folderlocation']

    main(folderlocation, vpdfdir, featdfdir, playerdropdir)
