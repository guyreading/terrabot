import pandas as pd
import argparse
import pickle
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np


def featurise_features(featdf, params):
    # adjust features dataset for chosen encoding
    game = featdf.iloc[:, :1]
    rounddata = featdf.iloc[:, 1:7]
    bontiles = featdf.iloc[:, 7:16]
    playerdata = featdf.iloc[:, 17:18]
    colours = featdf.iloc[:, 18:25]
    mapdata = featdf.iloc[:, -1:]

    onehot_encoder = OneHotEncoder(sparse=False)
    ordinal_encoder = OrdinalEncoder()

    if params['prepare-step2']['round-features'] == 'ordinal':
        rounddatanp = ordinal_encoder.fit_transform(rounddata)
        rounddata = pd.DataFrame(data=rounddatanp, columns=rounddata.columns)
    else:  # one-hot
        rounddatanp = onehot_encoder.fit_transform(rounddata)
        rounddata = pd.DataFrame(data=rounddatanp, columns=onehot_encoder.get_feature_names())
    if params['prepare-step2']['playercount-features'] == 'ordinal':
        playerdatanp = ordinal_encoder.fit_transform(playerdata)
        playerdata = pd.DataFrame(data=playerdatanp, columns=playerdata.columns)
    else:  # one-hot
        playerdatanp = onehot_encoder.fit_transform(playerdata)
        playerdata = pd.DataFrame(data=playerdatanp, columns=onehot_encoder.get_feature_names())
    if params['prepare-step2']['map-features'] == 'ordinal':
        mapdatanp = ordinal_encoder.fit_transform(mapdata)
        mapdata = pd.DataFrame(data=mapdatanp, columns=mapdata.columns)
    else:  # one-hot
        mapdatanp = onehot_encoder.fit_transform(mapdata)
        mapdata = pd.DataFrame(data=mapdatanp, columns=onehot_encoder.get_feature_names())

    featdf = pd.concat([game, rounddata, bontiles, playerdata, colours, mapdata], axis=1)
    return featdf


def main(params):
    vpdfdir = params['prepare']['vp-data-dir']
    featdfdir = params['prepare']['feature-data-dir']
    pickledir = params['prepare-step2']['pickle-dir']

    vpdf = pd.read_csv(vpdfdir)
    featdf = pd.read_csv(featdfdir)

    vpdf = vpdf.sort_values('game')
    featdf = featdf.sort_values('game')

    featdf = featdf.drop(columns=['Unnamed: 0'])
    featdf = featurise_features(featdf, params)

    each_faction_dataset = dict()

    colnames = list(vpdf.columns)
    factions = [x for x in colnames if x != 'game' and x != 'Unnamed: 0']

    for faction in factions:
        faction_dataset = {}
        vpdf = vpdf.sort_index()
        indexes = pd.isnull(vpdf[faction])
        vpdata = pd.Series(index=vpdf['game'][~indexes], data=vpdf[faction][~indexes].values)

        featdf = featdf.sort_index()
        featdata = featdf[~indexes]
        featdata.index = featdata['game']
        featdata = featdata.drop(columns=['game'])

        faction_dataset['vp'] = vpdata
        faction_dataset['features'] = featdata
        each_faction_dataset[faction] = faction_dataset

    with open(pickledir, 'wb') as pklfile:
        pickle.dump(each_faction_dataset, pklfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input DVC params.')
    parser.add_argument('--params', type=str)
    args = parser.parse_args()
    paramsdir = args.params

    with open(paramsdir, 'r') as fd:
        params = yaml.safe_load(fd)

    main(params)

