import pandas as pd
import argparse
import pickle
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np


def featurise_features(featdf, params):
    # adjust features dataset for chosen encoding
    X = featdf.to_numpy()
    game = X[:, :1]
    rounddata = X[:, 1:7]
    bontiles = X[:, 7:16]
    playerdata = X[:, 17:18]
    colours = X[:, 18:25]
    mapdata = X[:, -1:]

    onehot_encoder = OneHotEncoder(sparse=False)
    ordinal_encoder = OrdinalEncoder()

    if params['prepare-step2']['round-features'] is 'ordinal':
        rounddata = ordinal_encoder.fit_transform(rounddata)
    else:  # one-hot
        rounddata = onehot_encoder.fit_transform(rounddata)
    if params['prepare-step2']['playercount-features'] is 'ordinal':
        playerdata = ordinal_encoder.fit_transform(playerdata)
    else:  # one-hot
        playerdata = onehot_encoder.fit_transform(playerdata)
    if params['prepare-step2']['map-features'] is 'ordinal':
        mapdata = ordinal_encoder.fit_transform(mapdata)
    else:  # one-hot
        mapdata = onehot_encoder.fit_transform(mapdata)

    featdata = np.hstack((game, rounddata, bontiles, playerdata, colours, mapdata))
    return featdata


def main(params):
    vpdfdir = params['prepare']['vp-data-dir']
    featdfdir = params['prepare']['feature-data-dir']
    pickledir = params['prepare-step2']['pickle-dir']

    vpdf = pd.read_csv(vpdfdir)
    featdf = pd.read_csv(featdfdir)

    vpdf = vpdf.sort_values('game')
    featdf = featdf.sort_values('game')

    featdf = featdf.drop(columns=['Unnamed: 0'])
    featdatanp = featurise_features(featdf, params)

    each_faction_dataset = dict()

    colnames = list(vpdf.columns)
    factions = [x for x in colnames if x != 'game' and x != 'Unnamed: 0']

    for faction in factions:
        faction_dataset = {}
        indexes = pd.isnull(vpdf[faction])
        vpdata = pd.Series(index=vpdf['game'][~indexes], data=vpdf[faction][~indexes].values)
        vpdata.sort_index()

        featdata = featdf[~indexes]
        featdata.index = featdata['game']
        featdata = featdata.drop(columns=['game'])
        featdata.sort_index()
        featdata2 = featdatanp[~indexes, :]
        featdata2 = featdata2[featdata2[:, 0].argsort()]  # sort by games (first col)

        faction_dataset['vp'] = vpdata
        faction_dataset['features'] = featdata
        faction_dataset['featuresnp'] = featdata2
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

