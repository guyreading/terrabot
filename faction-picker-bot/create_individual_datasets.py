import pandas as pd
import argparse
import pickle
import yaml


def main(vpdfdir, featdfdir, pickledir):

    vpdf = pd.read_csv(vpdfdir)
    featdf = pd.read_csv(featdfdir)

    vpdf = vpdf.sort_values('game')
    featdf = featdf.sort_values('game')

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
        featdata = featdata.drop(columns=['game', 'Unnamed: 0'])
        featdata.sort_index()

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

    vpdfdir = params['prepare']['vp-data-dir']
    featdfdir = params['prepare']['feature-data-dir']
    pickledir = params['prepare-step2']['pickle-dir']

    main(vpdfdir, featdfdir, pickledir)

