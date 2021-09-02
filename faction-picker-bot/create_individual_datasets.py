import pandas as pd
import argparse
import pickle
import yaml


def main(vpdfdir, featdfdir, pickledir):

    vpdf = pd.read_csv(vpdfdir)
    featdf = pd.read_csv(featdfdir)

    each_faction_dataset = dict()

    colnames = list(vpdf.columns)
    factions = [x for x in colnames if x != 'game' and x != 'Unnamed: 0']

    each_faction_dataset = {}

    for faction in factions:
        faction_dataset = {}
        indexes = pd.isnull(vpdf[faction])
        faction_dataset['vp'] = vpdf[faction][~indexes]
        faction_dataset['features'] = featdf[~indexes]
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

