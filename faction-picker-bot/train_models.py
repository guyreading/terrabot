from pathlib import Path
import numpy as np
import lightgbm as lgb
import argparse
import pickle
import yaml
import math


def main(params):
    pickledir = params['prepare-step2']['pickle-dir']
    modeldir = params['training']['model-dir']

    Path(modeldir).mkdir(parents=True, exist_ok=True)
    with open(pickledir, 'rb') as fd:
        each_faction_dataset = pickle.load(fd)

    # get model parameters
    num_round = params['training']['num-rounds']
    param = {'num_leaves': params['training']['num-leaves'], 'objective': 'regression'}

    # get dataset split parameters
    trainsplit = params['training']['train-proportion']
    valsplit = params['training']['val-proportion']
    testsplit = params['training']['test-proportion']
    assert trainsplit + valsplit + testsplit == 1, "dataset train/val/test split /= 1"

    for faction in each_faction_dataset.keys():
        print(faction)

        # make the data
        Xdata = each_faction_dataset[faction]['features']
        Xdata = Xdata.drop(['Unnamed: 0', 'game'], axis=1)
        trainidx = math.ceil(Xdata.shape[0] * trainsplit)
        validx = math.ceil(Xdata.shape[0] * (trainsplit + valsplit))
        traindata = Xdata.iloc[:trainidx, :]
        ytrain = np.array(each_faction_dataset[faction]['vp'].iloc[:trainidx])
        valdata = Xdata.iloc[trainidx:validx, :]
        yval = np.array(each_faction_dataset[faction]['vp'].iloc[trainidx:validx])
        testdata = Xdata.iloc[validx:, :]
        ytest = np.array(each_faction_dataset[faction]['vp'].iloc[validx:])

        traindata = lgb.Dataset(traindata, label=ytrain)

        # train model
        bst = lgb.train(param, traindata, num_round)

        # save model
        bst.save_model(modeldir + f'/{faction}_model.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input DVC params.')
    parser.add_argument('--params', type=str)
    args = parser.parse_args()
    paramsdir = args.params

    with open(paramsdir, 'r') as fd:
        params = yaml.safe_load(fd)

    main(params)
