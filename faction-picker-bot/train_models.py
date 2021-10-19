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
    modelmetricsdir = params['training']['model-metrics-dir']

    Path(modeldir).mkdir(parents=True, exist_ok=True)
    Path(modelmetricsdir).mkdir(parents=True, exist_ok=True)

    with open(pickledir, 'rb') as fd:
        each_faction_dataset = pickle.load(fd)

    # get model parameters
    num_round = params['training']['num-rounds']

    # get dataset split parameters
    trainsplit = params['training']['train-proportion']
    valsplit = params['training']['val-proportion']
    testsplit = params['training']['test-proportion']
    assert trainsplit + valsplit + testsplit == 1, "dataset train/val/test split != 1"

    for faction in each_faction_dataset.keys():
        print(faction)

        # make the data
        Xdata = each_faction_dataset[faction]['features']
        trainidx = math.ceil(Xdata.shape[0] * trainsplit)
        validx = math.ceil(Xdata.shape[0] * (trainsplit + valsplit))
        traindata = Xdata.iloc[:trainidx, :]
        ytrain = np.array(each_faction_dataset[faction]['vp'].iloc[:trainidx])
        valdata = Xdata.iloc[trainidx:validx, :]
        yval = np.array(each_faction_dataset[faction]['vp'].iloc[trainidx:validx])
        # testdata = Xdata.iloc[validx:, :]
        # ytest = np.array(each_faction_dataset[faction]['vp'].iloc[validx:])

        # train model - this is running one of the training scripts below
        training_routine = params['training']['training-routine'] + "(traindata, ytrain, valdata, yval, num_round)"
        model, evaldict = eval(training_routine)

        # save model
        model.save_model(modeldir + f'/{faction}_model.txt')

        # save eval results
        pickle.dump(evaldict, open(modelmetricsdir + f'{faction}_results.pkl', 'wb'))


def lgb_train_method(traindata, ytrain, valdata, yval, num_round):
    """Train a LightGBM from the train script"""
    traindataset = lgb.Dataset(traindata, label=ytrain)
    valdataset = lgb.Dataset(valdata, label=yval)

    model_kwargs = params['training']['model-kwargs']

    # train model
    evaldict = {}
    model = lgb.train(model_kwargs,
                      traindataset,
                      num_round,
                      valid_sets=[valdataset, traindataset],
                      valid_names=['validation', 'train'],
                      early_stopping_rounds=200,
                      evals_result=evaldict,
                      verbose_eval=False
                      )

    return model, evaldict


def lgb_kfolds_scikitlearn(traindata, ytrain, valdata, yval, num_round):
    """Train a LightGBM with the scikitlearn API + some kfolds"""
    increment = round(len(traindata) * 0.29)
    data_idx = increment
    split_rounds = params['training']['split-rounds']
    modelkwargs = params['training']['model-kwargs']
    evaldictslist = []

    model = lgb.LGBMRegressor(
        n_estimators=num_round,
        **modelkwargs
    )

    for roundno in range(split_rounds):
        evaldict = {}
        evalcallback = lgb.record_evaluation(evaldict)
        start = data_idx % len(traindata)
        data_idx += increment
        end = data_idx % len(traindata)
        if end < start:
            xsubset = np.vstack((traindata.iloc[start:, :], traindata.iloc[:end, :]))
            ysubset = np.concatenate((ytrain[start:], ytrain[:end]))
        else:
            xsubset = traindata.iloc[start:end, :]
            ysubset = ytrain[start:end]

        model.fit(
            xsubset, ysubset,
            eval_set=[(valdata, yval), (traindata, ytrain)],
            eval_names=['validation', 'train'],
            verbose=False,
            callbacks=[evalcallback],
        )

        evaldictslist.append(evaldict)

    model = model.booster_

    return model, evaldictslist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input DVC params.')
    parser.add_argument('--params', type=str)
    args = parser.parse_args()
    paramsdir = args.params

    with open(paramsdir, 'r') as fd:
        params = yaml.safe_load(fd)

    main(params)
