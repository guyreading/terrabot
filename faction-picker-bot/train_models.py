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
    param = {'num_leaves': params['training']['num-leaves'],
             'objective': 'regression',
             'max_depth': params['training']['max-depth'],
             'boosting_type': params['training']['boosting-type']
             }

    # get dataset split parameters
    trainsplit = params['training']['train-proportion']
    valsplit = params['training']['val-proportion']
    testsplit = params['training']['test-proportion']
    assert trainsplit + valsplit + testsplit == 1, "dataset train/val/test split != 1"

    for faction in each_faction_dataset.keys():
        print(faction)

        # make the data
        Xdata = each_faction_dataset[faction]['featuresnp'][:, 1:]  # remove game
        trainidx = math.ceil(Xdata.shape[0] * trainsplit)
        validx = math.ceil(Xdata.shape[0] * (trainsplit + valsplit))
        traindata = Xdata[:trainidx, :]
        ytrain = np.array(each_faction_dataset[faction]['vp'].iloc[:trainidx])
        valdata = Xdata[trainidx:validx, :]
        yval = np.array(each_faction_dataset[faction]['vp'].iloc[trainidx:validx])
        # testdata = Xdata[validx:, :]
        # ytest = np.array(each_faction_dataset[faction]['vp'].iloc[validx:])

        traindataset = lgb.Dataset(traindata, label=ytrain)
        valdataset = lgb.Dataset(valdata, label=yval)

        path = 'path2'

        if path == 'path1':
            # PATH 1
            # train model
            evaldict = {}
            model = lgb.train(param,
                              traindataset,
                              num_round,
                              valid_sets=[valdataset, traindataset],
                              valid_names=['validation', 'train'],
                              early_stopping_rounds=200,
                              evals_result=evaldict,
                              verbose_eval=False
                              )
        elif path == 'path2':
            # PATH 2:
            increment = round(len(traindata) * 0.29)
            data_idx = increment
            split_rounds = params['training']['split-rounds']
            evaldictslist = []

            model = lgb.LGBMRegressor(
                boosting_type=params['training']['boosting-type'],
                n_estimators=num_round,
                learning_rate=0.1,
                num_leaves=params['training']['num-leaves'],
                max_depth=params['training']['max-depth'],
            )

            for roundno, singleround in enumerate(range(split_rounds)):
                print(f'round: {roundno}')
                evaldict = {}
                evalcallback = lgb.record_evaluation(evaldict)
                start = data_idx % len(traindata)
                data_idx += increment
                end = data_idx % len(traindata)
                if end < start:
                    xsubset = np.vstack((traindata[start:, :], traindata[:end, :]))
                    ysubset = np.concatenate((ytrain[start:], ytrain[:end]))
                else:
                    xsubset = traindata[start:end, :]
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

        # save model
        model.save_model(modeldir + f'/{faction}_model.txt')

        if path == 'path1':
            # save eval results
            pickle.dump(evaldict, open(modelmetricsdir + f'{faction}_results.pkl', 'wb'))
        if path == 'path2':
            # save eval results
            pickle.dump(evaldictslist, open(modelmetricsdir + f'{faction}_results.pkl', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input DVC params.')
    parser.add_argument('--params', type=str)
    args = parser.parse_args()
    paramsdir = args.params

    with open(paramsdir, 'r') as fd:
        params = yaml.safe_load(fd)

    main(params)
