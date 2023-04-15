from pathlib import Path
import numpy as np
import lightgbm as lgb
import argparse
import pickle
import yaml
import math

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# import tensorflow_datasets as tfds
# import tensorflow_probability as tfp


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

    model_kwargs = params['training']['lgbt-model-kwargs']

    # train model
    evaldict = {}
    model = lgb.train(model_kwargs,
                      traindataset,
                      num_round,
                      valid_sets=[valdataset, traindataset],
                      valid_names=['validation', 'train'],
                      early_stopping_rounds=10,
                      evals_result=evaldict,
                      verbose_eval=False
                      )

    return model, evaldict


def lgb_kfolds_scikitlearn(traindata, ytrain, valdata, yval, num_round):
    """Train a LightGBM with the scikitlearn API + some kfolds"""
    increment = round(len(traindata) * 0.29)
    data_idx = increment
    split_rounds = params['training']['split-rounds']
    modelkwargs = params['training']['lgbt-model-kwargs']
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


def nn_train_method(traindata, ytrain, valdata, yval, num_round):
    # initialising hyperparameters
    model_kwargs = params['training']['nn-model-kwargs']
    batch_size = 256
    num_epochs = 100
    
    # create inputs
    inputs = {}
    for feature_name in traindata.columns:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    
    # create model
    input_values = [value for _, value in sorted(inputs.items())]
    features = keras.layers.concatenate(input_values)
    features = layers.BatchNormalization()(features)

    # Create hidden layers with deterministic weights using the Dense layer.
    for units in model_kwargs['hidden_units']:
        features = layers.Dense(units, activation="sigmoid")(features)
    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # train
    if model_kwargs['loss'] == 'mse':
        loss = keras.losses.MeanSquaredError()
    else:
        raise('Unsupported loss present')

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=model_kwargs['learning_rate']),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    model.fit(train_dataset, epochs=model_kwargs['num_epochs'], validation_data=test_dataset)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input DVC params.')
    parser.add_argument('--params', type=str)
    args = parser.parse_args()
    paramsdir = args.params

    with open(paramsdir, 'r') as fd:
        params = yaml.safe_load(fd)

    main(params)
