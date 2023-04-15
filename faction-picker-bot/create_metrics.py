import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
import numpy as np
from sklearn import metrics
import pandas as pd
import argparse
import pickle
import math
import json
import yaml
import os


def main(params):
    pickledir = params['prepare-step2']['pickle-dir']
    modeldir = params['training']['model-dir']
    modelmetricsdir = params['training']['model-metrics-dir']
    metricsdir = params['create-metrics']['metrics-dir']
    metricsdir2 = params['create-metrics']['metrics-dir2']
    metricsdir3 = params['create-metrics']['metrics-dir3']

    # get dataset split parameters
    trainsplit = params['training']['train-proportion']
    valsplit = params['training']['val-proportion']
    testsplit = params['training']['test-proportion']

    Path(metricsdir).mkdir(parents=True, exist_ok=True)
    models = os.listdir(modeldir)
    model_metrics = dict()
    model_plot_pd = pd.DataFrame()

    with open(pickledir, 'rb') as fd:
        each_faction_dataset = pickle.load(fd)

    for model in models:
        # load model
        modelfile = modeldir + model
        faction = model.split('_')[0]
        bst = lgb.Booster(model_file=modelfile)  # init model
        evaldict = pickle.load(open(modelmetricsdir + f'{faction}_results.pkl', 'rb'))

        # make the data
        Xdata = each_faction_dataset[faction]['features']
        validx = math.ceil(Xdata.shape[0] * (trainsplit + valsplit))
        testdata = Xdata.iloc[validx:, :]
        ytest = np.array(each_faction_dataset[faction]['vp'].iloc[validx:])

        # get predictions & residuals
        ypred = bst.predict(testdata)
        residuals = np.array(ypred) - ytest
        avgres = np.mean(residuals)
        original = np.array(ytest) - np.mean(ytest)
        MAE = metrics.mean_absolute_error(ypred, ytest)
        model_metrics[faction] = MAE

        # make plots
        line = list(range(250))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(12, 4.8)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title(f'{faction} residuals')
        ax1.plot(ypred, ytest, 'bo', line, line, 'r--')
        ax1.set(xlabel='y_pred', ylabel='y_real')
        # ax1.xlim([0, 250])
        # ax1.ylim([0, 250])

        ax2.set_title(f'{faction} residuals histogram')
        ax2.hist(original, bins=100, color='r')
        ax2.hist(residuals, bins=100)
        ax2.set(xlabel='Difference +/- of y_pred relative to y_real')
        h = ax2.plot([avgres, avgres], [0, 100], 'r--')

        # get data for fig 3
        if type(evaldict) == list:
            trainmetrics = []
            valmetrics = []
            dictmetric = list(evaldict[0]['train'].keys())[0]
            for eachevaldict in evaldict:
                trainmetrics += eachevaldict['train'][dictmetric]
                valmetrics += eachevaldict['validation'][dictmetric]
        else:
            dictmetric = list(evaldict['train'].keys())[0]
            trainmetrics = evaldict['train'][dictmetric]
            valmetrics = evaldict['validation'][dictmetric]

        ax3.set_title(f'{faction} training plot')
        ax3.plot(range(len(trainmetrics)), trainmetrics, 'r', label='train')
        ax3.plot(range(len(valmetrics)), valmetrics, 'g', label='validation')
        ax3.set(xlabel='train step', ylabel='l2 loss')
        ax3.legend()
        model_plot_pd[faction] = valmetrics

        plt.subplots_adjust(wspace=0.8)

        # save plot
        plt.savefig(metricsdir + f'/{faction} charts.png')

    with open(metricsdir2, 'w') as fp:
        json.dump(model_metrics, fp)

    model_plot_pd['step'] = list(range(model_plot_pd.shape[0]))
    model_plot_pd.to_csv(metricsdir3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input DVC params.')
    parser.add_argument('--params', type=str)
    args = parser.parse_args()
    paramsdir = args.params

    with open(paramsdir, 'r') as fd:
        params = yaml.safe_load(fd)

    main(params)
