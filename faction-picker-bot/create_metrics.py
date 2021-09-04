import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
import numpy as np
import argparse
import pickle
import yaml
import os


def main(pickledir, modeldir, metricsdir):
    Path(metricsdir).mkdir(parents=True, exist_ok=True)
    models = os.listdir(modeldir)

    with open(pickledir, 'rb') as fd:
        each_faction_dataset = pickle.load(fd)

    for model in models:
        # load model
        modelfile = Path(modeldir, model)
        faction = model.split('_')[0]
        bst = lgb.Booster(model_file=modelfile)  # init model

        # make the data
        Xdata = each_faction_dataset[faction]['features']
        Xdata = Xdata.drop(['Unnamed: 0', 'game'])
        traindata = lgb.Dataset(Xdata, label=np.array(each_faction_dataset[faction]['vp']))

        # get predictions & residuals
        ypred = bst.predict(Xdata)
        residuals = np.array(ypred) - np.array(each_faction_dataset[faction]['vp'])
        avgres = np.mean(residuals)

        # make plots
        line = list(range(250))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title(f'{faction} residuals')
        ax1.plot(ypred, np.array(each_faction_dataset[faction]['vp']), 'bo', line, line, 'r--')
        ax1.set(xlabel='y_pred', ylabel='y_real')
        # ax1.xlim([0, 250])
        # ax1.ylim([0, 250])

        ax2.set_title(f'{faction} residuals histogram')
        ax2.hist(residuals, bins=100)
        ax2.set(xlabel='Difference +/- of y_pred relative to y_real')
        h = ax2.plot([avgres, avgres], [0, 600], 'r--')

        # save plot
        plt.savefig(f'{faction} charts.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input DVC params.')
    parser.add_argument('--params', type=str)
    args = parser.parse_args()
    paramsdir = args.params

    with open(paramsdir, 'r') as fd:
        params = yaml.safe_load(fd)

    pickledir = params['prepare-step2']['pickle-dir']
    modeldir = params['training']['model-dir']
    metricsdir = params['create-metrics']['metrics-dir']

    main(pickledir, modeldir, metricsdir)
