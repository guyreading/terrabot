import lightgbm as lgb
from pathlib import Path
import argparse
import pickle
import yaml
import shap
import os


def main(pickledir, modeldir, shapdir):
    Path(shapdir).mkdir(parents=True, exist_ok=True)
    models = os.listdir('../' + modeldir)

    with open(pickledir, 'rb') as fd:
        each_faction_dataset = pickle.load(fd)

    for model in models:
        # load model
        modelfile = modeldir + model
        faction = model.split('_')[0]
        bst = lgb.Booster(model_file=modelfile)  # init model
        bst.params["objective"] = "regression"

        # load data
        Xdata = each_faction_dataset[faction]['features']
        Xdata = Xdata.drop(['Unnamed: 0', 'game'], axis=1)

        explainer = shap.Explainer(bst)
        shap_values = explainer(Xdata)

        factionshap = shapdir + faction + '_shap.pkl'
        with open(factionshap, 'w') as fp:
            pickle.dump(shap_values, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input DVC params.')
    parser.add_argument('--params', type=str)
    args = parser.parse_args()
    paramsdir = args.params

    with open(paramsdir, 'r') as fd:
        params = yaml.safe_load(fd)

    pickledir = params['prepare-step2']['pickle-dir']
    modeldir = params['training']['model-dir']
    shapdir = params['shap-metrics']['shap-dir']

    main(pickledir, modeldir, shapdir)