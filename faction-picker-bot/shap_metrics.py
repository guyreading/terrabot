import lightgbm as lgb
import argparse
import yaml
import shap

def main():
    test = 1


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
    metricsdir2 = params['create-metrics']['metrics-dir2']

    main(pickledir, modeldir, metricsdir, metricsdir2)