import pandas as pd
import pickle
import yaml
import os

script_dir = os.path.dirname(__file__)
filename = os.path.join(script_dir, '../params.yaml')

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)

vpdf = pd.read_csv('D://PycharmProjects/TerraBot/data/faction-picker-bot/vpdata.csv')
featdf = pd.read_csv('D://PycharmProjects/TerraBot/data/faction-picker-bot/featdata.csv')

each_faction_dataset = {}

colnames = list(vpdf.columns)
factions = [x for x in colnames if x != 'game' and x != 'Unnamed: 0']

each_faction_dataset = {}

for faction in factions:
    faction_dataset = {}
    indexes = pd.isnull(vpdf[faction])
    faction_dataset['vp'] = vpdf[faction][~indexes]
    faction_dataset['features'] = featdf[~indexes]
    each_faction_dataset[faction] = faction_dataset

