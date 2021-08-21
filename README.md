# terrabot
Visualise &amp; play (or advise plays for) Terra Mystica.

## Organisation of repo
Currently, there are two main aspects to the repository: 
1. Visualising map in terms of distance from a faction's home terrain. To run this, clone the repo
   and enter `python visualisations/terravisgui.py`
2. Developing a model that learns which faction to pick based on starting information (round tiles,
   bonus tiles, other factions picked). The data has been made for this, now.
   
## Data provenance
The original (noted as "local" in the notebooks) data came from the 
[Kaggle dataset](https://www.kaggle.com/lemonkoala/terra-mystica).

A future development direction is to use the 
[orignal snellman data from source](https://terra.snellman.net/data), however json
manipulation is needed and for now the local csv data is used for ease.

## Faction picking
The data for a faction picker bot is hosted on 
[kaggle terra mystica faction picker data](https://www.kaggle.com/guyar1/terra-mystica-faction-picker-data). 

The script used to create this data can be found in this repo at `faction-picker-bot/creatingVPdata.ipynb`.

I am currently in the process of training a model to pick the best faction based on intial game states.

Notes to self:
1. Set up DVC & connect to AWS for practice
2. add data prep to dvc pipeline
3. add model training step to dvc pipeline
