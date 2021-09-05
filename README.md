# terrabot
Visualise &amp; play (or advise plays for) Terra Mystica.

## Organisation of repo
Currently, there are two main aspects to the repository: 
1. Visualising map in terms of distance from a faction's home terrain. To run this, clone the repo
   and enter `python visualisations/terravisgui.py`
2. Developing a model that learns which faction to pick based on starting information (round tiles,
   bonus tiles, other factions picked). So far, the data processing step has been made for this.
   
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

## Tools 
[DVC](https://dvc.org/) for linking the data to github.  
[CML](https://cml.dev/) for ML model development evaluations.  
[Gradio](https://gradio.app/) for interface to model by non-coders.  
[Shapely](https://github.com/slundberg/shap) for charts of feature importance for each game.


## Currently in Progress
1. Set up DVC & connect to AWS for practice (done)
2. add data prep to dvc pipeline (done)
3. Add extra prep for making data for each action to pipeline (done)
4. add model training step to dvc pipeline (done)
5. Add shapely metrics viewer (in progress)
6. Add CML for model dev
7. Add gradio for model interface (Can link this to website?)

