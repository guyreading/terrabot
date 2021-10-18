# terrabot
Visualise &amp; play (or advise plays for) Terra Mystica.

## Setting Up The Environment
1. Clone this repository and cd into it.
2. Create your TerraBot conda environment:
`conda env create --file terrabot-conda-file.yml`  
3. Re-create the most recent faction-picker-bot experiment pipeline: `dvc repro`
4. Look at the dvc `params.yaml` file if you'd like to adjust managed aspects of the current model, 
   else you can adjust the training script yourself. Looking at `dvc.yaml` will show you
   the data creation/model training pipeline. After having run `dvc repro` the first time, 
   on changing the model you can run `dvc metrics show` to compare your model with the current
   model.


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
[MACEst](https://github.com/oracle/macest) for working out certainty values


## Current Work Progress/Plan
1. [x] Set up DVC & connect to AWS for practice
2. [x] add data prep to dvc pipeline
3. [x] Add extra prep for making data for each action to pipeline
4. [x] add model training step to dvc pipeline
5. [x] Adding extra options for making data (ordinal vs one-hot)
6. [x] Add training score vs epoch graphs & add this as a dvc plot
7. [ ] Make model MSE better (currently overfitting)
8. [ ] Add CML for model dev
9. [ ] Add gradio for model interface (Can link this to website?)
10. [ ] Use MACEst for working out certainty values
11. [ ] (Optional) put into/run within a Docker container for easier collaboration
12. [ ] (Optional) practice training in Sagemaker (+ ECR container?)
13. [ ] Add shapely metrics viewer

