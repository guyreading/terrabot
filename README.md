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
8. [ ] Build within CML for model dev
9. [x] Add gradio for model interface (Can link this to website?)
10. [ ] Use MACEst for working out certainty values
11. [ ] (Optional) put into/run within a Docker container for easier collaboration
12. [x] Add shapely metrics viewer

## Progress (Verbose)
### Report 23-10-21
DVC pipeline in place. We have multiple ways of creating the features and LightGBM can be adjusted 
in the params.yaml, so we never have to touch the code now. A number of experiments have been run 
however none of them do much better than taking the average score. Multiple LightGBM hyperparameters 
have been used to test, however no tree can fit much better. The features have been checked (making 
sure the indexes line up so sample in X = same sample in Y) so it's not the features.

Two thoughts come to mind:
1. The bad players's games are polluting the data. Bad players will perform badly, regardless of starting 
conditions. We could remove these players to see if the model performs better.
2. We could add another feature (or set of featuers) for the faction we want to predict's player skill, 
plus we could add the opponents' player skill. Thoughts against this is that this feature might dominate 
   but I really want to find features in the starting state of the game, not that good players will play well.

### Report 28-11-22
Gradio interface was built. It uses the map that can show distance (in digs) from home terrain, it predicts 
final score for any faction given initial setup using the previously trained models, and it explains where 
the score comes from using SHAPly values. 

More thought was given to improving model performance (as noted in previous report).
1. Some games will interfere/contribute towards a players progress more than others. Eg. a 5 player game will 
affect a players game more than a 2 player game. This may add extra noise relative to starting factions. What's more, 
different player skill matching will affect each players' games. 
Good player 1 vs good player 2 will reduce final score of good player 1 relative to playing against a bad player.
2. Do we take bidding into account? 