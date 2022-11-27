import gradio as gr
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import lightgbm as lgb
import yaml
import numpy as np
import sys
import os

current_path = os.getcwd() + '\\' + __file__
vis_path = os.path.dirname(os.path.dirname(current_path)) + '\\visualisations'
sys.path.append(vis_path)
import terravisualisation as tmvis

matplotlib.use("Agg")

bontiledict = {'SPD + 2C':'BON1', 
               'cult + 4C':'BON2', 
               '+6C':'BON3', 
               '+3pw 1 ship':'BON4', 
               '+1W + 3PW':'BON5', 
               'pass-vp:SA/SH*4 + 2W':'BON6', 
               'pass-vp:TP*2 + 1W':'BON7', 
               '+1P':'BON8', 
               'pass-vp:D*1 + 2C':'BON9', 
               'pass-vp: ship*3 + 3pw':'BON10' 
                }

bontiledict_reverse = {'BON1': 'SPD + 2C',
               'BON2': 'cult + 4C',
               'BON3': '+6C',
               'BON4': '+3pw 1 ship',
               'BON5': '+1W + 3PW',
               'BON6': 'pass-vp:SA/SH*4 + 2W',
               'BON7': 'pass-vp:TP*2 + 1W',
               'BON8': '+1P',
               'BON9': 'pass-vp:D*1 + 2C',
               'BON10': 'pass-vp: ship*3 + 3pw'      
                }

round_tiles_dict = {'SPADE >> 2':'SCORE1',
                    'TOWN >> 5':'SCORE2',
                    'D >> 2':'SCORE3',
                    'SA/SH >> 5':'SCORE4',
                    'D >> 2':'SCORE5',
                    'TP >> 3':'SCORE6',
                    'SA/SH >> 5':'SCORE7',
                    'TP >> 3':'SCORE8',
                    'TE >> 4':'SCORE9'}

round_tiles_dict_reverse = {'SCORE1': 'SPADE >> 2',  
                    'SCORE2': 'TOWN >> 5',  
                    'SCORE3': 'D >> 2',  
                    'SCORE4': 'SA/SH >> 5',  
                    'SCORE5': 'D >> 2',  
                    'SCORE6': 'TP >> 3',  
                    'SCORE7': 'SA/SH >> 5',  
                    'SCORE8': 'TP >> 3',  
                    'SCORE9': 'TE >> 4'}


round_tiles = list(round_tiles_dict.keys())
bontiles = list(bontiledict.keys())

factions = ['Witches', 'Auren', 'Giants', 'Chaos Magicians', 'Darklings', 'Alchemists',
            'Swarmlings', 'Mermaids', 'Fakirs', 'Nomads', 'Engineers', 'Dwarves', 'Halflings', 'Cultists']

players = ['2players', '3players', '4players', '5players']

maps = ['map1', 'map2', 'map3']

faction_cols = ['Yellow', 'Red', 'Grey', 'Black', 'Blue', 'Green', 'Brown']

with open('params.yaml', 'r') as fd:
    params = yaml.safe_load(fd)

vpdfdir = params['prepare']['vp-data-dir']
featdfdir = params['prepare']['feature-data-dir']
pickledir = params['prepare-step2']['pickle-dir']

feature_columns = ['x0_SCORE1', 'x0_SCORE2', 'x0_SCORE3', 'x0_SCORE4', 'x0_SCORE5',
       'x0_SCORE6', 'x0_SCORE7', 'x0_SCORE8', 'x0_SCORE9', 'x1_SCORE1',
       'x1_SCORE2', 'x1_SCORE3', 'x1_SCORE4', 'x1_SCORE5', 'x1_SCORE6',
       'x1_SCORE7', 'x1_SCORE8', 'x1_SCORE9', 'x2_SCORE1', 'x2_SCORE2',
       'x2_SCORE3', 'x2_SCORE4', 'x2_SCORE5', 'x2_SCORE6', 'x2_SCORE7',
       'x2_SCORE8', 'x2_SCORE9', 'x3_SCORE1', 'x3_SCORE2', 'x3_SCORE3',
       'x3_SCORE4', 'x3_SCORE5', 'x3_SCORE6', 'x3_SCORE7', 'x3_SCORE8',
       'x3_SCORE9', 'x4_SCORE1', 'x4_SCORE2', 'x4_SCORE3', 'x4_SCORE4',
       'x4_SCORE5', 'x4_SCORE6', 'x4_SCORE7', 'x4_SCORE8', 'x4_SCORE9',
       'x5_SCORE2', 'x5_SCORE3', 'x5_SCORE4', 'x5_SCORE5', 'x5_SCORE6',
       'x5_SCORE7', 'x5_SCORE8', 'x5_SCORE9', 'BON1', 'BON2', 'BON3', 'BON4',
       'BON5', 'BON6', 'BON7', 'BON8', 'BON9', 'BON10', 'no_players', 'red',
       'blue', 'green', 'black', 'grey', 'yellow', 'brown', 'x0_map1',
       'x0_map2', 'x0_map3']



def args_to_features(*args):
    # round1, round2, round3, round4, round5, round6, faction, map, playerschosen, bon_tiles, fac_cols = args
    Xdata = pd.DataFrame(data=np.zeros((1, len(feature_columns))), columns=feature_columns)

    for arg_no, user_input in enumerate(args):
        if  arg_no in range(6):  # if it's a round
            # map back to col name
            feat_label_name = f'x{arg_no}_{round_tiles_dict[user_input]}'
            Xdata[feat_label_name].iloc[0] = 1
        elif arg_no == 6:
            faction = user_input
            if faction == 'Chaos Magicians':
                faction = 'chaosmagicians'
        elif arg_no == 7: # map 
            feat_label_name = f'x0_{user_input}'
            Xdata[feat_label_name].iloc[0] = 1
        elif arg_no == 8: # playerschosen
            Xdata['no_players'].iloc[0] = int(user_input[0])
        elif arg_no == 9: # bon_tiles
            for bon_tile in user_input:
                Xdata[bontiledict[bon_tile]].iloc[0] = 1
        elif arg_no == 9: # fac_cols
            for fac_col in user_input:
                Xdata[fac_col.lower()].iloc[0] = 1

    return Xdata, faction

def display_map(faction, map):
    map_fig = plt.figure(tight_layout=True)

    x, y = tmvis.display_map(faction, plot=False)
    a = map_fig.add_subplot(111)
    a.hexbin(x, y, gridsize=(19, 9), cmap='magma')
    a.axis('off')
    return map_fig


def predict(*args):
    Xdata, faction = args_to_features(*args)

    modelfile = f'D://PycharmProjects/TerraBot/data/faction-picker-bot/models/{faction}_model.txt'
    bst = lgb.Booster(model_file=modelfile)

    return f'Final score: {round(bst.predict(Xdata)[0])}'


def interpret(*args):
    Xdata, faction = args_to_features(*args)
    modelfile = f'D://PycharmProjects/TerraBot/data/faction-picker-bot/models/{faction}_model.txt'
    bst = lgb.Booster(model_file=modelfile)
    bst.params["objective"] = "regression"
    explainer = shap.Explainer(bst)

    copycols = []
    for ii, column in enumerate(Xdata.columns):
        if column[-6:] in round_tiles_dict_reverse.keys():
            copycols.append(column[:3] + round_tiles_dict_reverse[column[-6:]])
        elif column in bontiledict_reverse.keys():
            copycols.append(bontiledict_reverse[column])
        else:
            copycols.append(column)
        
    Xdata.columns = copycols

    shap_values = explainer(Xdata)
    fig_m = plt.figure(tight_layout=True, facecolor=(0.125,0.172,0.203))
    ax = plt.gca()
    ax.set_facecolor((0.125,0.172,0.203))
    matplotlib.rcParams['axes.labelcolor'] = 'w'
    shap.plots.waterfall(shap_values[0])
    # shap.initjs()
    # shap.plots.force(shap_values[0])
    return fig_m



with gr.Blocks() as demo:
    gr.Markdown("""
    **Predict final faction score given the initial board setup 💰**:  This model uses an lightgbm regression to make prediction. 
    The [source code for this work is here](https://github.com/guyreading/terrabot/faction-picker-bot/gradio_interface.py).
    """)
    with gr.Row():
        with gr.Column():
            faction = gr.Dropdown(
                label="Faction",
                choices=factions,
                value=lambda: random.choice(factions),
            )

            round1_tile = gr.Dropdown(
                label="Round 1 tile",
                choices=round_tiles,
                value=lambda: random.choice(round_tiles),
            )

            round2_tile = gr.Dropdown(
                label="Round 2 tile",
                choices=round_tiles,
                value=lambda: random.choice(round_tiles),
            )

            round3_tile = gr.Dropdown(
                label="Round 3 tile",
                choices=round_tiles,
                value=lambda: random.choice(round_tiles),
            )

            round4_tile = gr.Dropdown(
                label="Round 4 tile",
                choices=round_tiles,
                value=lambda: random.choice(round_tiles),
            )

            round5_tile = gr.Dropdown(
                label="Round 5 tile",
                choices=round_tiles,
                value=lambda: random.choice(round_tiles),
            )

            round6_tile = gr.Dropdown(
                label="Round 6 tile",
                choices=round_tiles,
                value=lambda: random.choice(round_tiles),
            )

            bon_tiles_gr = gr.CheckboxGroup(label='Bonus tiles present', choices=list(bontiledict.keys()))
        
            map = gr.Dropdown(
                label="Map",
                choices=maps,
                value=lambda: random.choice(maps),
            )

            playerschosen = gr.Dropdown(
                label="No. Of Players",
                choices=players,
                value=lambda: random.choice(players),
            )

            fac_cols_gr = gr.CheckboxGroup(label='Other faction colours present', choices=faction_cols)


        with gr.Column():
            map_plot = gr.Plot(label='Distance from home terrain: darker is further')

            with gr.Row():
                predict_btn = gr.Button(value="Predict")
                interpret_btn = gr.Button(value="Explain")

            label = gr.Label(label=f'Prediction of final VP for faction:')
            plot = gr.Plot(label=f'Breakdown of prediction for faction:')

    predict_btn.click(
        predict,
        inputs=[
            round1_tile,
            round2_tile,
            round3_tile,
            round4_tile,
            round5_tile,
            round6_tile,
            faction,
            map,
            playerschosen,
            bon_tiles_gr,
            fac_cols_gr
        ],
        outputs=[label],
    )
    interpret_btn.click(
        interpret,
        inputs=[
            round1_tile,
            round2_tile,
            round3_tile,
            round4_tile,
            round5_tile,
            round6_tile,
            faction,
            map,
            playerschosen,
            bon_tiles_gr,
            fac_cols_gr
        ],
        outputs=[plot],
    )

    faction.change(
        display_map,
        inputs=[
            faction,
            map,
        ],
        outputs=[map_plot],
    )

    map.change(
        display_map,
        inputs=[
            faction,
            map,
        ],
        outputs=[map_plot],
    )

demo.launch()