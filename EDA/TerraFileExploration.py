import pandas as pd

folderLocation = "C:/Users/User1/PycharmProjects/TerraBot/terra-mystica"

games = pd.read_csv(f'{folderLocation}/games.csv')
game_events = pd.read_csv(f'{folderLocation}/game_events.csv')

# the extremely basic
print(f"There are {game_events['event'].nunique()} distinct moves in game_events.")
print(f"There are {game_events['game'].nunique()} distinct games in game_events.")
print(f"There are {game_events['round'].nunique()} distinct rounds in game_events.") # 7 - this is kinda wierd
print(f"There are {game_events['turn'].nunique()} distinct turns in game_events.") # 28!!!

uniqueMoves = pd.unique(game_events['event']) # okay, so leeching, passing & ordering etc is an "event". Makes more sense.
# so to categorise these, we've got:
# 7 "pick-color"s
# 13 "favor:"s (favor:any??)
# 7 "order:"'s (order 6 & 7?!)
# 2: advance:ship, advance:dig
# 6 cultist power related: 'leech-from-cultist:pw', 'decline-from-cultist:count', 'leech-from-cultist:count', 'decline-from-cultist:pw', 'cultist:cult', 'cultist:pw'
# 9 pass:BON
# 7 town:TW
#

moveChunking = dict()
uniqueMovesList = uniqueMoves.tolist()

chunkingVals = ["upgrade", "order", "pass", "action", "favor", "pick-color", "town", "decline", "advance", "leech"]

for ii in chunkingVals:
    matching = [s for s in uniqueMovesList if ii in s]
    moveChunking[ii] = matching
    for s in matching:
        uniqueMovesList.remove(s)

# values that require different addressing
matching = uniqueMovesList[5:8]
moveChunking["faction-specific"] = matching
for s in matching:
    uniqueMovesList.remove(s)

#everything else
moveChunking["Misc"] = uniqueMovesList