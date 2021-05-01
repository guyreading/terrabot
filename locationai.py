def load_one_game_data(gameno, gameevents=None):
    """
    1. find where the location data is kept & in what format
    2. Associate any dig & build / build action location with the board configuration immediately before the move  &
    with a game & player (to associate with y)
    3. output in an array: 1. location 2. board configuration before move 3. game & player
    """
    import pandas as pd

    folderlocation = "C:/Users/User1/PycharmProjects/TerraBot/terra-mystica"

    games = pd.read_csv(f'{folderlocation}/games.csv')
    locations = pd.read_csv(f'{folderlocation}/game_locations.csv')
    gamest = pd.read_csv(f'{folderlocation}/game_scoring_tiles.csv')
    gamefactions = pd.read_csv(f'{folderlocation}/game_factions.csv')
    gameoptions = pd.read_csv(f'{folderlocation}/game_options.csv')

    # takes a while to load so you might want it pre-loaded once for many games
    if gameevents is None:
        gameevents = pd.read_csv(f'{folderlocation}/game_events.csv')

    # get details for that one game
    ogame = games[games['game'] == gameno]
    olocation = locations[locations['game'] == gameno]
    ogamest = gamest[gamest['game'] == gameno]
    ogamefactions = gamefactions[gamefactions['game'] == gameno]
    ogameevents = gameevents[gameevents['game'] == gameno]
    ogameoptions = gameoptions[gameoptions['game'] == gameno]

    return ogame, olocation, ogamest, ogamefactions, ogameevents, ogameoptions


def location_finder_one_player():
    """"""



def feature_creation(location, boardconfig):
    """Features:
    Digs needed to get
    number of tiles this can open up
    Opponents digs needed (what if multiple opponents?)
    number of tiles open up for opponent (what if multiple opponents?)
    """


def y_creation(data):
    """find the final score for that player for that game
    How to calculate a score the for location tiles not taken? All we know is that their score should be less than the
    location picked (if the player picked optimally, which we don't actually know. We could just use the one they
    picked)
    """

def no_of_tiles_open_feature(data):
    """Creates a number that represents the amount of tiles this location opens up.
    Should be something like: immediate no of tiles opened up. And then is this the
    direction of where most other tiles are? Should this be two features? Yes.
    """

def calculate_available_locations(currentlocations, shippinglvl):
    """Given current location of a faction, work out where all possible areas they could move to are
    """

def nn_train(features, finalscore):
    """Takes features, final score (y) trains a neural network"""
