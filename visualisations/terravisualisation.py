"""terravisualisation contains all the data processing & matplotlib graphing functions related to board visualisation
functions for terraBot. However for an interaction GUI, this function is contained in "terravisgui". This GUI uses
the tools built in terravisualisation.
"""

import numpy as np


def rotate_around_dig_wheel(gremangles, rotation):
    from math import pi

    # rotate
    blumangles = (gremangles + ((rotation / 7) * pi) + pi) % (2 * pi) - pi

    # back to numbers
    bluemap = blumangles / (pi * (2 / 7))

    # create the array
    rotatedmap = 4 - abs(bluemap)

    return rotatedmap


def faction_map(faction):
    """faction_map outputs an array representing the Terra Mystica map, where they array
    shows the amount of digs every tile is away from a faction's home tile

    Arguments:
    faction -- a string referring to a TM faction

    Returns:
    finalarr -- a (9, 13) array where each element is [4 - the amount of digs a faction is away
    from their home tile], where an element maps to a tile. Rivers have a value of 0. Rows of the
    map constituting of 12 tiles have their final element set to 0, representing a blank tile
     """
    from math import pi

    # for green
    greenmap = np.array([[-3, 1, 0, -1, 3, 2, -3, -2, 2, 0, -1, 2, -2],
                         [3, 4, 4, -3, -2, 4, 4, 3, -2, 4, 4, 3, 4],
                         [4, 4, -2, 4, 1, 4, 0, 4, 0, 4, 1, 4, 4],
                         [0, -1, 3, 4, 4, 2, -1, 4, 2, 4, 2, -3, 4],
                         [-2, -3, 2, -1, -2, -3, 1, 3, 4, 4, 0, -2, -1],
                         [1, 0, 4, 4, 3, 0, 4, 4, 4, -3, 1, -3, 4],
                         [4, 4, 4, 1, 4, 2, 4, 0, 4, 3, -2, -1, 3],
                         [3, -1, -3, 4, 4, 4, -1, -2, 4, 1, -3, 1, 4],
                         [2, -2, 1, -1, 2, 0, 3, -3, 1, 4, -1, 0, 2]])

    gremangles = greenmap * (pi * (2/7))
        
    # river & empties mask
    rivermask = np.array([[False, False, False, False, False, False, False, False, False, False, False, False, False],
                          [False, True, True, False, False, True, True, False, False, True, True, False, True],
                          [True, True, False, True, False, True, False, True, False, True, False, True, True],
                          [False, False, False, True, True, False, False, True, False, True, False, False, True],
                          [False, False, False, False, False, False, False, False, True, True, False, False, False],
                          [False, False, True, True, False, False, True,  True,  True, False, False, False, True],
                          [True, True, True, False, True, False, True, False, True, False, False, False, False],
                          [False, False, False, True, True, True, False, False, True, False, False, False, True],
                          [False, False, False, False, False, False, False, False, False, True, False, False, False]])

    if faction == 'witches' or faction == 'auren':
        # create the array
        arr1 = 4 - abs(greenmap)

    elif faction == 'swarmlings' or faction == 'mermaids':
        arr1 = rotate_around_dig_wheel(gremangles, 2)

    elif faction == 'darklings' or faction == 'alchemists':
        arr1 = rotate_around_dig_wheel(gremangles, 4)

    elif faction == 'halflings' or faction == 'cultists':
        arr1 = rotate_around_dig_wheel(gremangles, 6)

    elif faction == 'engineers' or faction == 'dwarves':
        arr1 = rotate_around_dig_wheel(gremangles, -2)

    elif faction == 'chaos magicians' or faction == 'giants':
        arr1 = rotate_around_dig_wheel(gremangles, -4)

    elif faction == 'fakirs' or faction == 'nomads':
        arr1 = rotate_around_dig_wheel(gremangles, -6)

    else:
        return

    # apply river spacing mask over
    # alt: arr1 = np.multiply(arr1, 1 - rivermask)
    arr1[rivermask] = 0

    # turn back to integer from potential float
    arr1 = arr1.astype(int)

    # flip
    arr2 = np.flip(arr1, 0)

    # re-size
    arr3 = np.squeeze(np.resize(arr2, (1, 117)))

    # remove the added hexes
    finalarr = np.delete(arr3, [25, 51, 77, 103])

    return finalarr


def display_map(faction, plot=True):
    """takes the input, faction, and returns a map of the board
    where hex brightness relates to how many digs that faction
    needs to convert that hex into its home territory.

    Arguments:
    faction -- string or (9, 13) numpy array: where the string refers to a faction name,
    or the numpy array refers to map to be plotted in the format of faction_map

    Returns:
    None. Plots the map.
    """
    import matplotlib.pyplot as plt
    import numpy.matlib

    # make sure the faction string is all lowercase
    faction = faction.lower()

    # create array to display map on
    x1 = np.linspace(4.5, 16.5, 13)
    x2 = np.linspace(5, 16, 12)
    x3 = np.hstack((x1, x2))
    x4 = np.matlib.repmat(x3, 1, 4)
    x4 = np.squeeze(x4)
    x5 = np.hstack((x4, x1))
    x = np.squeeze(x5)

    y1 = np.linspace(6, 15, 9)
    y = np.repeat(y1, np.array([13, 12, 13, 12, 13, 12, 13, 12, 13]))

    if type(faction) == str:
        factionmap = faction_map(faction)
    else:  # assume it's already a map mask array
        factionmap = faction

    x = np.repeat(x, factionmap)
    y = np.repeat(y, factionmap)

    # need to define the size of the plot
    x = np.hstack((x, [1, 1, 20, 20]))
    y = np.hstack((y, [1, 20, 1, 20]))

    if plot:
        plt.hexbin(x, y, gridsize=(19, 9), cmap='magma')
        plt.show()
    else:
        return x, y
