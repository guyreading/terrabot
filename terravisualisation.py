"""terravisualisation contains all the board visualisation functions for terraBot
"""

import numpy as np

def faction_map(faction):
    """faction_map takes a string input argument, where faction refers to a faction name, and outputs a square array
     which shows the amount of digs every tile is away from their home tile
     """

    # instead of the following, I could have plotted the original map, and a mask of the rivers. Then, for
    # the original map, I could have arbitrarily set... green as 1, then clockwise round. Then for each
    # faction, say witches, made everything spin up and round 2. so green is 3. Make witches 3, and then minus
    # 3 from all tiles. The modulus is the amount of tile spacings away everything is. Then apply the mask
    # for the rivers at the end.

    # or a very elegant way of doing it would be implement rotation. how to do this? angle could be in 2pi/7.
    # angles that far away, either side (could use an abs() function) are then worked out into spaces away.

    from math import pi
    # for green
    """greenmap = np.array([[-3, 1, 0, -1, 3, 2, -3, -2, 2, 0, -1, 2, -2],
                         [3, 0, 0, -3, -2, 0, 0, -3, -2, 0, 0, -3, 0],
                         [0, 0, -2, 0, 1, 0, 3, 0, 3, 0, 2, 0, 0],
                         [3, 4, 1, 0, 0, 1, 4, 0, 1, 0, 1, 2, 0],
                         [3, 2, 1, 4, 3, 2, 2, 1, 0, 0, 3, 3, 4],
                         [2, 3, 0, 0, 1, 3, 0, 0, 0, 2, 2, 2, 0],
                         [0, 0, 0, 2, 0, 1, 0, 3, 0, 1, 3, 4, 1],
                         [1, 4, 2, 0, 0, 0, 4, 3, 0, 2, 2, 2, 0],
                         [1, 3, 2, 4, 1, 3, 1, 2, 2, 0, 4, 3, 1]]) * 2*pi/7
        
        # river & empties mask                              
        rivermask = np.array([[False, False, False, False, False, False, False, False, False, False, False, False, False],
                              [False, True, True, False, False, True, True, False, False, True, True, False, True],
                              [ True, True, False, True, False, True, False, True, False, True, False, True, True],
                              [False, False, False, True, True, False, False, True, False, True, False, False, True],
                              [False, False, False, False, False, False, False, False, True, True, False, False, False],
                              [False, False, True, True, False, False, True,  True,  True, False, False, False, True],
                              [ True, True, True, False, True, False, True, False, True, False, False, False, False],
                              [False, False, False, True, True, True, False, False, True False False, False, True],
                              [False, False, False, False, False, False, False, False, False, True, False, False, False]])
    """

    if faction == 'swarmlings' or faction == 'mermaids':
        # create the array
        arr1 = np.array([[2, 2, 3, 4, 1, 1, 2, 3, 1, 3, 4, 1, 3],
                         [1, 0, 0, 2, 3, 0, 0, 1, 3, 0, 0, 1, 0],
                         [0, 0, 3, 0, 2, 0, 3, 0, 3, 0, 2, 0, 0],
                         [3, 4, 1, 0, 0, 1, 4, 0, 1, 0, 1, 2, 0],
                         [3, 2, 1, 4, 3, 2, 2, 1, 0, 0, 3, 3, 4],
                         [2, 3, 0, 0, 1, 3, 0, 0, 0, 2, 2, 2, 0],
                         [0, 0, 0, 2, 0, 1, 0, 3, 0, 1, 3, 4, 1],
                         [1, 4, 2, 0, 0, 0, 4, 3, 0, 2, 2, 2, 0],
                         [1, 3, 2, 4, 1, 3, 1, 2, 2, 0, 4, 3, 1]])

    elif faction == 'darklings' or faction == 'alchemists':
        arr1 = np.array([[3, 1, 2, 3, 2, 1, 3, 4, 1, 2, 3, 1, 4],
                         [2, 0, 0, 3, 4, 0, 0, 2, 4, 0, 0, 2, 0],
                         [0, 0, 4, 0, 1, 0, 2, 0, 2, 0, 1, 0, 0],
                         [2, 3, 2, 0, 0, 1, 3, 0, 1, 0, 1, 3, 0],
                         [4, 3, 1, 3, 4, 3, 1, 2, 0, 0, 2, 4, 3],
                         [1, 2, 0, 0, 2, 2, 0, 0, 0, 3, 1, 3, 0],
                         [0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 4, 3, 2],
                         [2, 3, 3, 0, 0, 0, 3, 4, 0, 1, 3, 1, 0],
                         [1, 4, 1, 3, 1, 2, 2, 3, 1, 0, 3, 2, 1]])

    else:
        return

    # flip
    arr2 = np.flip(arr1, 0)

    # re-size
    arr3 = np.squeeze(np.resize(arr2, (1, 117)))

    # remove the added hexes
    finalarr = np.delete(arr3, [25, 51, 77, 103])
    return finalarr


def display_map(faction):
    """takes the input string, faction, and returns a map of the board
    where hex brightness relates to how many digs that faction
    needs to convert that hex into its home territory
    """
    import matplotlib.pyplot as plt
    import numpy.matlib

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

    plt.hexbin(x, y, gridsize=(19, 9), cmap='magma')

    plt.show()
