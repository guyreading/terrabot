import numpy as np


# instead of the following, I could have plotted the original map, and a mask of the rivers. Then, for the original map,
# I could have arbitrarily set... green as 1, then clockwise round. Then for each faction, say witches, made everything
# spin up and round 2. so green is 3. Make witches 3, and then minus 3 from all tiles. The modulus is the amount of tile
# spacings away everything is. Then apply the mask for the rivers at the end.

def faction_map(faction):
    """faction_map takes a string input argument, where faction refers to a faction name, and outputs a square array
     which shows the amount of digs every tile is away from their home tile"""
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
    finalArr = np.delete(arr3, [25, 51, 77, 103])
    return finalArr
