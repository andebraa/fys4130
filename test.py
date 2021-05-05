import numpy as np

L = 10
L1=10
np.random.seed(42069420)
M = np.random.randint(0, 2, size=(L, L1), dtype='int')

def boundary(indx, tuple = False):
    """
    the remainder in the devision will assert periodic bounadry conditions
    also makes search_neighbours useful for L1=1 aswell
    args:
        indx (tuple of ints): index in matrix
    """
    L = 10
    if tuple:
        return ((indx[0] + L) % L, (indx[1] + L) % L)
    else:
        return (indx + L) % L

def search_neighbours(M, neighbours, visited, site):
    new_neighbour = []
    x = np.array((1, 0))
    y = np.array((0, 1))
    print(site)
    print(site+y)
    print(site + x)

    if (M[site[0], boundary(site[1] + 1)] != M[site[0], site[1]]) and (boundary(site + y, tuple =True)
                                                     ) not in neighbours:  # neightbour aligned and not already in list
        neighbours.append((site[0], boundary(site[1]  + 1)))
        new_neighbour.append((site[0], boundary(site[1]  + 1)))
        visited.append((site[0], boundary(site[1]  + 1)))
        print('up')

    if (M[site[0], boundary(site[1] - 1)] != M[site[0], site[1]]) and (boundary(site - y, tuple =True)
                                                     ) not in neighbours:  # neightbour aligned and not already in list
        neighbours.append((site[0], boundary(site[1]  - 1)))
        new_neighbour.append((site[0], boundary(site[1]  - 1)))
        visited.append((site[0], boundary(site[1]  + 1)))
        print('down')


    if (M[boundary(site[0]+1), site[1]] != M[site[0], site[1]]) and (boundary(site + x, tuple =True)
                                                     ) not in neighbours:  # neightbour aligned and not already in list
        neighbours.append((boundary(site[0]+1), site[1]))
        new_neighbour.append((boundary(site[0]+1), site[1]))
        visited.append((site[0], boundary(site[1]  + 1)))
        print('right')


    if (M[boundary(site[0]-1), site[1]] != M[site[0], site[1]]) and (boundary(site - x, tuple =True)
                                                     ) not in neighbours:  # neightbour aligned and not already in list
        neighbours.append((boundary(site[0]-1), site[1]))
        new_neighbour.append((boundary(site[0]-1), site[1]))
        visited.append((site[0], boundary(site[1]  + 1)))
        print('left')

    return neighbours, new_neighbour

dimension = 2
neighbours = []
visited = []
init_idx = np.random.randint(0, L, size=(dimension))
M[boundary(init_idx, tuple=True)]*=-1
visited.append(init_idx)
neighbours, new_neighbour = search_neighbours(M, neighbours, visited, init_idx)
print(neighbours)
print(new_neighbour)
print(init_idx)
print(M)
M[boundary(init_idx,tuple=True)] = 99
print(M)
