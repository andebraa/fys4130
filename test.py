import numpy as np

L = 5
L1=5
np.random.seed(4206669)
M = np.random.randint(0, 2, size=(L, L1), dtype='int')
M[np.where(M == 0)] = -1
print(M)
def boundary(indx, tuple = False):
    """
    the remainder in the devision will assert periodic bounadry conditions
    also makes search_neighbours useful for L1=1 aswell
    args:
        indx (tuple of ints): index in matrix
    """
    L = 5
    if tuple:
        return ((indx[0] + L) % L, (indx[1] + L) % L)
    else:
        return (indx + L) % L

def search_neighbours(M, neighbours, visited, site):
    new_neighbour = []
    x = np.array((1, 0))
    y = np.array((0, 1))
#site[0], boundary(site[1] + 1)

    if (M[boundary(site + y, tuple =True)] != M[site[0], site[1]]) and boundary(site + y, tuple =True)\
                                                      not in neighbours: # neightbour aligned and not already in list

        if (boundary(site + y, tuple =True)) not in visited:
            neighbours.append((site[0], boundary(site[1] + 1)))
            new_neighbour.append((site[0], boundary(site[1] + 1)))
            visited.append((site[0], boundary(site[1] + 1)))

    if (M[boundary(site - y, tuple =True)] != M[site[0], site[1]]) and (boundary(site - y, tuple =True)
                                                     ) not in neighbours:  # neightbour aligned and not already in list
        if boundary(site - y, tuple =True) not in visited:
            neighbours.append((site[0], boundary(site[1] - 1)))
            new_neighbour.append((site[0], boundary(site[1] - 1)))
            visited.append((site[0], boundary(site[1] - 1)))


    if (M[boundary(site + x, tuple =True)] != M[site[0], site[1]]) and (boundary(site + x, tuple =True)
                                                     ) not in neighbours:  # neightbour aligned and not already in list
        if boundary(site + x, tuple =True) not in visited:
            neighbours.append((boundary(site[0]+1), site[1]))
            new_neighbour.append((boundary(site[0]+1), site[1]))
            visited.append((boundary(site[0]+1), site[1]))


    if (M[boundary(site - x, tuple =True)] != M[site[0], site[1]]) and (boundary(site - x, tuple =True)
                                                     ) not in neighbours:  # neightbour aligned and not already in list
        if boundary(site - x, tuple =True) not in visited:
            neighbours.append((boundary(site[0]-1), site[1]))
            new_neighbour.append((boundary(site[0]-1), site[1]))
            visited.append((boundary(site[0]-1), site[1]))

    return neighbours, new_neighbour

dimension = 2
neighbours = []
visited = []
init_idx = np.random.randint(0, L, size=(dimension))
state = M[boundary(init_idx,tuple=True)]
M[boundary(init_idx, tuple=True)]*=-1
visited.append(boundary(init_idx, tuple=True))
neighbours, new_neighbour = search_neighbours(M, neighbours, visited, init_idx)

M[boundary(init_idx,tuple=True)] = 99

for i in neighbours:
    M[boundary(i, tuple=True)] = 99
print(M)
M[np.where(M == 99)] = state

print(neighbours, new_neighbour)
site = new_neighbour[0]
print(site)
print('cunt')
visited.append(site)
neighbours, new_neighbour = search_neighbours(M, neighbours, visited, site)
neighbours.remove(site)
print(neighbours, new_neighbour)
print(visited)
