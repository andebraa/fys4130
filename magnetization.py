import numpy as np
import matplotlib.pyplot as plt

L = 16
L1=16
np.random.seed(42069420)
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
    L = 16
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

#-------------------------------------------------------------------------------
T_arr = np.linspace(1E-1,10,50)
mag = np.zeros(len(T_arr))
mag2 = np.zeros(len(T_arr))
mc_cycles = 1000

for t,T in enumerate(T_arr):
    site = np.random.randint(0, L, size=(2))
    state = M[boundary(site,tuple=True)]
    M[boundary(site, tuple=True)]*=-1
    M_init = np.array(M)

    for cycle in range(mc_cycles):

        p = 1 - np.exp(-2*(1/T))
        dimension = 2
        neighbours = []
        visited = []


        visited.append(boundary(site, tuple=True))
        neighbours, new_neighbour = search_neighbours(M, neighbours, visited, site)

        while neighbours != []:
            for ind,i in enumerate(new_neighbour):

                new_neighbour.remove(i)
                if np.random.uniform(0, 1) < p:
                    M[boundary(site, tuple=True)]*=-1
                    site = i
                    visited.append(boundary(site, tuple=True))
                    neighbours.remove(site)
                    neighbours, new_neighbour = search_neighbours(M, neighbours, visited, site)

                    break
                else:  # probability not reaced. Here we don't want to look for neighbours as they'll be islands
                    neighbours.remove(i)
            if new_neighbour == []:
                try:
                    new_neighbour.append(neighbours[0])
                except: #neighbours is empty, and we are done
                    break
        mag[t] += np.sum(M)
        # for e in range(len(M[0])):
        #     for f in range(len(M[0])):
        #         mag2[t] += mag[t]*M[e,f]
        for e in range(len(M[0])):
            for r in range(len(M[0])):
                mag2[t] += np.sum(M[e]*M[r])


        site = np.random.randint(0, L, size=(2))


        #print(M_init)
    m = mag/(mc_cycles*L**2)
    m2 = mag2/(mc_cycles*L**4)
#
plt.semilogx(T_arr,m)
plt.xlabel('T/J')
plt.ylabel('<m>')
plt.show()
print(M)
plt.semilogx(T_arr, m2)
plt.xlabel(r'T/J')
plt.ylabel(r'$<m^2>$')
plt.show()
