import numpy as np
import matplotlib.pyplot as plt
np.random.seed(69420)


class wolff_class():
    """
    class for generating a 2D or 1D matrix of spins +/- 1 and applying the wolff
    algorithm as a function of temperature
    """
    def __init__(self,L, L1 = 1 ,mc_cycles = 10000, T = 0.5, J = 1, Kb = 1,
                 dimension = 1 ):
        self.L = L
        self.L1 = L1
        self.mc_cycles = mc_cycles
        self.J = J
        self.Kb = Kb
        self.T = T
        self.dimension = dimension

    def wolff(self):
        """
        Applies the wolf algorithm on a generated (L,L1) matrix
        """
        L = self.L
        L1 = self.L1
        J = self.J
        Kb = self.Kb
        T = self.T
        sites = [] #indexes
        neighbours = [] #all neighbours are opposite to site once site is flipped


        p = 1 - np.exp((-2*J)/(Kb*T)) #TODO; hva skal være her?
        M = np.random.randint(0,2, size=(L,L1), dtype='int')
        M[np.where(M==0)] = -1
        print(M)
        sigma_0 = 0
        sigma_r = np.zeros(M.shape[0])
        boundary = self.boundary
        search_neighbours = self.search_neighbours
        mc_cycles = self.mc_cycles

        for i in range(mc_cycles):
            init_idx = np.random.randint(0,L, size=(self.dimension))
            if self.dimension == 1:
                for r in range(M.shape[0]):
                    sigma_0 += M[0]
                    sigma_r[r] += M[r]

            M[boundary(init_idx)] *= -1 #gotta flip atleast one
            site = init_idx
            sites.append(site)
            neighbours, new_neighbour = search_neighbours(M, neighbours, site)
            while neighbours != []:
                print('cunt')
                for j in new_neighbour:

                    if np.random.uniform(0,1) > p: #with probability p
                        site = new_neighbour[j]
                        M[boundary(site)] *= -1 #flip flip flipadelphia
                        sites.append(site)
                        #neighbours.remove(site)
                        neighbours, new_neighbour = search_neighbours(M, neighbours, site)
                        break
                    elif new_neighbour == []:
                        break #exits one loop
                    else: #probability not reaced. Here we don't want to look for neighbours as they'll be islands
                        neighbours.remove(new_neighbour[j])
                        del new_neighbour[j]
                if new_neighbour == []:
                    new_neighbour.append(neighbours[0]) #if site is a dead end, first element
                                                        #of neightbours is set as next site


        c = (sigma_0*sigma_r)/mc_cycles #- (sigma_0/mc_cycles)*(sigma_r/mc_cycles)
        plt.plot(c)
        plt.show()
        return M


    def boundary(self,indx):
        """
        the remainder in the devision will assert periodic bounadry conditions
        also makes search_neighbours useful for L1=1 aswell
        args:
            indx (tuple of ints): index in matrix
        """
        L = self.L
        if self.dimension ==2:
            return ((indx[0] + L) % L, (indx[1] +L) %L)
        else:
            return (indx + L) %L


    def search_neighbours(self, M, neighbours, site):

        new_neighbour = []
        boundary = self.boundary
        x = np.array((1,0))
        y = np.array((0,1))

        if np.all(M[boundary(site+y)] != M[site]) and \
        (boundary(site+y)) not in neighbours: #neightbour aligned and not already in list
            neighbours.append(boundary(site+y))
            new_neighbour.append(boundary(site+y))

        if np.all(M[boundary(site-y)] != M[site]) and \
        (boundary(site-y)) not in neighbours: #neightbour aligned and not already in list
            neighbours.append(boundary(site-y))
            new_neighbour.append(boundary(site-y))

        if np.all(M[boundary(site+x)] != M[site]) and \
        (boundary(site+x)) not in neighbours: #neightbour aligned and not already in list
            neighbours.append(boundary(site+x))
            new_neighbour.append(boundary(site+x))

        if np.all(M[boundary(site-x)] != M[site]) and \
        (boundary(site-x)) not in neighbours: #neightbour aligned and not already in list
            neighbours.append(boundary(site-x))
            new_neighbour.append(boundary(site-x))

        return neighbours, new_neighbour

    # def C(M,r):
    #     "NOTE only functional for 1D M"
    #     if M.size[1] != None:
    #         raise ValueError "Only viable for 1D matrices"
    #         exit()
    #     c = np.zeros(len(M))
    #     for i in len(M):
    #         c[i] = M[0]*M[i] - M[]



L = 10
inst =wolff_class(L)
M = inst.wolff()
print(M)
