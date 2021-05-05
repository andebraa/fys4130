import numpy as np
import matplotlib.pyplot as plt
np.random.seed(6942069)


class wolff_class():
    """
    class for generating a 2D or 1D matrix of spins +/- 1 and applying the wolff
    algorithm as a function of temperature
    """

    def __init__(self, L, L1=1, mc_cycles=1, J=1, Kb=1,
                 dimension=1):
        self.L = L
        self.L1 = L1
        self.mc_cycles = mc_cycles
        self.J = J
        self.Kb = Kb
        self.dimension = dimension

    def wolff(self, T):
        """
        Applies the wolf algorithm on a generated (L,L1) matrix
        """

        L = self.L
        L1 = self.L1
        J = self.J
        Kb = self.Kb
        if L1 > 1:
            self.dimension = 2
            print('dimension = 2')
        neighbours = []  # all neighbours are opposite to site once site is flipped
        visited = []

        #p = 1 - np.exp((-2 * J) / (Kb * T))  # TODO; hva skal være her?
        p = 1- np.exp(-2*(1/T))
        if self.dimension == 2:
            M = np.random.randint(0, 2, size=(L, L1), dtype='int')
        else:
            M = np.random.randint(0, 2, size=(L), dtype='int')#.reshape(-1,1)
        M[np.where(M == 0)] = -1
        m_init = M
        print(M)
        M_ = M


        sigma_0 = 0
        sigma_r = np.zeros(M.shape[0])#.reshape(-1,1)
        sigma_rr = np.zeros(M.shape[0])#.reshape(-1,1)
        mag = 0
        boundary = self.boundary
        search_neighbours = self.search_neighbours
        mc_cycles = self.mc_cycles

        for i in range(mc_cycles):
            init_idx = np.random.randint(0, L, size=(self.dimension))
            if self.dimension == 1:
                sigma_r += M
                sigma_0 += M[0]
                sigma_rr += M*M[0]
            mag += np.sum(M)
            #mag2 += np.abs(np.sum(M**2))

            M[boundary(init_idx)] *= -1  # gotta flip atleast one
            visited.append(init_idx)
            site = init_idx
            neighbours, new_neighbour = search_neighbours(M, neighbours, visited, site)
            while neighbours != []:
                for i,elem in enumerate(new_neighbour):
                    if np.random.uniform(0, 1) > p:  # with probability p
                        site = elem
                        #M[boundary((site[0], site[1]),tuple=True)] *= -1  # flip flip flipadelphia
                        M[boundary(site,tuple=True)] *= -1
                        neighbours.remove(site)
                        neighbours, new_neighbour = search_neighbours(
                                                M, neighbours, visited, site)
                        M_[site] = 99
                        print(M_)
                        break
                    elif new_neighbour == []: #No neighbours that can be flipped
                        break  # exits one loop,
                    else:  # probability not reaced. Here we don't want to look for neighbours as they'll be islands
                        neighbours.remove(elem)
                        del new_neighbour[i]
                if new_neighbour == []: #If no neighbours of site can be flipped, fetch previous neighbours that got skipped
                    # if site is a dead end, first element
                    try:
                        new_neighbour.append(neighbours[0])
                    except:
                        break
                    # of neightbours is set as next site
        sigma_r_avg = sigma_r/mc_cycles
        sigma_0_avg = sigma_0/mc_cycles
        sigma_rr_avg = sigma_rr/mc_cycles
        # - (sigma_0/mc_cycles)*(sigma_r/mc_cycles)
        c = sigma_rr_avg-sigma_0_avg*sigma_r_avg
        r_ = np.arange(M.shape[0])
        if self.dimension == 1:
            plt.plot(r_, self.anal_c(r_, T), label='analytic')
            plt.plot(r_,c, label='numerical')
            plt.legend()

        if self.dimension == 2:
            m = mag/(mc_cycles*L**2)
            plt.plot(T,m, '*')
        return M

    def boundary(self, indx, tuple = False):
        """
        the remainder in the devision will assert periodic bounadry conditions
        also makes search_neighbours useful for L1=1 aswell
        args:
            indx (tuple of ints): index in matrix
        """
        L = self.L
        if tuple:
            return ((indx[0] + L) % L, (indx[1] + L) % L)
        else:
            return (indx + L) % L

    def search_neighbours(self, M, neighbours, visited, site):
        new_neighbour = []
        boundary = self.boundary
        if self.dimension == 2:
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

            if (M[site[0], boundary(site[1] - 1)] != M[site[0], site[1]]) and (boundary(site - y, tuple =True)
                                                             ) not in neighbours:  # neightbour aligned and not already in list
                neighbours.append((site[0], boundary(site[1]  - 1)))
                new_neighbour.append((site[0], boundary(site[1]  - 1)))
                visited.append((site[0], boundary(site[1]  + 1)))


            if (M[boundary(site[0]+1), site[1]] != M[site[0], site[1]]) and (boundary(site + x, tuple =True)
                                                             ) not in neighbours:  # neightbour aligned and not already in list
                neighbours.append((boundary(site[0]+1), site[1]))
                new_neighbour.append((boundary(site[0]+1), site[1]))
                visited.append((site[0], boundary(site[1]  + 1)))


            if (M[boundary(site[0]-1), site[1]] != M[site[0], site[1]]) and (boundary(site - x, tuple =True)
                                                             ) not in neighbours:  # neightbour aligned and not already in list
                neighbours.append((boundary(site[0]-1), site[1]))
                new_neighbour.append((boundary(site[0]-1), site[1]))
                visited.append((site[0], boundary(site[1]  + 1)))


        else:
            x = 1
            if M[boundary(site + x)] != M[site] and boundary(site + x) not in neighbours:  # neightbour aligned and not already in list
                neighbours.append(boundary(site + x))
                new_neighbour.append(boundary(site + x))


            if np.all(M[boundary(site - x)] != M[site]) and (boundary(site - x)
                                                             ) not in neighbours:  # neightbour aligned and not already in list
                neighbours.append(boundary(site - x))
                new_neighbour.append(boundary(site - x))


        return neighbours, new_neighbour

    def anal_c(self, r,T):
        L = self.L
        J = self.J
        beta = 1/T
        c1 = (np.tanh(beta*J)**r)/(1+np.tanh(beta*J)**L)
        c2 = ((1/np.tanh(beta*J))**r)/(1+(1/np.tanh(beta*J))**L)
        return c1 + c2

    def plot_c(self, M, c):
        plt.plot(r_, self.anal_c(r, T), label='analytic')
        plt.plot(r,c, label='numerical')
        plt.legend()

#
# L = 16
# inst = wolff_class(L)
# T = 0.7
# M = inst.wolff(T)
# print(M)
# plt.show()

# Ts = np.linspace(1,5,10)
# for T in Ts:
#     L = 16
#     inst = wolff_class(L, L, dimension = 2)
#     M = inst.wolff(T)
# plt.show()
#
T = 1
L = 10
inst = wolff_class(L, L, dimension = 2)
M = inst.wolff(T)
plt.show()
