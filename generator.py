"""
Generator for walk on the kd lattice
"""
import os, sys
import itertools
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as pp
import IPython as ip

class DWRW(object):
    def __init__(self, n_dims, minima_pos, tilt, roughness, random_source=None):
        """Initialize the iterator for a random walk on a
        random energy lattice with two native states
        
        Parameters
        ----------
        n_dims : int
            The dimensionality of the lattice
        minima_pos : int 
            The two minima will be at positions (+minima_pos, 0, 0, ..., 0) and
             (-minima_pos, 0, 0, ..., 0)
        tilt : float
            Controls the bias towards the native states in the potential
            energy surface.
        roughness : float
            Controls the roughness of the potential energy surface
        random_rouce : np.random.RandomState or None
            Where to pick the random numbers from. The default is
            from numpy.random
        
        Notes
        -----
        
        At latice point `\vec{v} = (v_1, v_2, v_2, ... v_k) \in \mathbb{Z}`,
        the potential energy is given as `E(n) \sim \mathcal{N}(\mu = tilt \cdot 
        min(d(v, n1), d(v, n2)), \sigma=roughness)`
        
        Where `d` is the manhattan distance `d(v1, v2) = \sum_i^k |v1_i - v2_i|`
        
        Transitions are done with a metropolis criterion, and the proposal distribution
        is uniform over the neighbors.
        """
        self.n_dims = n_dims
        self.minima_pos = minima_pos
        self.tilt = tilt
        self.roughness = roughness
        self.random = np.random if random_source is None else random_source
        
        # current point -- initialize to (0, 0, ... ,0)
        self.pt = np.zeros(self.n_dims, dtype=np.int)
        
        # record the energy of every proposed or visited point
        self._energies = {} 
        self.current_energy = self.energy(self.pt)
    
    def set_point(self, point):
        point = np.array(point, dtype=int)
        if not tuple(point) in self._energies:
            raise ValueError("I haven't been here before, so you can't"
                             "set my position to %s" % point)
        self.pt = point

    def get_point(self):
        return tuple(self.pt)
    
    def energy(self, point):
        """Energy of a point
        
        Parameters
        ----------
        point : array_like, shape=[n_dims, dtype=int]
            Vector in `\mathcal{Z}^k` to evaluate the energy at
        
        Returns
        -------
        energy : float
            The local energy
        """
        # having the point as a tuple makes it immutable, so we can use
        # it as a dictionary key without fear
        point = tuple(point)
        if point not in self._energies:
            # the min manhattan distance of point to (+p,0,0,0,0,0 ), (-p, 0,0,0,0)
            # is just the sum of its coordinates in the 1 to k places plus the min of
            # its difference from p and -p
            d = np.sum(np.abs(point[1:])) + min(abs(point[0] - self.minima_pos),
                                                abs(point[0] + self.minima_pos))
            self._energies[point] = np.random.normal(loc=self.tilt*d,
                                                    scale=self.roughness)
        return self._energies[point]
        
    def next(self):
        """
        Get the next point, which will be adjacent to the current point (or 
        will stay the current point)
        
        Returns
        -------
        point : tuple
            The points are always returned as tuples, regardless of their
            internal representation (which is sometimes tuple and sometimes
            array)
        """
        # r1 is which dimension to flip, and r2 is whether to go +1 or -1
        # in that dimension
        r1, r2 = self.random.randint(self.n_dims), self.random.randint(2)
        orig_r1 = self.pt[r1]
        if r2 == 0:
            self.pt[r1] += 1
        else:
            self.pt[r1] -= 1
        
        proposal_energy = self.energy(self.pt)
        
        # metropolis criterion
        if proposal_energy < self.current_energy:
            self.current_energy = proposal_energy 
            return tuple(self.pt)
        
        # compute the acceptance ratio
        accept_ratio = np.exp(-(proposal_energy - self.current_energy))
        if self.random.uniform() < accept_ratio:
            self.current_energy = proposal_energy
            return tuple(self.pt)

        self.pt[r1] = orig_r1
        return tuple(self.pt)
    
    
    def __iter__(self):
        return self
        
def get_1d_energies(generator):
    position = sorted(np.array(generator._energies.keys())[:,0])
    energy = np.zeros(len(position))
    for i,x in enumerate(position):
        energy[i] = generator._energies[(x,)]

    return position, energy

def test():
    n_dims = 1
    n_steps = 100000
    
    g = DWRW(n_dims=n_dims, minima_pos=20, tilt=0.1, roughness=0.5)
    seq = np.zeros((n_steps, n_dims), dtype=int)
    freq = defaultdict(lambda: 0)

    for i, pt in enumerate(itertools.islice(g, n_steps)):
        seq[i] = pt
        freq[pt] += 1

    pp.subplot(211)
    pp.title('trajectory')
    pp.plot(seq[:, 0], 'b-')
    pp.subplot(212)
    pp.title('Energy')
    position, energy = get_1d_energies(g)
    origin = np.where(np.array(position)==0)[0][0]
    energy -= energy[origin]
    pp.plot(position, energy, 'r-', label='true_energy')
    fe = -np.log([freq[(p,)] for p in position])
    fe -= fe[origin]
    pp.plot(position, fe, label='-log(counts)')
    pp.legend()
    

    
    pp.savefig('fig.png')
    os.system('open fig.png')
        
if __name__ == '__main__':
    test()
        
