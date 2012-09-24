import os, sys
import numpy as np
import scipy.sparse
import operator
import itertools
from msmbuilder.MSMLib import estimate_transition_matrix
from msmbuilder import msm_analysis
from collections import defaultdict


class Sampler(object):
    "Interface for samplers"
    def __init__(self, walker, n_steps):
        self.walker = walker
        self.n_steps = n_steps

        # number of transition counts, as a defaultdict
        # the keys are going to be pairs of points (k-tuples)
        self.tcounts_dd = defaultdict(lambda : 0)
        self.all_points = set()

        # number of times a transition ends in each state
        # self.all_points also includes the initial point, even
        # if you never jump TO it, whereas n_visits_to only
        # includes points that a transition ended in.
        self.n_visits_to = defaultdict(lambda : 0)
        self.n_visits_from = defaultdict(lambda : 0)
        self.n_visits_tofrom = defaultdict(lambda : 0)
        

    def run(self):
        # make sure to record the initial point as visited
        # walker.set_point is not allowed to go to a point
        # that hasn't been visited yet, so in general we
        # only have to add new_pt
        self.all_points.add(self.walker.get_point())

        for i in xrange(self.n_steps):
            old_pt = self.walker.get_point()
            new_pt = self.walker.next()
            self.tcounts_dd[(old_pt, new_pt)] += 1
            #print 'old', old_pt, 'new', new_pt
            
            self.n_visits_from[old_pt] += 1
            self.n_visits_to[new_pt] += 1
            self.n_visits_tofrom[old_pt] += 1
            self.n_visits_tofrom[new_pt] += 1
            
            self.all_points.add(new_pt)
            
            # run the adaptive sampling algorithm
            self.step()

    def step(self):
        pass

    def tprob(self):
        # associate each point with a unique index starting at 0
        n_pts = len(self.all_points)
        mapping = {}
        for i, pt in enumerate(self.all_points):
            mapping[pt] = i

        # convert the indices of the counts matrix to ints
        tcounts = scipy.sparse.dok_matrix((n_pts,n_pts), dtype=np.double)
        for key, val in self.tcounts_dd.iteritems():
            p1, p2 = key
            i, j = mapping[p1], mapping[p2]
            tcounts[i,j] = val
        
        tcounts = tcounts + tcounts.T

        # call msmbuilder libraries
        # this one is just the regular row normalization estimator, and is not
        # guarenteed to be reversible
        return mapping, estimate_transition_matrix(tcounts)


class MinCounts(Sampler):
    def step(self):
        # sort by value, then get the first element -- this is the state with
        # the fewest counts
        # http://writeonly.wordpress.com/2008/08/30/sorting-dictionaries-by-value-in-python-improved/
        pt, counts = sorted(self.n_visits_tofrom.iteritems(), key=operator.itemgetter(1))[0]
        
        #print 'pt', pt, 'counts', counts
        self.walker.set_point(pt)


class Meta1(Sampler):
    def __init__(self, walker, n_steps, w, sigma):
        super(Meta1, self).__init__(walker, n_steps)
        self.w = w
        self.sigma = sigma

    def step(self):
        mapping, tprob = self.tprob()
        
        # if we can guarentee that the counts matrix is reversible, we can
        # do this faster without the eigensolver, but I don't want to do that yet.
        vectors = msm_analysis.get_eigenvectors(tprob, min(5, tprob.shape[0]))[1]
        populations = vectors[:, 0]
        
        # this chooses the point from the multinomial, but its now indexed
        # in the mapped (integer) space
        chosen = np.where(np.random.multinomial(1, populations) == 1)[0][0]
        
        # back out the chosen item as a tuple, such that mapping[k] == chosen
        k = next(k for k,v in mapping.iteritems() if v == chosen)
        
        self.walker.set_point(k)

    











    
