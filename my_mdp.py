"""Markov Decision Processes (Chapter 17)

First we define an MDP, and the special case of a GridMDP, in which
states are laid out in a 2-dimensional grid.  We also represent a policy
as a dictionary of {state:action} pairs, and a Utility function as a
dictionary of {state:number} pairs.  We then define the value_iteration
and policy_iteration algorithms."""

# from utils import *

from sys import maxint
from itertools import product, izip, count
from toolz import first
from collections import defaultdict
import numpy as np
import random
from scipy.sparse import csr_matrix, dok_matrix
from scipy.spatial.distance import cityblock as l1distance
from heapq import heappush, heappop
ftype = np.float32


class VVMdp:

    def __init__(self,
                 _startingstate,
                 _transitions,  # dictionary of key:values   (s,a,s):proba
                 _rewards,  # dictionary of key:values   s: vector of rewards
                 _gamma=.9):

        try:
            states = sorted(
                {st for (s, a, s2) in _transitions.iterkeys() for st in (s, s2)}
            )
            actions = sorted(
                {a for (s, a, s2) in _transitions.iterkeys()}
            )

            n , na = len(states) , len(actions)

            stateInd = {s: i for i, s in enumerate(states)}
            actionInd = {a: i for i, a in enumerate(actions)}

            self.startingStateInd = stateInd[_startingstate]

            d = len(_rewards[first(_rewards.iterkeys())])
            assert all(d == len(np.array(v,dtype=ftype)) for v in _rewards.itervalues()),\
                   "incorrect reward vectors"
            assert set(_rewards.keys()).issubset(states) ,\
                   "states appearing in rewards should also appear in transitions"

        except ValueError,TypeError:

            print "transitions or rewards do not have the correct structure"
            raise


        # convert rewards to nstates x d matrix
        rewards = np.zeros((n,d), dtype=ftype)
        for s, rv in _rewards.iteritems():
            rewards[stateInd[s], :] = rv

        self.rmax = np.max( [sum(abs(rewards[s,:])) for s in range(n)] )

        # Convert Transitions to nstates x nactions array of sparse 1 x nstates matrices
        # sparse matrices are build as DOK matrices and then converted to CSR format
        # build reverse_transition dictionary of state:set of states
        transitions = np.array(
            [[ dok_matrix((1, n), dtype=ftype) for _ in actions ] for _ in states ],
            dtype=object
        )

        rev_transitions = defaultdict(set)

        for (s, a, s2), p in _transitions.iteritems():
            si, ai, si2 = stateInd[s], actionInd[a], stateInd[s2]
            transitions[si,ai][0, si2] = p
            rev_transitions[si2].add(si)

        for s, a in product(range(n), range(na)):
            transitions[s,a] = transitions[s,a].tocsr()
            assert 0.99 <= transitions[s,a].sum() <= 1.01, "probability transitions should sum up to 1"

        # autoprobability[s,a] = P(s|s,a)
        self.auto_probability = np.array( [[transitions[s,a][0,s] for a in range(na)] for s in range(n)] ,dtype=ftype )

        # copy local variables in object variables
        self.states , self.actions , self.nstates , self.nactions, self.d = states,actions,n,na,d
        self.stateInd,self.actionInd = stateInd,actionInd
        self.rewards , self.transitions, self.rev_transitions = rewards , transitions, rev_transitions
        self.gamma = _gamma


    def R(self, state):
        "Return a vector reward for this state."
        return self.rewards[state]

    def T(self, state, action):
        """Transition model.  From a state and an action, return all
        of (state , probability) pairs."""
        _tr = self.transitions[state,action]
        return izip(_tr.indices, _tr.data)

    def set_Lambda(self,l):
        self.Lambda = np.array(l,dtype=ftype)

    def expected_vec_utility(self,s,a, U):
        "The expected vector utility of doing a in state s, according to the MDP and U."
        return np.sum( (p*U[s2] for s2,p in self.T(s,a)) )
        #som=0.0
        #for s2,p in self.T(s,a):
        #    som += p*U[s2]
        #    #print "->",p,"/\/",s2, ", U(.)=",U[s2]
        #return som

    def expected_utility(self,s,a,U):
        # assumes self.Lambda numpy array exists
        return sum( (p*(U[s2].dot(self.Lambda)) for s2,p in self.T(s,a)) )


    def value_iteration(self, epsilon=0.001):
        "Solving an MDP by value iteration. [Fig. 17.4]"
        U1 = np.zeros( (self.nstates,self.d) , dtype=ftype)
        gamma , R , expected_utility = self.gamma , self.R , self.expected_utility

        while True:
            U = U1.copy()
            delta = 0.0
            for s in range(self.nstates):
                Q = [expected_utility(s, a, U) for a in range(self.nactions)]
                U1[s] = R(s) + gamma * self.expected_vec_utility(s,np.argmax(Q),U)
                delta = max(delta, (U1[s] - U[s]).dot(self.Lambda) )
            if delta < epsilon * (1 - gamma) / gamma:
                return U

    def best_action(self,s,U):
        return np.argmax( [self.expected_utility(s,a, U) for a in range(self.nactions)] )

    def best_policy(self, U):
        """Given an MDP and a utility function U, determine the best policy,
        as a mapping from state to action. (Equation 17.4)"""
        pi = np.zeros((self.nstates),np.int)
        for s in range(self.nstates):
            pi[s] = self.best_action(s,U)
        return pi

    def readable_policy(self,pi):
        return {self.states[s]:self.actions[a] for s,a in pi.iteritems()}

    def policy_iteration(self):
        "Solve an MDP by policy iteration [Fig. 17.7]"
        U = np.zeros( (self.nstates,self.d) , dtype=ftype)
        pi = {s:random.randint(0,self.nactions-1) for s in range(self.nstates)}
        while True:
            U = self.policy_evaluation(pi, U,k=20)
            unchanged = True
            for s in range(self.nstates):
                a = self.best_action(s,U)
                if a != pi[s]:
                    pi[s] = a
                    unchanged = False
            if unchanged:
                return pi,U


    def policy_evaluation(self,pi, U1, k=maxint , epsilon=0.001):
        """Return an updated utility mapping U from each state in the MDP to its
        utility, using an approximation (modified policy iteration)."""
        R, gamma ,expect_vec_u = self.R, self.gamma , self.expected_vec_utility

        for i in count(0):
            U = U1.copy()
            delta = 0.0
            for s in range(self.nstates):
                U1[s] = R(s) + gamma * expect_vec_u(s,pi[s],U)

                delta = max( delta, l1distance(U1[s],U[s]) )

            if i > k or delta < epsilon * (1 - gamma) / gamma:
                return U

    def prioritized_sweeping_policy_evaluation(self,pi, U1, k=maxint , epsilon=0.001):
        """Return an updated utility mapping U from each state in the MDP to its
        utility, using an approximation (modified policy iteration)."""
        R, gamma ,expect_vec_u = self.R, self.gamma , self.expected_vec_utility
        h = []

        for s in range(self.nstates):
            heappush( h , (-self.rmax-random.uniform(0,1),s) )

        for i in count(0):
            U = U1.copy()

            (delta,s) = heappop(heap)

            U1[s] = R(s) + gamma * expect_vec_u(s,pi[s],U)

            delta = l1distance(U1[s],U[s])

            if i > k or delta < epsilon * (1 - gamma) / gamma:
                return U



def test_VVMDP():
    monMdp = VVMdp(
        _startingstate='buro',
        _transitions={
            ('buro', 'bouger', 'couloir'): 0.4,
            ('buro', 'bouger', 'buro'): 0.6,
            ('buro', 'rester', 'buro'): 1,
            ('couloir', 'bouger', 'couloir'): 1,
            ('couloir', 'bouger', 'buro'): 0,
            ('couloir', 'rester', 'couloir'): 1,
            ('couloir', 'rester', 'buro'): 0
        },
        _rewards={
            'buro': [0.0, 0.0],
            'couloir': [1.0, 0.0]
        }
    )

    import pprint

    print "--state indices and state set--"
    pprint.pprint(monMdp.stateInd)
    pprint.pprint(monMdp.states)
    print "--action indices and action set--"
    pprint.pprint(monMdp.actionInd)
    pprint.pprint(monMdp.actions)
    print "--nstates,nactions,d--"
    print monMdp.nstates, monMdp.nactions, monMdp.d
    print "--rewards--"
    pprint.pprint(monMdp.rewards)
    print "--transitions--"
    pprint.pprint([[M.toarray() for M in L] for L in monMdp.transitions])
    print "--reverse transitions--"
    pprint.pprint(monMdp.rev_transitions)
    print "-- testing T --"
    pprint.pprint(list(monMdp.T(0, 0)))
    #U = np.zeros((2,2),dtype=ftype)
    #U[1] = [3,4]
    #return monMdp, U

    ########################################

    n = 10
    _t =       { ((i,j),'v',(min(i+1,n-1),j)):0.9 for i,j in product(range(n),range(n)) }
    _t.update( { ((i,j),'v',(max(i-1,0),j)):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'^',(max(i-1,0),j)):0.9 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'^',(min(i+1,n-1),j)):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'>',(i,min(j+1,n-1))):0.9 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'>',(i,max(j-1,0))):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'<',(i,max(j-1,0))):0.9 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'<',(i,min(j+1,n-1))):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'X',(i,j)):1 for i,j in product(range(n),range(n)) } )


    _r = { (i,j):[0.0,0.0] for i,j in product(range(n),range(n))}
    _r[(n-1,0)] = [1.0,0.0]
    _r[(0,n-1)] = [0.0,1.0]
    _r[(n-1,n-1)] = [1.0,1.0]

    def show_grid_policy(vvmdp,pi):
        print np.matrix([[vvmdp.actions[pi[vvmdp.stateInd[(i,j)]]] for j in range(n)] for i in range(n)])

    print "--creating grid---"
    gridMdp = VVMdp (
        _startingstate=(0,0),
        _transitions= _t,
        _rewards=_r
    )
    print "---value iteration---"

    gridMdp.set_Lambda( [1,0] )
    pi,U = gridMdp.policy_iteration()
    print "--- value iteration ended ---"
    pi = gridMdp.best_policy(U)
    show_grid_policy(gridMdp,pi)

if __name__ == '__main__':
    test_VVMDP()
    exit()



