"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
np.set_printoptions(threshold=np.nan)

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.s = 0 	#state
        self.a = 0	#last action

        self.Q=np.zeros((num_states,num_actions))
        #print self.Q
        #print self.Q.shape

	#dont update q table in querysetstate
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if np.count_nonzero(self.Q[s])==0:
            action = rand.randint(0, self.num_actions-1)
        else:
            action =np.argmax(self.Q[s])
        self.a = action
        if self.verbose: print "s =", s,"a =",action
        #print "Self.s is",self.s
        #print
        """
        print
        print "Func 2"
        print self.s
        print action
        """
        return action

    def author(self):
    	return 'yjallan3'

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        #update the Q table
        #(self.s,self.a,s_prime,r)
        #set your state to s_prime
        #pick your action using the q table...sometimes
        #sometimes means you take a random action accoding to rar
        self.Q[self.s,self.a]=(1-self.alpha)*(self.Q[self.s,self.a])+self.alpha*(r+self.gamma*np.max(self.Q[s_prime]))
        #if (r==100):
        #print "s is", self.s
        #print "s_prime is", s_prime
        #print "Q update is ", (1-self.alpha)*(self.Q[self.s,self.a])+self.alpha*(r+self.gamma*np.max(self.Q[s_prime]))
        #print(self.Q)
        """
        print
        print "Func 3"
        print "old state",self.s
        print "new state",s_prime
        print "action",self.a
        print "reward",r
        print self.Q
        """

        #print
        self.s = s_prime
        var1=rand.uniform(0, 1)
        if var1>self.rar:
        	action =np.argmax(self.Q[s_prime])
        else:
            action = rand.randint(0, self.num_actions-1)
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        self.a = action
        self.rar=self.rar*self.radr
        #print self.rar
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
