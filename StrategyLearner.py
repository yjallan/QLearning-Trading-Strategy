"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import random
import QLearner as ql
import random as rand

np.random.seed(1481090000)
random.seed(1481090000)

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.learner = ql.QLearner(num_states=1000,\
        	num_actions = 3, \
        	alpha = 0.5, \
            gamma = 0.9, \
            rar = 0.5, \
            radr = 0.999, \
            dyna = 0, \
            verbose=False) #initialize the learner

    def author(self):
        return 'yjallan3'

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 

        sd_NEW=sd+dt.timedelta(-30)
        #print "HERE"
        #print sd
        #print sd_NEW
        #print ed
        #print symbol
        #print sv
        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms=[symbol]
        #print syms
        dates = pd.date_range(sd, ed)
        dates_NEW = pd.date_range(sd_NEW, ed)
        #print dates
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_all_NEW = ut.get_data(syms, dates_NEW)  # automatically adds SPY
        #print prices_all
        prices = prices_all[syms]  # only portfolio symbols
        prices_NEW = prices_all_NEW[syms]  # only portfolio symbols
        #print "Prices dataframe"
        #print prices
        #print
        #print "Prices_NEW dataframe"
        #print prices_NEW
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        prices_SPY_NEW = prices_all_NEW['SPY']  # only SPY, for comparison later
        #print prices_SPY
        
  
        # example use with new colname 
        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        #print volume_all
        volume = volume_all[syms]  # only portfolio symbols
        #print volume
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        #print volume_SPY
        if self.verbose: print volume

        #MY CODE BEGINS HERE

        ###############################################
        ###############INDICATORS######################
        ###############################################
        days1=15
        days2=5

        #STANDARD DEVIATIONS 15 day rolling period
        std =pd.rolling_std(prices_NEW,days1,ddof=0)
        std=std.ix[sd:ed]
        #print
        #print "Rolling Standard deviation dataframe"
        #print std
        
        #DAILY RETURNS
        #print
        #print prices.ix[0]
        #print len(prices)
        #print        
        daily_returns = pd.DataFrame(data=0.0, columns=prices_NEW.columns,index=prices_NEW.index)
        #print daily_returns
        for x in range(1,len(prices_NEW)):    
            daily_returns.ix[x]= (prices_NEW.ix[x]/prices_NEW.ix[x-1]) -1     
        #print
        #print daily_returns
        daily_returns_NEW=daily_returns #saving this to use in volatility
        #print
        #print "Daily Returns dataframe"
        daily_returns=daily_returns.ix[sd:ed]
        #print daily_returns        
        
        #SMA OF 15 DAY
        sma=pd.DataFrame(data=0.0, columns=prices_NEW.columns,index=prices_NEW.index)
        for x in range(1,len(prices_NEW)):
            #print            
            sma.ix[x]= prices_NEW.ix[x-(days1-1):x+1].mean()
        #print
        #print sma
        sma=sma.ix[sd:ed]
        #print
        #print "SMA dataframe"
        #print sma

        #BOLLINGER BAND VALUE AS GIVEN IN COURSE WIKI
        bb=pd.DataFrame(data=0.0, columns=prices.columns,index=prices.index)
        for x in range(0,len(prices)):
            #print            
            bb.ix[x]= (prices.ix[x]-sma.ix[x])/std.ix[x]        
        #print
        #print "Bollinger value dataframe"
        #print bb
        
        #MOMENTUM - 5 day momentum
        momentum=pd.DataFrame(data=0.0, columns=prices_NEW.columns,index=prices_NEW.index)
        for x in range(1,len(prices_NEW)):
            #print
            #print prices_NEW.ix[x]
            #print prices_NEW.ix[x-5]
            momentum.ix[x]= (prices_NEW.ix[x]/prices_NEW.ix[x-days2]) -1 
        #print
        #print momentum
        momentum=momentum.ix[sd:ed]
        #print
        #print "Momentum dataframe"
        #print momentum

        #Volatility (Standard deviation of daily return)
        volatility=pd.rolling_std(daily_returns_NEW,days1,ddof=0)
        volatility=volatility.ix[sd:ed]
        #print
        #print "Full Daily returns"
        #print daily_returns_NEW
        #print
        #print "Rolling Volatility dataframe"
        #print volatility


        ###############################################
        ###############Q-LEARNING######################
        ###############################################

        #BINNING        
        bin_bb=pd.qcut(bb.ix[:,0],10,labels=range(0,10))
        #print
        #print "Bin for Bollinger Bands"
        #print bin_bb        

        bin_momentum=pd.qcut(momentum.ix[:,0],10,labels=range(0,10))        
        #print
        #print "Bin for Momentum"
        #print bin_momentum

        bin_volatility=pd.qcut(volatility.ix[:,0],10,labels=range(0,10))        
        #print
        #print "Bin for volitility"
        #print bin_volatility

        #STATES
        state=pd.DataFrame(data=0, columns=prices.columns,index=prices.index,dtype=int)
        for i in range(len(state)):
            state.ix[i]=(bin_bb.ix[i]*100+bin_momentum.ix[i]*10+bin_volatility.ix[i])
        
        #print
        #print "The state dataframe is"
        #print state
        #print                                   
        
        #TRANSACTION COST
        TC=9.95
        #TC=0
        for epoch in range(15):
        #for epoch in range(1):
            #HOLDINGS
            holdings=pd.DataFrame(0.0,columns=prices.columns,index=prices.index)        
            holdings['Cash']=0.0
            holdings['shareprice']=prices[:]        
            holdings.ix[0,1]=sv

            s=state.ix[0,0]           
            a= self.learner.querysetstate(s)
            #print
            #print
            #print "FOR LOOP BEGINS HERE"
            #print "the value of S is", s
            #print "the value of A is", a
            
            if (a==0): #long 200
                holdings.ix[1,0]=200
                #holdings.ix[1,1]=holdings.ix[0,1]-200*prices.ix[1,0]-TC
                holdings.ix[1,1]=holdings.ix[0,1]-200*prices.ix[0,0]-TC
                #cash next day = cash previous day - 200*share price of previous -TC
            elif (a==1): #short 200
                holdings.ix[1,0]=-200
                holdings.ix[1,1]=holdings.ix[0,1]+200*prices.ix[0,0]-TC
            else: # no position
                holdings.ix[1,0]=0
                holdings.ix[1,1]=holdings.ix[0,1]
            
            #print "The holdings DataFrame is"
            #print holdings
            r=(holdings.ix[1,0]*holdings.ix[1,2]+holdings.ix[1,1])/sv-1
            #r=0
            # reward = ((number of shares* share price) +cash in hand)/(previous day portval) - 1
            
            #print "reward for 0th iteration is", r
            s_prime=state.ix[1,0]
            
            #print "S_prime is", s_prime
            

            for i in range(1,prices.shape[0]-1):
            #for i in range(1,5):                
                a= self.learner.query(s_prime,r)
                #print
                #print "Second for loop iteration #",i
                #print "the value of S is", s_prime
                #print "the value of A is", a
                if (a==0): #long 200
                    holdings.ix[i+1,0]=200
                    if (holdings.ix[i+1,0]==holdings.ix[i,0]):
                        holdings.ix[i+1,1]=holdings.ix[i,1]
                        #print "A"
                    else:
                        #holdings.ix[i+1,1]=holdings.ix[i,1]-abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i+1,0]-TC
                        holdings.ix[i+1,1]=holdings.ix[i,1]-abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i,0]-TC
                        #print "B"
                    #cash next day = cash previous day - 200*share price -TC

                elif (a==1): #short 200
                    holdings.ix[i+1,0]=-200
                    if (holdings.ix[i+1,0]==holdings.ix[i,0]):
                        holdings.ix[i+1,1]=holdings.ix[i,1]
                        #print "C"
                    else:
                        #holdings.ix[i+1,1]=holdings.ix[i,1]+abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i+1,0]-TC                    
                        holdings.ix[i+1,1]=holdings.ix[i,1]+abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i,0]-TC                    
                        #print "D"
                else:
                    holdings.ix[i+1,0]=0
                    if (holdings.ix[i+1,0]==holdings.ix[i,0]):
                        holdings.ix[i+1,1]=holdings.ix[i,1]
                        #print "E"
                    elif (holdings.ix[i+1,0]>holdings.ix[i,0]):
                        #holdings.ix[i+1,1]=holdings.ix[i,1]-abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i+1,0]-TC
                        holdings.ix[i+1,1]=holdings.ix[i,1]-abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i,0]-TC
                        #print "F"
                    else:
                        #holdings.ix[i+1,1]=holdings.ix[i,1]+abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i+1,0]-TC                    
                        holdings.ix[i+1,1]=holdings.ix[i,1]+abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i,0]-TC                    
                        #print "G"
                #print "The holdings DataFrame is"
                #print holdings
                r=(holdings.ix[i+1,0]*holdings.ix[i+1,2]+holdings.ix[i+1,1])/((holdings.ix[i,0]*holdings.ix[i,2]+holdings.ix[i,1]))-1
                #r=(holdings.ix[i,0]*holdings.ix[i,2]+holdings.ix[i,1])/((holdings.ix[i-1,0]*holdings.ix[i-1,2]+holdings.ix[i-1,1]))-1
                #print "The reward is",r
                #print "reward", r
                s_prime=state.ix[i+1,0]
                
                #print "S_prime is", s_prime
            #checking Qtable
            #if (epoch==0):
                #r=100
                #self.learner.query(s_prime,r)                       

        holdings['portval']=0            
        holdings['trades']=0
        holdings.ix[:,3]=holdings.ix[:,0]*holdings.ix[:,2]+holdings.ix[:,1]
        for p in range (0,len(holdings)-1):                
            holdings.ix[p,4]=holdings.ix[p+1,0]-holdings.ix[p,0]
        #print
        #print "The FINAL holdings DataFrame is"
        #print holdings
            

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        #print "TEST POLICY"
        #print

        sd_NEW=sd+dt.timedelta(-30)
        #print "HERE"
        #print sd
        #print sd_NEW
        #print ed
        #print symbol
        #print sv
        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms=[symbol]
        #print syms
        dates = pd.date_range(sd, ed)
        dates_NEW = pd.date_range(sd_NEW, ed)
        #print dates
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_all_NEW = ut.get_data(syms, dates_NEW)  # automatically adds SPY
        #print prices_all
        prices = prices_all[syms]  # only portfolio symbols
        prices_NEW = prices_all_NEW[syms]  # only portfolio symbols
        #print "Prices dataframe"
        #print prices
        #print
        #print "Prices_NEW dataframe"
        #print prices_NEW
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        prices_SPY_NEW = prices_all_NEW['SPY']  # only SPY, for comparison later
        #print prices_SPY
        
  
        # example use with new colname 
        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        #print volume_all
        volume = volume_all[syms]  # only portfolio symbols
        #print volume
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        #print volume_SPY
        if self.verbose: print volume

        #MY CODE BEGINS HERE

        ###############################################
        ###############INDICATORS######################
        ###############################################
        days1=15
        days2=5

        #STANDARD DEVIATIONS 15 day rolling period
        std =pd.rolling_std(prices_NEW,days1,ddof=0)
        std=std.ix[sd:ed]
        #print
        #print "Rolling Standard deviation dataframe"
        #print std
        
        #DAILY RETURNS
        #print
        #print prices.ix[0]
        #print len(prices)
        #print        
        daily_returns = pd.DataFrame(data=0.0, columns=prices_NEW.columns,index=prices_NEW.index)
        #print daily_returns
        for x in range(1,len(prices_NEW)):    
            daily_returns.ix[x]= (prices_NEW.ix[x]/prices_NEW.ix[x-1]) -1     
        #print
        #print daily_returns
        daily_returns_NEW=daily_returns #saving this to use in volatility
        #print
        #print "Daily Returns dataframe"
        daily_returns=daily_returns.ix[sd:ed]
        #print daily_returns        
        
        #SMA OF 15 DAY
        sma=pd.DataFrame(data=0.0, columns=prices_NEW.columns,index=prices_NEW.index)
        for x in range(1,len(prices_NEW)):
            #print            
            sma.ix[x]= prices_NEW.ix[x-(days1-1):x+1].mean()
        #print
        #print sma
        sma=sma.ix[sd:ed]
        #print
        #print "SMA dataframe"
        #print sma

        #BOLLINGER BAND VALUE AS GIVEN IN COURSE WIKI
        bb=pd.DataFrame(data=0.0, columns=prices.columns,index=prices.index)
        for x in range(0,len(prices)):
            #print            
            bb.ix[x]= (prices.ix[x]-sma.ix[x])/std.ix[x]        
        #print
        #print "Bollinger value dataframe"
        #print bb
        
        #MOMENTUM - 5 day momentum
        momentum=pd.DataFrame(data=0.0, columns=prices_NEW.columns,index=prices_NEW.index)
        for x in range(1,len(prices_NEW)):
            #print
            #print prices_NEW.ix[x]
            #print prices_NEW.ix[x-5]
            momentum.ix[x]= (prices_NEW.ix[x]/prices_NEW.ix[x-days2]) -1 
        #print
        #print momentum
        momentum=momentum.ix[sd:ed]
        #print
        #print "Momentum dataframe"
        #print momentum

        #Volatility (Standard deviation of daily return)
        volatility=pd.rolling_std(daily_returns_NEW,days1,ddof=0)
        volatility=volatility.ix[sd:ed]
        #print
        #print "Full Daily returns"
        #print daily_returns_NEW
        #print
        #print "Rolling Volatility dataframe"
        #print volatility
        
        ###############################################
        ###############Q-LEARNING######################
        ###############################################

        #BINNING        
        bin_bb=pd.qcut(bb.ix[:,0],10,labels=range(0,10))
        #print
        #print "Bin for Bollinger Bands"
        #print bin_bb        

        bin_momentum=pd.qcut(momentum.ix[:,0],10,labels=range(0,10))        
        #print
        #print "Bin for Momentum"
        #print bin_momentum

        bin_volatility=pd.qcut(volatility.ix[:,0],10,labels=range(0,10))        
        #print
        #print "Bin for volitility"
        #print bin_volatility

        #STATES
        state=pd.DataFrame(data=0, columns=prices.columns,index=prices.index,dtype=int)
        for i in range(len(state)):
            state.ix[i]=(bin_bb.ix[i]*100+bin_momentum.ix[i]*10+bin_volatility.ix[i])

        #start_state=bin_bb.ix[0]*100+bin_momentum.ix[0]*10+bin_volatility.ix[0]
        #print
        #print "The state dataframe is"
        #print state                                        
        
        #TRANSACTION COST
        TC=9.95        

        holdings=pd.DataFrame(0.0,columns=prices.columns,index=prices.index)        
        holdings['Cash']=0.0
        holdings['shareprice']=prices[:]        
        holdings.ix[0,1]=sv

        s=state.ix[0,0]
        a= self.learner.querysetstate(s)        
        #print        
        #print "the value of S is", s
        #print "the value of A is", a
            
        if (a==0): #long 200
            holdings.ix[1,0]=200
            holdings.ix[1,1]=holdings.ix[0,1]-200*prices.ix[0,0]-TC
            #cash next day = cash previous day - 200*share price -TC
        elif (a==1): #short 200
            holdings.ix[1,0]=-200
            holdings.ix[1,1]=holdings.ix[0,1]+200*prices.ix[0,0]-TC
        else: # no position
            holdings.ix[1,0]=0
            holdings.ix[1,1]=holdings.ix[0,1]
            
        #print "The holdings DataFrame is"
        #print holdings
        
        ### NO REWARD FUNCTION NEEDED SINCE TRAINING IS OVER

        s_prime=state.ix[1,0]            
        #print "S_prime is", s_prime

        for i in range(1,prices.shape[0]-1):            
            a= self.learner.querysetstate(s_prime)
            #print            
            #print "the value of S is", s_prime
            #print "the value of A is", a
            if (a==0): #long 200
                holdings.ix[i+1,0]=200
                if (holdings.ix[i+1,0]==holdings.ix[i,0]):
                    holdings.ix[i+1,1]=holdings.ix[i,1]
                    #print "A"
                else:
                    holdings.ix[i+1,1]=holdings.ix[i,1]-abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i,0]-TC
                    #print "B"
                #cash next day = cash previous day - 200*share price -TC
            elif (a==1): #short 200
                holdings.ix[i+1,0]=-200
                if (holdings.ix[i+1,0]==holdings.ix[i,0]):
                    holdings.ix[i+1,1]=holdings.ix[i,1]
                    #print "C"
                else:
                    holdings.ix[i+1,1]=holdings.ix[i,1]+abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i,0]-TC                    
                    #print "D"
            else:
                holdings.ix[i+1,0]=0
                if (holdings.ix[i+1,0]==holdings.ix[i,0]):
                    holdings.ix[i+1,1]=holdings.ix[i,1]
                    #print "E"
                elif (holdings.ix[i+1,0]>holdings.ix[i,0]):
                    holdings.ix[i+1,1]=holdings.ix[i,1]-abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i,0]-TC
                    #print "F"
                else:
                    holdings.ix[i+1,1]=holdings.ix[i,1]+abs(holdings.ix[i+1,0]-holdings.ix[i,0])*prices.ix[i,0]-TC                    
                    #print "G"
            #print "The holdings DataFrame is"
            #print holdings
            s_prime=state.ix[i+1,0]

        #print
        #print "Q-Learning DONE"
        #print "The HOLDINGS DataFrame is"
        #print holdings
        holdings['trades']=0
        #print "Length of holdings is ", len(holdings)
        for x in range(0,len(holdings)-1):            
            holdings.ix[x,3]= holdings.ix[x+1,0]-holdings.ix[x,0]
        holdings['portval']=0
        holdings['portval']=holdings.ix[:,0]*holdings.ix[:,2]+holdings.ix[:,1]
        #print
        #print holdings
        #print
        #print holdings['trades']
        result=holdings.copy()
        #print
        print "The result is"
        print result
        result.drop(result.columns[[0,1,2,4]],inplace=True,axis=1)
        result.ix[-2]=0
        result.ix[-1]=0
        #print
        #print result
        #print
        #print "Result DataFrame after dropping is"
        #print result
        
        """
        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:,:] = 0 # set them all to nothing
        trades.values[3,:] = 200 # add a BUY at the 4th date
        trades.values[5,:] = -200 # add a SELL at the 6th date 
        trades.values[6,:] = 200 # add a SELL at the 7th date 
        trades.values[8,:] = -400 # add a BUY at the 9th date
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        print
        print "IGNORE THIS FOR NOW"
        print trades
        """
        return result

if __name__=="__main__":
    print "One does not simply think up a strategy"
