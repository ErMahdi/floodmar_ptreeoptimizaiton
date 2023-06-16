from __future__ import division
import numpy as np
import pandas as pd
import pickle 
cfs_to_taf = 2.29568411 * 10**-5 * 86400 / 1000
taf_to_cfs = 1000 / 86400 * 43560


def water_day(d):
    return d - 274 if d >= 274 else d + 91


def max_release(S):
    # rule from http://www.usbr.gov/mp/cvp//cvp-cas/docs/Draft_Findings/130814_tech_memo_flood_control_purpose_hydrology_methods_results.pdf
    storage = [90, 100, 400, 600, 975]
    # make the last one 130 for future runs
    release = cfs_to_taf * np.array([0, 35000, 40000, 115000, 130000])
    return np.interp(S, storage, release)
    

    
def recharge(x0, x1, x2, prior_rech):
    if prior_rech <40:
        max_lim = 5
    elif prior_rech < 43:
        max_lim = 2
    else:
        max_lim = 0
    #this function prevnets recharge in dry and critical year
    features = np.array([1, x0, x1, x2, x0**2, x0*x1, x0*x2, x1**2, x1*x2, x2**2, x0**3, x0**2*x1, x0**2*x2, x0*x1**2, x0*x1*x2, x0*x2**2, x1**3, x1**2*x2, x1*x2**2, x2*3])
    coefs = np.array([1.0353,	-9.05756, -16.9389,	-0.185705, 92.4603, -157.165, 0.00858223, -46.9463, 0.272325, 7.23785e-08, -6.5868, 4.81094,
                      0.0001193, 13.8283, -0.00218489, 3.72385e-09, 14.9746, -0.0439376, -9.5806e-08, 6.03578e-13])


    return max(min(np.sum(features*coefs)/1000,max_lim),0) if x1 not in [1,2] else 0
    

def tocs(d):
    # d must be water-year date
    # TAF of flood capacity in upstream reservoirs. simplified version.
    # approximate values of the curve here:
    # http://www.hec.usace.army.mil/publications/ResearchDocuments/RD-48.pdf
    tp = [0, 50, 151, 200, 243, 366]
    sp = [975, 400, 400, 750, 975, 975]
    
    return np.interp(d, tp, sp)
    
    

def volume_to_height(S):  # from HOBBES data
    sp = [0, 48, 93, 142, 192, 240, 288, 386, 678, 977]
    ep = [210, 305, 332, 351, 365, 376, 385, 401, 437, 466]
    return np.interp(S, sp, ep)


class Folsom_MAR():

    def __init__(self, datafile, sd, ed,
                 fit_historical=False, use_tocs=False,
                 cc=False, scenario=None, multiobj=False):
        
        self.df = pd.read_csv(datafile, index_col=0, parse_dates=True)[sd:ed]
        self.K = 975  # capacity, TAF
        self.turbine_elev = 134  # feet
        self.turbine_max_release = 8600  # cfs
        self.max_safe_release = 130000  # cfs
        extended =  self.df.index.union(pd.date_range((self.df.index +pd.DateOffset(1))[-1], periods=3, freq='D'))
 
        self.dowy = np.array([water_day(d) for d in (extended).dayofyear])
        self.Month = self.df.index.month  
        self.Day = self.df.index.day
        self.WYT = self.df.WYT.values       
        self.D = np.loadtxt('folsom/data/demand_folsim.txt')[self.dowy[:-3]]
        self.T = len(self.df.index)
        self.fit_historical = fit_historical
        self.use_tocs = use_tocs
        self.cc = cc
        self.multiobj = multiobj
        self.Q = self.df.inflow.values


    def set_scenario(self, s):
        self.scenario = s
        self.annQ = self.annQs[s].values
        self.lp3 = self.lp3s[s].values
        self.wyc = self.wycs[s].values
        self.Q = self.df[s].values

    def f(self, P, mode='optimization'):

        T = self.T
        S, R, target, shortage_cost, flood_cost , RechPenalty, TocPenalty = [
            np.zeros(T) for _ in range(7)]
        Rech = np.ones(T)*0
        PRech = np.ones(T)*0
        K = self.K
        D = self.D
        Q = self.Q
        WYT = self.WYT
        dowy = self.dowy
        R[0] = D[0]
        policies = [None]
 
        month = self.Month
        day = self.Day



        if not self.cc:
            S[0] = self.df.storage.values[0]
        else:
            S[0] = 500

        for t in range(1, T):

            if not self.cc:
                policy, rules = P.evaluate([S[t - 1], self.dowy[t], Q[t], month[t], WYT[t]])
            else:
                y = self.years[t] - 2000
                policy, rules = P.evaluate([S[t - 1], Q[t], dowy[t],
                                            self.annQ[y], self.lp3[y], self.wyc[y]])
            
            
     
                
                
            if policy == 'Release_Demand':
                target[t] = D[t]
            elif policy == 'Recharge':
                if month[t] in [11,12,1,2,3]:
                    anticident_rech = np.sum(Rech[t-(day[t]-1):t])
                    Rech[t] = recharge(month[t],WYT[t],1000*Q[t], anticident_rech)  
                target[t] = D[t]+ Rech[t]
            elif policy == 'Hedge_90':
                target[t] = D[t] *0.9
            elif policy == 'Hedge_80':
                target[t] = D[t] *0.8
            elif policy == 'Hedge_70':
                target[t] = D[t]*0.7
            elif policy == 'Hedge_60':
                target[t] = D[t]*0.6
            elif policy == 'Hedge_50':
                target[t] = D[t]*0.5
            elif policy == 'TOC':
                target[t] = max(0.2*(Q[t] + S[t - 1] - tocs(dowy[t])), D[t])
                
            elif policy == 'TOC_Recharge':
                if month[t] in [11,12,1,2,3]:
                    anticident_rech = np.sum(Rech[t-(day[t]-1):t])
                    Rech[t] = recharge(month[t],WYT[t],1000*Q[t], anticident_rech) 
                target[t] = max(0.2*(Q[t] + S[t - 1] - tocs(dowy[t])), (D[t]+Rech[t]))
                
            if self.use_tocs:
                target[t] = max(
                    0.2 * (Q[t] + S[t - 1] - tocs(dowy[t])), target[t])
            elif policy == 'Flood_Control_Rech':
                if month[t] in [11,12,1,2,3]:
                    anticident_rech = np.sum(Rech[t-(day[t]-1):t])
                    Rech[t] = recharge(month[t],WYT[t],1000*Q[t], anticident_rech) 
                target[t] = max(0.2 * (Q[t] + S[t - 1] - 0.0), D[t]+Rech[t], 0.0)  # default

            if mode == 'simulation':
                policies.append(policy)
                if month[t] in [11,12,1,2,3]:
                    anticident_rech = np.sum(PRech[t-(day[t]-1):t])
                    PRech[t] = recharge(month[t],WYT[t],1000*Q[t], anticident_rech) 
            
            # max/min release
            # k = 0.2
            R[t] = min(target[t], S[t - 1] + Q[t])
            R[t] = min(R[t], max_release(S[t - 1]))
            # R[t] = np.clip(R[t], (1-k)*R[t], (1+k)*R[t]) # inertia --
            R[t] += max(S[t - 1] + Q[t] - R[t] - K, 0)  # spill
            S[t] = S[t - 1] + Q[t] - R[t]
    
            # squared deficit. Also penalize any total release over 100 TAF/day
            # should be able to vectorize this.
            shortage_cost[t] = max(D[t] - R[t], 0)**2 / \
                T  # + max(R[t]-100, 0)**2

            if R[t] > cfs_to_taf * self.max_safe_release:
                # flood penalty, high enough to be a constraint
                flood_cost[t] += 10**3 * \
                    (R[t] - cfs_to_taf * self.max_safe_release)

            
                
            #penalyzing recharge in non-recharge months
            if policy in ['Recharge','Flood_Control_Rech', 'TOC_Recharge'] and (month[t] not in [1,2,3,11,12]):
                RechPenalty[t] = 100
                Rech[t] = 0
            
        Rech[0] = 0
        

            
        
        

        if mode == 'simulation' or self.multiobj:
            
            topcons = tocs(dowy)            
            df = self.df.copy()
            df['Ss'] = pd.Series(S, index=df.index)
            df['Rs'] = pd.Series(R, index=df.index)
            df['demand'] = pd.Series(D, index=df.index)
            df['target'] = pd.Series(target, index=df.index)
            head = (volume_to_height(df.Ss) - self.turbine_elev)
            power_release = (taf_to_cfs * df.Rs.copy()).clip(0, self.turbine_max_release)

            df['HydRs'] = pd.Series(power_release, index = df.index)
            df['head'] =  pd.Series(head, index = df.index)
            # P = 0.85*((Hydro_R*1000)/59.5)*(h/(10000*1.181*0.25))
            df['power'] = (24 * 0.85 * 10**-4 / 1.181) * (head * power_release)
            df['Rech'] = pd.Series(Rech, index=df.index)
            df['PRech'] = pd.Series(PRech, index = df.index)
            df['tocs'] = pd.Series(topcons[:-3], index=df.index)
            df['scarcity_cost'] = pd.Series(shortage_cost, index = df.index)
            
        
        if mode == 'simulation':
            df['policy'] = pd.Series(
                policies, index=df.index, dtype='category')
            return df
        else:
            if self.fit_historical:
                return np.sqrt(np.mean((S - self.df.storage.values)**2))
            else:
                if not self.multiobj:
                    return shortage_cost.sum() + flood_cost.sum() - np.sum(Rech)/(2393) +  RechPenalty.sum() # + EOP
                else:
                    J1 = shortage_cost.sum()  # water supply
                    J2 = taf_to_cfs * df.Rs.max()  # peak flood
                    # (3) environmental alteration:
                    # integrate between inflow/outflow exceedance curves
                    ixi = np.argsort(Q)
                    ixo = np.argsort(R)
                    J3 = np.trapz(np.abs(Q[ixi] - R[ixo]), dx=1.0 / T)
                    # (4) Maximize hydropower generation (GWh)
                    # Max turbine release 8600 cfs, 215 MW. Average annual production 620 GWh.
                    # http://www.usbr.gov/mp/EWA/docs/DraftEIS-Vol2/Ch16.pdf
                    # http://rivers.bee.oregonstate.edu/book/export/html/6
                    J4 = -df.power.resample('AS-OCT').sum().mean() / 1000
                    return [J1, J2, J3, J4]
