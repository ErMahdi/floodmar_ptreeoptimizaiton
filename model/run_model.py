import sys
sys.path.append('..')

import numpy as np
from folsom.folsom_mar import Folsom_MAR


from ptreeopt import PTreeOpt

import logging
#%%
# Example to run optimization and save results
np.random.seed(17)

model = Folsom_MAR('folsom/data/folsom-daily-w2016.csv',
                sd='1993-10-01', ed='2016-09-30', use_tocs=False)


import time

st = time.time()
algorithm = PTreeOpt(model.f,
                     feature_bounds=[[90, 975], [1, 365], [1, 400], [1,12], [1,5]],
                     feature_names=['Storage', 'Day', 'Inflow', 'Month', 'WYT'],

                     discrete_actions=True,
                     action_names=['Release_Demand', 'Recharge', 'Hedge_90',
                                   'Hedge_80', 'Hedge_70', 'Hedge_60',
                                   'Flood_Control_Rech',
                                   # 'TOC', 'TOC_Recharge',
                                   ],
                     mu=20,
                     cx_prob=0.7,
                     population_size=100,
                     max_depth=7
                     )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
        format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')
    
    best_solution, best_score, snapshots = algorithm.run(max_nfe=100000, 
                                                     log_frequency=100,
                                                     snapshot_frequency=100)
    ttime = time.time()-st
    print(ttime)