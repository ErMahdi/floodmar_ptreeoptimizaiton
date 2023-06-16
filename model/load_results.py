import sys
sys.path.append('..')

import numpy as np
# from folsom import Folsom
from folsom.folsom_mar import Folsom_MAR
from ptreeopt import PTreeOpt
from matplotlib import pyplot as plt
import pickle

#%%
'''Loading results with TOC activated'''

with open('Final_results/Feb23_2023_500Kfe', 'rb') as handle:
    snapshots1 = pickle.load(handle)
    
    

model1 = Folsom_MAR('folsom/data/folsom-daily-w2016.csv',
                sd='1993-10-01', ed='2016-09-30', use_tocs=False)


P1 = snapshots1['best_P'][-1]
df1 = model1.f(P1, mode='simulation')
#%%
fig, ax = plt.subplots(1, figsize = (6, 3), dpi = 300)
df1['Ss'].plot(ax= ax, color = (0.2,0.3,0.6),
              linestyle = '-', label = 'Policy Tree');
df1['storage'].plot(ax= ax, color = (0.7,0.2,0.3),
              linestyle = '-.', label = 'Historical');
df1['tocs'].plot(ax = ax, color = (0.5,0.5,0.5),
                linestyle = '--', label = 'TOC');
plt.legend()
