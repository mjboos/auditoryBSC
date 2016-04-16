
# coding: utf-8

# # speechnotparallel model visualisation

# In[1]:

from __future__ import division
import sys
import numpy as np
import joblib
import pylab as plt
import h5py
import glob
import yaml
from pulp.em.camodels.bsc_et import BSC_ET
from pulp.em.annealing import LinearAnnealing
plt.rcParams['image.cmap'] = 'viridis'
#np.seterr(all='raise')


# In[7]:

model_name = 'LogZscored2c0'

with open('model_specifications/' + model_name + '.yml','r') as fh:
    model_specifications = yaml.load(fh)

#data = joblib.load('/home/mboos/MasterThesis/data/preprocessed/100ms_speech_10ms_stepsize_zscored_full.pkl')
data = joblib.load('data/flat_patches.pkl')
if isinstance(data,dict):
    patchsize = data['patchsize']
    data = data['data']
else:
    patchsize = (10,48)

model_specifications['D'] = patchsize[0] * patchsize[1]


# In[8]:



# In[3]:

with h5py.File(glob.glob('output/'+model_name+'*')[0] + '/result.h5','r') as results:
    L,  mu, W, sigma, pi = [results[key][()] for key in ['L',
                                                       'mu',
                                                       'W',
                                                       'sigma',
                                                       'pi']]


# In[14]:

bsc = BSC_ET(*[model_specifications[key] for key in ['D','H','Hprime','gamma']])

data = {'y':data}
model_params = {'W':W[-1],'pi':pi[-1],'sigma':sigma[-1]}

anneal = LinearAnnealing(1)
anneal['T'] = [(0,1.)]
inferred_data = bsc.inference(anneal,model_params,data,no_maps=1)

#test_patches = np.reshape(data['y'],(data['y'].shape[0],)+patchsize)

#inferred_data = {key:np.squeeze(value) for key,value in inferred_data.iteritems()}


# In[47]:



# In[66]:

joblib.dump(inferred_data['s'],'log_sparse_patches.pkl', compress=3)

