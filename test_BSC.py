
# coding: utf-8

# # Auditory BSC

# Imports for pylib

# In[1]:

from __future__ import division
import sys
import numpy as np
import joblib
import pylab as plt
#get_ipython().magic(u'matplotlib inline')
import glob
from pulp.utils import create_output_path
from pulp.utils.parallel import pprint
from pulp.utils.barstest import generate_bars
from pulp.utils.autotable import AutoTable
import tables
from pulp.utils.parallel import stride_data

from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing

from pulp.em.camodels.bsc_et import BSC_ET

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import StandardScaler

plt.rcParams['image.cmap'] = 'viridis'

#np.seterr(all='raise')


# ## Pre-processing the data
# 
# 1. Log-Transform of Mel-Spectrogram
# 2. Extracting image patches
# 3. Compute and substract intercept
# 4. Normalize? Before or after extraction?

# Import the Mel-Frequency spectrum (MFS) data from the Forrest Gump Audio-Movie

# In[2]:

lqmfs_list = glob.glob('data/*.mfs')
feature_list = [np.genfromtxt(lqmfs_fn,delimiter=',') for lqmfs_fn in lqmfs_list]
ft_freq = feature_list[0].shape[1]


# Extracts speech patches (for a small part to test) of the log of the MFS data

# In[3]:

test_part = (feature_list[0])
scaler = StandardScaler()

#perc_clip = lambda x : np.percentile(x,0.999)

#test_part = perc_clip(test_part)

test_part = scaler.fit_transform(test_part)

test_part -= np.min(test_part)-0.001


test_part = np.log(test_part)

#test_part = test_part / np.sum(test_part,axis=1)[:,np.newaxis]
patchsize = (4,48)
patches_flat = np.reshape(extract_patches_2d(test_part,patchsize),(test_part.shape[0] - patchsize[0] + 1,-1))

#patches_flat = scaler.fit_transform(patches_flat)

#patches_flat = (patches_flat-np.mean(patches_flat))/np.std(patches_flat)


# In[4]:

output_path = create_output_path()

N = patches_flat.shape[0]
D = patches_flat.shape[1]

H = 100

Hprime = 7
gamma = 5

model = BSC_ET(D, H, Hprime, gamma, to_learn=['W','sigma','pi'])

data = {'y':patches_flat}


out_fname = output_path + "/data.h5"


# In[5]:



#setting up logging/output
print_list = ('T', 'Q', 'pi', 'sigma', 'N', 'MAE', 'L')
dlog.set_handler(print_list, TextPrinter)
h5store_list = ('W', 'pi', 'sigma', 'y', 'MAE', 'N','L','Q')
dlog.set_handler(h5store_list, StoreToH5, output_path +'/result.h5')

###### Initialize model #######
# Initialize (Random) model parameters
model_params = model.standard_init(data)
model_params['mu'] = np.mean(data['y'],axis=0)

#### Choose annealing schedule #####
#Linear Annealing
anneal = LinearAnnealing(100)
#Increases variance by a muliplicative factor that slowly goes down to 1
anneal['T'] = [(0., 6.), (.3, 1.)]      # [(iteration, value),... ]
#Reduces truncation rate so as not to prematurely exclude data 
anneal['Ncut_factor'] = [(0, 2.0), (.25, 1.)]     
#Simulated annealing of parameters
anneal['W_noise'] = [(0, 2.0), (.3, 0.0)]
#Include prior parameters in the annealing schedule
anneal['anneal_prior'] = False


# Create and start EM annealing
em = EM(model=model, anneal=anneal)
em.data = data
em.lparams = model_params
em.run()

dlog.close(True)
pprint("Done")


