
# coding: utf-8




from __future__ import division
import sys
import yaml
import numpy as np
import joblib
import pylab as plt
#get_ipython().magic(u'matplotlib inline')
from pulp.utils import create_output_path
from pulp.utils.parallel import pprint
from pulp.utils.autotable import AutoTable
import tables
from pulp.utils.parallel import stride_data

from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing

from pulp.em.camodels.bsc_et import BSC_ET

plt.rcParams['image.cmap'] = 'viridis'

data_file, model_file, job_name = sys.argv[1:]

patches = joblib.load(data_file)['data']

with open(model_file,'r') as model_fh:
    model_spec = yaml.load(model_fh)

H, Hprime, gamma, n_anneal, Ncut, Wnoise, T = [model_spec[key] for key in ['H','Hprime','gamma', 'n_anneal',
                                                                'N_cut', 'W_noise', 'T']] 

N = patches.shape[0]
D = patches.shape[1]

#### Choose annealing schedule #####
#Linear Annealing
anneal = LinearAnnealing(n_anneal)

#Increases variance by a muliplicative factor that slowly goes down to 1
anneal['T'] = T      # [(iteration, value),... ]

#Reduces truncation rate so as not to prematurely exclude data 
anneal['Ncut_factor'] = Ncut     

#Simulated annealing of parameters
#anneal['W_noise'] = Wnoise

#Include prior parameters in the annealing schedule
anneal['anneal_prior'] = False


output_path = create_output_path(basename = job_name)

model = BSC_ET(D, H, Hprime, gamma, to_learn=['W','sigma','pi','mu'])

data = {'y':patches}


out_fname = output_path + "/data.h5"


#setting up logging/output
print_list = ('T', 'Q', 'pi', 'sigma', 'N', 'MAE', 'L')
dlog.set_handler(print_list, TextPrinter)
h5store_list = ('W', 'pi', 'sigma', 'y', 'MAE', 'N','L','Q','mu')
dlog.set_handler(h5store_list, StoreToH5, output_path +'/result.h5')

###### Initialize model #######
# Initialize (Random) model parameters
model_params = model.standard_init(data)

#model_params['mu'] = np.mean(data['y'],axis=0)

# Create and start EM annealing
em = EM(model=model, anneal=anneal)
em.data = data
em.lparams = model_params
em.run()

dlog.close(True)
pprint("Done")


