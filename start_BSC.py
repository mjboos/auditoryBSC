from __future__ import division
import yaml
import sys
import os
import argparse
import glob
from time import strftime
import subprocess
from itertools import product
from pulp.utils.autotable import AutoTable


def list_of_tuples(arg_string):
    '''parses the string representation of a list of tuples into a list of tuples'''
    return [tuple( float(nr) for nr in tup.strip('()').split(',')) for tup in arg_string.strip('[]').split('),') ]

def augment_filename(filename,data_path='/home/mboos/MasterThesis/data/preprocessed/'):
    return data_path + filename


model_spec_path = '/home/mboos/MasterThesis/model_specifications/'

log_path = '/home/mboos/MasterThesis/logs/'

##default values for the model parameters
H = 50
Hprime = 8
gamma = 5
n_anneal = 100
N_cut = [(0,2.0),(.25,1.)]
T = [(0.,6.),(.3,1.)]


parser = argparse.ArgumentParser()

parser.add_argument('--H',default = H,type = int, nargs = '*')
parser.add_argument('--Hprime',default = Hprime,type = int, nargs = '*')
parser.add_argument('--gamma',default = gamma,type = int, nargs = '*')
parser.add_argument('--n_anneal',default = n_anneal,type = int)
parser.add_argument('--N_cut',default = N_cut,type = list_of_tuples)
#parser.add_argument('--W_noise',default = W_noise,type = list_of_tuples)
parser.add_argument('--T',default = T,type = list_of_tuples)
parser.add_argument('--data',default = max(glob.glob('/home/mboos/MasterThesis/data/preprocessed/*.h5'),key=os.path.getctime).split('/')[-1],type = augment_filename, nargs = '*')
parser.add_argument('--name',default = 'model@' + strftime("%H:%M:%S"), type = str)
parser.add_argument('--parallel', default = 1, type = int)


arg_namespace = vars(parser.parse_args())

parallel = arg_namespace.pop('parallel')


common_dict = {key:value for key,value in arg_namespace.iteritems() if key in ['T', 'N_cut']}

common_dict.update({key:value for key,value in arg_namespace.iteritems() if not isinstance(value,list)})

if len(common_dict) == len(arg_namespace):
    model_spec_fn = model_spec_path + arg_namespace['name'] + '.yml'
    log_fn = log_path + arg_namespace['name'] + '.log'


    
    if os.path.isfile(model_spec_fn):
        model_spec_fn = model_spec_path + arg_namespace['name'] + strftime("%H:%M:%S") + '.yml'

    if os.path.isfile(log_fn):
        log_fn = log_path + arg_namespace['name'] + strftime("%H:%M:%S") + '.log'

    #write model file
    with open(model_spec_fn,'w+') as fh:
        fh.write(yaml.dump(arg_namespace))

    with open(log_fn,'w+') as fh:
        subprocess.Popen(['nohup','python','analyze_BSC.py',arg_namespace['data'],model_spec_fn,arg_namespace['name']],stdout=fh)

    print('Job started!')
else:
    list_keys, list_values = zip(*((key,value) for key,value in arg_namespace.iteritems() if key not in common_dict.keys()))

    model_name = arg_namespace['name']

    for i,combination in enumerate(product(*list_values)):
        model_dict = dict(zip(list_keys,combination))
        model_dict.update(common_dict)
        model_dict['name'] += 'c{}'.format(i)
        model_spec_fn = model_spec_path + model_dict['name'] + '.yml'
        log_fn = log_path + model_dict['name'] + '.log'

        if os.path.isfile(model_spec_fn):
            model_spec_fn = model_spec_path + model_dict['name'] + strftime("%H:%M:%S") + '.yml'

        if os.path.isfile(log_fn):
            log_fn = log_path + model_dict['name']+ strftime("%H:%M:%S") + '.log'

        #write model file
        with open(model_spec_fn,'w+') as fh:
            fh.write(yaml.dump(model_dict))

        with open(log_fn,'w+') as fh:
            subprocess.Popen(['nohup','python','analyze_BSC.py',model_dict['data'],model_spec_fn,model_dict['name']],stdout=fh)

        print('Job started!')
