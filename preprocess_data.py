from __future__ import division
import joblib
import sys
from sklearn.preprocessing import StandardScaler
from numpy import reshape
import argparse

def zscore_data(data_fn,name,n_splits=3):
    '''Z-scores the data in data_fn and saves the n_splits splits'''
    names = ['full','half','quarter','8th','16th']
    data = joblib.load('/home/mboos/MasterThesis/data/' + data_fn)
    patchsize = data.shape[1:]
    data = reshape(data,(data.shape[0],-1))
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    for n in xrange(n_splits):
        joblib.dump({'data':data[:int(data.shape[0]/(2**n))],'patchsize':patchsize},'data/preprocessed/' + name + '_zscored_' + names[n] + '.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data',type=str)
    parser.add_argument('name',type=str)
    parser.add_argument('--splits',type=int,default=3)
    args = vars(parser.parse_args())
    zscore_data(args['data'],args['name'],args['splits'])

