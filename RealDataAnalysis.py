# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 17:20:26 2015

@author: mboos
"""

#trying the real data
from scipy.io import loadmat
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import ElasticNetCV,ElasticNet,LinearRegression
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.svm import SVR,NuSVR
from sklearn import grid_search
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.feature_extraction.image import extract_patches_2d
from sparse_filtering import SparseFiltering
#%%
#helper functions


def recover_stimuli(X_lagged,nfeat):
    '''X_lagged has frames X nlags*nfeat, starting from no. of columns/nfeat ms of audio begin'''
    unlagged_stimuli = []
    for i,lagged_row in enumerate(X_lagged):
        if unlagged_stimuli == [] or not all(X_lagged[i-1,:nfeat]==X_lagged[i,nfeat:2*nfeat]):
            unlagged_stimuli.append(np.reshape(lagged_row,(nfeat,lagged_row.shape[0]/nfeat),order='F')[:,::-1])
        else:
            unlagged_stimuli[-1] = np.concatenate((unlagged_stimuli[-1],lagged_row[:nfeat,None]),axis=-1)
    return unlagged_stimuli
#    


def codes_to_stimuli(codes,nstims,nfeat,nlags,patchsize):
    '''brings codes into lagged stimuli representation'''
    #start at 25ms, which is index (25-patchsize[1]) (time) 
    lagged_stim = []
    patch_stim_lens = [(i-patchsize[1]+1)*(nfeat-patchsize[0]+1) for i in nstims]
    #for each stimulus    
    for i,stimsize in enumerate(patch_stim_lens):
        lagged_stim.append([])
        #in the code vector  
        for j in xrange(int(np.sum(patch_stim_lens[:i])),int(np.sum(patch_stim_lens[:i]))+nstims[i]-nlags+1):
            #for all timewindows in the lagged region
            one_lagged = (codes[j:j+(nfeat-patchsize[0]+1)*(nstims[i]-patchsize[1]+1):(nstims[i]-patchsize[1]+1)]).flatten()
            for k in xrange(int(j)+1,int(j)+nlags-patchsize[1]+1):
                #now stride over features for this timepoint
                one_lagged = np.concatenate(((codes[k:k+(nfeat-patchsize[0]+1)*(nstims[i]-patchsize[1]+1):(nstims[i]-patchsize[1]+1)]).flatten(),one_lagged))
            lagged_stim[-1].append(one_lagged)
        lagged_stim[-1] = np.vstack(lagged_stim[-1])
    return lagged_stim

def get_X_stim_sizes(X,nfeat):
    ex = enumerate(X)
    ex.next()
    return [ i for i,lagged_row in ex if not np.all(X[i-1,:nfeat]==X[i,nfeat:2*nfeat])]

#%%
#for YAO
mat = loadmat('/home/cmicheli/Frontiers/Feat_Specgram_STFT_25_lags_subjYAO.mat')
X = mat["Feat"]
resp = loadmat('/home/cmicheli/Frontiers/Resp_Specgram_STFT_subjYAO_HG_band.mat')
Y = np.concatenate(resp["Resp"][0,:],axis=1).T
#sort it
Y = Y[np.squeeze(np.argsort(np.concatenate([mat["assignAV"],mat["assignA"]]),axis=0)),:]
YAO_labels = loadmat('/home/cmicheli/Frontiers/YAO_layout.mat')

X[np.isinf(X)] = 0


electrode_names = [ str(lab[0][0]) for lab in YAO_labels['labels'] ]

#freq = YAO_pow['pow_dpss']['freq'][0][0][0]


#use G13
y = Y[:,electrode_names.index('G13')]


#%%
#creating different features
#add squared and cubed features


#%%

#%for standardizing in lagged stimuli space
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#add intercept
X = np.hstack((np.ones(X.shape[0])[:,None],X))

train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.1)


#%%
#try elastic net

#alpha equals lambda here
lambda_grid = [0.01, 0.1 , 1, 10,100]
l1_ratio_grid = [0.1,0.3,0.5,0.7,0.9]

enet_CV = ElasticNetCV(l1_ratio=l1_ratio_grid,alphas=lambda_grid,cv=3,n_jobs=-1,verbose=True)

enet_CV.fit(train_X,train_Y)

#%%
#show
enet_CV.score(test_X,test_Y)
plt.plot(enet_CV.predict(test_X),test_Y,'o')
#%%
#try svr

svr = SVR(kernel = 'rbf',C=1,cache_size=2000)

SVR_params = { 'C' : [1e-1,1.0,1e2,1e3,1e4] }
svr_rs = grid_search.RandomizedSearchCV(svr,SVR_params,verbose=True,n_jobs=-1)

svr.fit(train_X[:,whichones[0]],train_Y)

#%%
#try bagging/boosting etc
#rfr = RandomForestRegressor(n_estimators = 30,n_jobs = 2)

#rfr.fit(train_X,train_Y)



gbr = GradientBoostingRegressor(loss='ls',n_estimators=100,learning_rate=0.1,max_leaf_nodes=9,verbose=True,subsample=0.5)

gbr.fit(train_X,train_Y)

test_deviance = [ gbr.loss_(y_pred,test_Y) for y_pred in gbr.staged_decision_function(test_X)]

#%%
####################################
#DECOMPOSITION STARTS HERE
#############################
#try approx pca, probabilistic PCA, sparse pca, kernel pca
#with: linear, non-linear methods

#try the decomposition on the unlagged audio, then re-lag them
#no normalization in X needed here

#for 33 features
nfeat=33
unlagged_stimuli = recover_stimuli(X,nfeat)
nstims = [stim.shape[1] for stim in unlagged_stimuli]

#%%
#no patches

unlagged_stimuli = np.hstack(unlagged_stimuli).T

#test if normalization needed
stim_scaler = preprocessing.StandardScaler()
stim_scaler.fit(unlagged_stimuli)
unlagged_stimuli = stim_scaler.transform(unlagged_stimuli)

#%%
#extract patches
#needs unlagged stimuli as list

patchsize = (16,16)
stimuli_patches = np.vstack([ patches.reshape((patches.shape[0],-1)) for stimulus in unlagged_stimuli   for patches in [extract_patches_2d(stimulus,patchsize)] ])


#normalize them
patch_scaler = preprocessing.StandardScaler()
stimuli_patches = patch_scaler.fit_transform(stimuli_patches)



#%%
nfeat = 15
rpca = decomposition.RandomizedPCA(n_components=nfeat,whiten=True)
rpca.fit(unlagged_stimuli)

unlagged_stimuli = rpca.transform(unlagged_stimuli)

#%%
#sparse pca
spca = decomposition.SparsePCA(n_jobs=-1)
spca.fit(unlagged_stimuli)

unlagged_stimuli = spca.transform(unlagged_stimuli)


#%%
#dictionary minibatch
mbdic = decomposition.MiniBatchDictionaryLearning(n_components=50,verbose=True)
mbdic.fit(stimuli_patches)

#%%
#visualize

V = mbdic.components_
plt.figure()
for i,comp in enumerate(V):
    plt.subplot(10,10,i+1)
    plt.imshow(comp.reshape(patchsize),interpolation='nearest')
    
    
#%%
#now construct code representation for stimuli
codes = mbdic.transform(stimuli_patches[:sum(patch_stim_lens[:100]),:])    
#how are these patches constructed?
#over last dimension first, slide by 1 timepoint
#every ncol-9 move on up (or down if origin is upper left)
#to get time representation, take coefficients up until this time point (watch out there is overlap in patches)
#(so on last dimension) then take all 24 patches on frequency axes (stride by ncol-9)
nlags = 25
lagged_codes = codes_to_stimuli(codes,nstims,nfeat,nlags,patchsize)

#%%
#classify with new data
#first re-create lagged representation from data
nlags = 25

unlagged_stimuli = [ unlagged_stimuli[sum(nstims[:i]):sum(nstims[:i])+nstimsize,:].T  for i,nstimsize in enumerate(nstims) ] 

#lagged_stimuli = [  np.vstack( np.concatenate((np.zeros((nlags-i)*nfeat),stimulus[:,max(0,i-nlags):i].flatten(order='F'))) if i < nlags else stimulus[:,max(0,i-nlags):i].flatten(order='F') for i in xrange(1,stimulus.shape[1]+1) ) for stimulus in unlagged_stimuli ]

new_X = [  np.vstack( np.concatenate((stimulus[:,max(0,i-nlags):i][:,::-1].flatten(order='F'),np.zeros((nlags-i)*nfeat))) if i < nlags else stimulus[:,max(0,i-nlags):i][:,::-1].flatten(order='F') for i in xrange(1,stimulus.shape[1]+1) )[nlags-1:,:] for stimulus in unlagged_stimuli ]

#now stack these vertically
new_X = np.vstack(new_X)

#%%
#standardize again? prob not
predictor_scaler = preprocessing.StandardScaler() 
new_X = predictor_scaler.fit_transform(new_X)

#%%
new_X = np.hstack((np.ones(new_X.shape[0])[:,None],new_X))
train_X,test_X,train_Y,test_Y = train_test_split(new_X,y,test_size=0.3)


