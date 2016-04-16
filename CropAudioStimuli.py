from __future__ import division
import numpy as np
from scipy.io import loadmat

duration = np.array([902,882,876,976,924,878,1084,676])
slice_nr = duration / 2

audiosegments = loadmat('/home/mboos/SpeechEncoding/AudioStimuli/TimeSeriesStimuli_16KHz.mat')
audiosegments = audiosegments['TS_data']
#indices of slice segments to delete
idxs = [ np.arange(slr-4,slr+4)+np.sum(slice_nr[:i]) for i,slr in enumerate(slice_nr[:-1])]

audiosegments = np.delete(audiosegments,idxs,axis=1).T

#now filter only speech

