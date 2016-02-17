from __future__ import division
import numpy as np
import scipy as sp

def find_speech_chunks(speecharray,i):
	start = speecharray[i,0]
	stop = speecharray[i,1]
	if speecharray.shape[0] == i+1:
		return ((start,stop),-1)
	for j in xrange(i+1,speecharray.shape[0]):
		if speecharray[j,0] < stop:
			stop = max(speecharray[j,1],stop)
		else:
			break            
	return ((start,stop),j)


