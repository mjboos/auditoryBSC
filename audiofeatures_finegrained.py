#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audiofeatures - script to extract audio features from WAV files

Dependencies:
    numpy - http://www.numpy.org/ - fundamental package for scientific computing with Python (BSD)
    essentia - http://essentia.upf.edu/ - audio analysis and audio-based music information retrieval (GPLv3)

Author: Michael A. Casey
Affiliation: Bregman Labs, Dartmouth College, Hanover, New Hampshire, USA
"""

usage="""usage: ./audiofeatures "PATH_TO_STIMULI/*.wav"

output (creates "features" directory in current directory if absent):
    features/fileA.hq_mfs
    features/fileA.lq_mfs
    features/fileA.mfcc
    features/fileA.mfs
    ...
    features/fileZ.hq_mfs
    features/fileZ.lq_mfs
    features/fileZ.mfcc
    features/fileZ.mfs

Features:
    hq_mfs  - high-quefrency mel-frequency spectrum (iDCT-II mfcc[nCoefs:])
    lq_mfs  - low-quefrency mel-frequency spectrum (iDCT-II mfcc[:nCoefs])
    mfcc    - mel-frequency cepstral coefficients >0 (DCT-II of mfs)
    mfs     - mel-frequency spectrum (magnitudes) 
    
Features are factored into low and high quefrency components such that:
    mfs = lq_mfs * hq_mfs
"""

import numpy as np
import essentia
import essentia.standard as es
import sys, os, glob

def dct_II(N):    
    """
    Create N x N matrix of discrete cosine transform coefficients
    Uses DCT-II formula:
    	https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    """
    d = np.array([np.cos(np.pi/N*(np.arange(N)+0.5)*k) for k in np.arange(N)],dtype='f4')
    d[0] *= 1/np.sqrt(2)
    d *= np.sqrt(2.0/N)
    return d

def extract_features(filename, nBands=96, nCoefs=13, N=442, H=221):
    """
    Extract audio features from overlapping windowed frames 
    inputs:
        filename - audio filename
        nBands - number of Mel bands [48]
        nCoefs - cepstrum coefficients to keep in low quefrency spectrum [13]
    outputs:
        pool - an essentia pool structure containing:
        	filename- name of audio file  
        	hq_mfs	- high-quefrency mel-frequency spectrum (iDCT-II mfcc[nCoefs:])
        	lq_mfs	- low-quefrency mel-frequency spectrum (iDCT-II mfcc[:nCoefs])
        	mfcc	- mel-frequency cepstral coefficients >0 (DCT-II of mfs)
        	mfs 	- mel-frequency spectrum (magnitudes) 
    
    Features are factored into low and high quefrency components such that:
    		mfs = lq_mfs * hq_mfs
    """
    loader = es.MonoLoader(filename=filename)
    audio = loader()
    pool = essentia.Pool()
    win = es.Windowing(type = 'hann')
    spectrum = es.Spectrum()
    melbands = es.MelBands(inputSize=N/2+1, numberBands=nBands)
    DCT = dct_II(nBands)
    for frame in es.FrameGenerator(audio, frameSize = N, hopSize = H):
        fft = spectrum(win(frame))
        mfs = melbands(fft)
        pool.add('mfs', mfs)
        mfcc = np.dot(DCT,20*np.log10(mfs+np.finfo(float).eps))
        pool.add('mfcc', mfcc)
        lq_mfs = np.dot(DCT[:nCoefs].T, mfcc[:nCoefs])
        lq_mfs = 10**(lq_mfs/20.)
        pool.add('lq_mfs', lq_mfs)
        hq_mfs = np.dot(DCT[nCoefs:].T, mfcc[nCoefs:])
        hq_mfs = 10**(hq_mfs/20.)
        pool.add('hq_mfs', hq_mfs)
    if filename is not None:        
	    pool.add('filename', os.path.splitext(os.path.basename(filename))[0])
    return pool

if __name__ == "__main__":
	expr = 'stimuli'+os.sep+'*.wav' if len(sys.argv)<2 else sys.argv[1]
print(expr)
file_list = sorted(glob.glob(expr))
if len(file_list)<1:
	print usage
	sys.exit(1)

directory = '/home/mboos/Work/PraktikumMD/fgs/'
if not os.path.exists(directory):
    os.makedirs(directory)

for f in file_list:
    feats = extract_features(f)
    for key in feats.descriptorNames():
            if key != 'filename':
                np.savetxt(os.path.join(directory, feats['filename'][0]+'fg.%s'%key), feats[key], delimiter=',')

