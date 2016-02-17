from __future__ import division
import numpy as np
import librosa as lbr
import os
import sys
import glob

def extract_CQFT(wav_name,bins=84,stepsize=441,Fs=44100,N_fft=882):
    '''Extracts the log-spectrogram (CQFT) from wav_name'''
    wavcontent = lbr.load(wav_name,sr=Fs)
    return lbr.cqt(wavcontent[0],hop_length=stepsize,sr=Fs,n_bins=bins)

if __name__ == '__main__':
    folder = sys.argv[1]
    files = glob.glob(folder + '/*.wav')
    for wavfile in files:
        extract_CQFT(wavfile)
        np.savetxt(os.path.join(folder + '/' + )
