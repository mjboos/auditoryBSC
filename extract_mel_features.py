from __future__ import division
import numpy as np
import librosa as lbr
import sys
import glob
from scipy.io.wavfile import read

def extract_mel(wav_name, bins=48, stepsize=4410, n_fft=8820, fmax=8000):
    '''Extracts the mel-frequency spectrogram from wav_name'''
    sr, wavcontent = read(wav_name)
    wavcontent = np.mean(wavcontent,axis=1) 
    return lbr.feature.melspectrogram(wavcontent,hop_length=stepsize,sr=sr,n_mels=bins,fmax=fmax).T

if __name__ == '__main__':
    folder = sys.argv[1]
    files = glob.glob(folder + '/*.wav')
    for wavfile in files:
        mel = extract_mel(wavfile)
        np.savetxt(wavfile.split('.wav')[0] + '.mfs',mel,delimiter=',')

