from __future__ import division
from json import load
import numpy as np
import joblib
import glob
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.preprocessing import StandardScaler
import sys

patchsize = sys.argv[1:]


if patchsize == []:
    patchsize = (10,48)
else:
    patchsize = map(int,patchsize)

with open('DialogData/german_dialog_20150211.json') as fh:
	dialog = load(fh)
with open('DialogData/german_audio_description_20150211.json') as fh:
	description = load(fh)

dialog_SE = [(anno['begin'],anno['end']) for anno in dialog['annotations']]
description_SE = [(anno['begin'],anno['end']) for anno in description['annotations']]

speech_SE = dialog_SE + description_SE
speech_arr = np.array(speech_SE)
speech_arr = speech_arr[np.argsort(speech_arr[:,0]),:]

#MFS stepsize is 10ms, speech begin/end is in ms, so we divide by 10
speech_arr = speech_arr / 10

duration = np.array([902,882,876,976,924,878,1084,676])

movie = [np.genfromtxt(segment,delimiter=',') for segment in sorted(glob.glob('data/evenfinergrained/*'))]

#cut off parts of the movie that come after the last TR slice
movie = [movie[i][:-int(np.round(100.*((movie[i].shape[0]/100.) % 2))),:] for i in xrange(len(movie))]

def cut_out_overlap(left_array,right_array):
    return np.concatenate((left_array[:-800],right_array[800:]),axis=0)

movie = reduce(cut_out_overlap,movie)

movie_patches = extract_patches_2d(movie,patchsize)

#joblib.dump(movie_patches,'data/movie_patches.pkl',compress=3)


#delete non-speech parts
speech = np.concatenate([ movie_patches[spt[0]:spt[1]] for spt in speech_arr])


joblib.dump(speech,'data/speech_patches.pkl')
