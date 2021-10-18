import numpy as np
import h5py
import logging
import scipy.io
from tqdm import tqdm 
from sklearn import preprocessing
import os
import pandas as pd
import re 

ROOT_DIR = '/itet-stor/wolflu/net_scratch/projects/EEGEyeNet_experimental/data/dots_data/synchronised_min/' # modify it if necessary 
DATASET = 'dots'
SAVE_DIR = "/itet-stor/wolflu/net_scratch/projects/EEGEyeNet_experimental/data/segmentation/data_steam_"
FILE_PATTERN = re.compile('(ep|EP).._DOTS._EEG.mat')

def load_sEEG(abs_dir_path):
    """
    Extracts the sEEG.event section of a participants mat file 
    Returns the events as a numpy array, accessible event after event (time series)
    Filters out everything else, like participants pushing buttons 
    """
    #f = scipy.io.loadmat(abs_dir_path)
    #f = h5py.File(abs_dir_path)

    if h5py.is_hdf5(abs_dir_path):
        f = h5py.File(abs_dir_path, 'r')
    else:
        f = scipy.io.loadmat(abs_dir_path)
        
    sEEG = f['sEEG']
    df = pd.DataFrame(sEEG[0])
    events = df['event'][0][0] # dereferenced to obtain the fixation, saccade, blinks, ... 
    data = df['data'][0].T # transpose to access time series 
    #print("Events shape: {}".format(events.shape))

    return events, data # access the i-th event via events[i]

def label_data(verbose=True):

    # Loop over all directories and extract and concat the events from all people
    X = np.array([])
    y = np.array([])

    for subdir, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            # Get the correct path to the current file
            path = os.path.join(subdir, file)
            events, data = load_sEEG(path) # access event i via events[i]

            for i in tqdm(range(len(events)), f"going over event {i}"):
                
                # Read the current event
                event = events[i]
                event_name = event[0][0] # dereference the event name, e.g. 'L_saccade' 
                event_start_time = int(event[1])
                event_end_time = int(event[4])
                event_eeg = np.array(data[event_start_time:event_end_time])

                print(f"event name {event_name}")
                print(f"event shape {event_eeg.shape}")

                # append to dataset
                X = np.concatenate((X, event_eeg))
                
    
    #X = X[:,:,:129]  # Cut off the last 4 columns (time, x, y, pupil size)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    np.savez(SAVE_DIR + DATASET + ".npz", EEG=X, labels=y)
    return X, y

label_data()

