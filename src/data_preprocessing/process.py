# -*- coding: utf-8 -*-

'''
Split dataset into training and testing
Data augmenting
Extract mfcc
'''

import numpy as np
import pandas as pd
import librosa as lr
import os
from glob import glob


def noise_injection(x, noise_factor):
    '''
    Noise injection in the data
    x: vector (signal of one audio)
    noise_factor: factor to know string we apply the noise
    '''
    noise = np.random.randn(len(x))
    new_x = x + noise_factor * noise
    new_x = new_x.astype(type(x[0]))
    return new_x

def shifting_time(x, rate, shift_max, shift_dir='left'):
    '''
    Shifting of the data according to the time
    '''
    shift_val = np.random.randint(rate * shift_max)
    if (shift_dir == 'right'):
        shift_val = -shift_val
    # else we assume that its 'left'
    new_x = np.roll(x, shift_val)
    return new_x


def pitch_changing(x, rate):
    '''
    Changing the pitch of the audio
    '''
    return lr.effects.pitch_shift(x, rate, 3)

## TO CHANGE
def data_augmenting(x, label, rate, n_mfcc=30):
    '''
    Augment the number of data
    '''
    X_new = []
    y_new = []

    # pitch data augmenting
    pitched_data = pitch_changing(x, rate)
    X_new.append(lr.feature.mfcc(y=pitched_data , sr=rate, n_mfcc=n_mfcc).T)

    # shifting time data augmenting
    l = np.random.randint(low=5, high=100, size=4)
    for rd in l:
        x_shifted_l = shifting_time(x, rate, rd, shift_dir='left')
        x_shifted_noised = noise_injection(x_shifted_l, 0.0003)
        X_new.append(lr.feature.mfcc(y=x_shifted_noised, sr=rate, n_mfcc=n_mfcc).T)
        X_new.append(lr.feature.mfcc(y=x_shifted_l, sr=rate, n_mfcc=n_mfcc).T)

        x_shifted_r = shifting_time(x, rate, rd, shift_dir='right')
        x_shifted_noised = noise_injection(x_shifted_r, 0.0003)
        X_new.append(lr.feature.mfcc(y=x_shifted_noised, sr=rate, n_mfcc=n_mfcc).T)
        X_new.append(lr.feature.mfcc(y=x_shifted_r, sr=rate, n_mfcc=n_mfcc).T)

    # noise injection data augmenting
    x_noised_1 = noise_injection(x, 0.0003)
    x_noised_2 = noise_injection(x, 0.0006)
    X_new.append(lr.feature.mfcc(y=x_noised_1, sr=rate, n_mfcc=n_mfcc).T)
    X_new.append(lr.feature.mfcc(y=x_noised_2, sr=rate, n_mfcc=n_mfcc).T)

    # append real data
    X_new.append(lr.feature.mfcc(y=x, sr=rate, n_mfcc=n_mfcc).T)
    for i in range(20):
        y_new.append(label)

    return X_new, np.array(y_new)

def build_dataset(data_path, label_path_file, nb_list, n_mfcc=30):
    '''
    data_path: path to the directory with the .wav files
    label_path_file: path to file that contains the labels associated with numbers (csv)
    n_mfcc: number of mfcc to extract
    '''
    ## extraction of labels
    labels = pd.read_csv(label_path_file, header=None)
    labels.rename(columns={0: 'Number', 1: 'Label'}, inplace=True)

    # ['URTI', 'Healthy' ...]
    diseases = labels['Label'].unique()
    diseases = np.delete(diseases, np.argwhere(diseases == 'Asthma'))
    diseases = np.delete(diseases, np.argwhere(diseases == 'LRTI'))
    diseases = np.delete(diseases, np.argwhere(diseases == 'Bronchiectasis'))
    diseases = np.delete(diseases, np.argwhere(diseases == 'Bronchiolitis'))

    files = glob(data_path + "*.wav")

    nb_data_test, nb_data = sum(nb_list), len(files)
    X_train, X_test = [], []
    y_train = []
    y_test = np.zeros(nb_data_test)
    idx_te = 0

    for idx, file_path in enumerate(files):
        # get the label
        file_name = file_path.split('/')[-1]
        numb = file_name.split('_')[0]

        # get associated label of value 'numb'
        label_str = np.array(labels.loc[labels['Number'] == int(numb)])[0, 1]

        if (label_str == 'Asthma' or label_str == 'LRTI' \
            or label_str == 'Bronchiectasis' or label_str == 'Bronchiolitis'):
            continue

        # get the signal
        signal, rate = lr.load(file_path)

        label = int(np.where(diseases == label_str)[0])

        ## get data for training
        if nb_list[label] == 0: ## already all test data got
            y = np.array([label])
            ## augment data
            if (label != 2): # no need to augment COPD class
                x, y = data_augmenting(signal, label, rate, n_mfcc=n_mfcc)
            else:
                x = [lr.feature.mfcc(y=signal, sr=rate, n_mfcc=n_mfcc).T]
            X_train += x
            y_train = np.concatenate((y_train, y))
        else: # get data for testing
            y_test[idx_te] = label
            idx_te += 1
            nb_list[label] -= 1
            X_test.append(lr.feature.mfcc(y=signal, sr=rate, n_mfcc=n_mfcc).T)
        print("data nb {}".format(idx))
    return X_train, y_train, X_test, y_test, diseases

def create_dirs(list_dir):
    '''
    Create directories
    list_dir: list of directories to create
    '''
    for d_name in list_dir:
        os.mkdir(d_name)

def mat_to_str(mat):
    '''
    numpy array with the mfcc
    '''
    mat = mat.flatten()
    m = mat.shape[0]
    s = ""
    for i in range(m):
        s += str(mat[i])
        if (i != m - 1):
            s += ','
    return s

def save_dataset(path, X, y):
    '''
    path: path to save the data
    X: data to save
    y: labels to save
    '''
    data_file = path + "data"
    label_file = path + "labels"
    f_data = open(data_file, "a+")
    f_labels = open(label_file, "a+")

    for idx, vec in enumerate(X):
        vec_str = mat_to_str(vec)
        f_data.write(vec_str)
        f_data.write('\n')

        label_str = str(int(y[idx]))
        if (idx != len(X) - 1):
            label_str += ','
        f_labels.write(label_str)

    f_data.close()
    f_labels.close()

def save_labels_diseases(path, diseases):
    '''
    Save in a file labels associated to the disease
    path: where to save
    disease: list of the diseases
    '''
    f = open(path, "a+")
    f.write("Label      Disease\n")
    for idx, elm in enumerate(diseases):
        f.write(str(idx) + "      ->  " + elm + "\n")
    f.close()

if __name__ == "__main__":
    dataset_dir = '../../respiratory-dataset/'
    dataset_dir_te = '../../respiratory-dataset/test/'
    dataset_dir_tr = '../../respiratory-dataset/train/'

    data_path = '../../audio_files/'
    label_path_file = '../../patient_diagnosis.csv'

    ## build dataset
    nb_list = [7, 10, 20, 10]
    print("Build the dataset")
    X_train, y_train, X_test, y_test, diseases = build_dataset(data_path, label_path_file, nb_list, n_mfcc=30)

    ## create directories
    print()
    print("create directories")
    create_dirs([dataset_dir, dataset_dir_te, dataset_dir_tr])

    ## save into files
    print()
    print("save the dataset into files")
    save_dataset(dataset_dir_te, X_test, y_test)
    save_dataset(dataset_dir_tr, X_train, y_train)

    ## save the labels associated to each disease
    save_labels_diseases(dataset_dir + 'diseases', diseases)
