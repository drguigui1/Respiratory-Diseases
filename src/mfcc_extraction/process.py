# -*-coding: utf-8 -*-

'''
Process the dataset
Extract mfcc
Create file of training labeled data
Create file of testing labeled data
'''

import os
import pandas as pd
import numpy as np
import librosa as lr

from glob import glob


def build_dataset(data_path, label_path_file, n_mfcc):
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

    files = glob(data_path + "*.wav")

    X = []
    y = np.zeros(len(files))

    for idx, file_path in enumerate(files):
        # get the signal
        signal, rate = lr.load(file_path)

        # get the label
        file_name = file_path.split('/')[-1]
        numb = file_name.split('_')[0]

        # mfcc extraction
        mfccs = lr.feature.mfcc(y=signal, sr=rate, n_mfcc=n_mfcc)
        if idx % 100 == 0:
            print("mfcc extracted for {}".format(idx))

        # get associated label of value 'numb'
        label_str = np.array(labels.loc[labels['Number'] == int(numb)])[0, 1]

        if (label_str == 'Asthma' or label_str == 'LRTI'):
            continue

        y[idx] = np.where(diseases == label_str)[0]
        X.append(mfccs)
    return X, y, diseases

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

    data_path = '../../audio_files/'
    label_path_file = '../../patient_diagnosis.csv'

    n_mfcc = 50

    ## build the dataset
    print("Building the dataset")
    X, y, diseases = build_dataset(data_path, label_path_file, n_mfcc)

    ## create directories
    print()
    print("create directories")
    create_dirs([dataset_dir])

    ## save into files
    print()
    print("save the dataset into files")
    save_dataset(dataset_dir, X, y)

    ## save the labels associated to each disease
    save_labels_diseases(dataset_dir + 'diseases', diseases)
