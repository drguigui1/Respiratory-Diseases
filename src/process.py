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

        y[idx] = np.where(diseases == label_str)[0]
        X.append(mfccs)
        break
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
    f_data = open(data_file, "w")
    f_labels = open(label_file, "w")

    for idx, vec in enumerate(X):
        vec_str = mat_to_str(vec)
        f_data.write(vec_str)

        label_str = str(y[idx])
        if (idx != len(X) - 1):
            label_str += ','
        f_labels.write(label_str)

    f_data.close()
    f_labels.close()

if __name__ == "__main__":
    dataset_dir = '../respiratory-dataset/'
    training_dir = '../respiratory-dataset/train/'
    testing_dir = '../respiratory-dataset/test/'

    data_train_path = '../audio_files/'
    data_test_path = '../test/'
    label_path_file = '../patient_diagnosis.csv'

    n_mfcc = 50

    ## build the training dataset
    print("Building the dataset")
    X_train, y_train, diseases = build_dataset(data_train_path, label_path_file, n_mfcc)

    ## build the testing dataset
    X_test, y_test, _ = build_dataset(data_test_path, label_path_file, n_mfcc)

    ## create directories
    print("create directories")
    create_dirs([dataset_dir, training_dir, testing_dir])

    ## save into files
    print("save the dataset into files")
    save_dataset(training_dir, X_train, y_train)
    save_dataset(testing_dir, X_test, y_test)
