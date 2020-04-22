# -*-coding: utf-8 -*-

'''
Process the dataset
Extract mfcc
Create file of training labeled data
Create file of testing labeled data
'''

import os
import pandas as pd

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
        pass

    return X, y, diseases

def create_dirs(list_dir):
    '''
    Create directories
    list_dir: list of directories to create
    '''
    for d_name in list_dir:
        os.mkdir(d_name)

def save_dataset(path, X, y):
    '''
    '''
    pass

if __name__ == "__main__":
    dataset_dir = './respiratory-dataset'
    training_dir = './respiratory-dataset/train'
    testing_dir = './respiratory-dataset/test'

    data_train_path = '../audio_path/'
    data_test_path = '../test/'
    label_path_file = '../patient_diagnosis.csv'

    n_mfcc = 50

    ## build the training dataset
    X_train, y_train, diseases = build_dataset(data_train_path, label_path_file, n_mfcc)

    ## build the testing dataset
    X_test, y_test, _ = build_dataset(data_test_path, label_path_file, n_mfcc)

    ## create directories
    create_dir([dataset_dir, training_dir, testing_dir])

    ## save into files
    save_dataset(training_dir, X_train, y_train)
    save_dataset(testing_dir, X_test, y_test)
