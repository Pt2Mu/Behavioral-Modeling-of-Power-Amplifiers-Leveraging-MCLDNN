import numpy as np
from scipy.io import loadmat

def load_and_prepare_data(Test_ID):
    folder_name = './Datasets/'
    dataset_train = 'DataTrain'
    dataset_test_list = ['DataTrain','DataTest1','DataTest2','DataTest3','DataTest4']
    dataset_test = dataset_test_list[Test_ID]
    data_train = loadmat(folder_name+ dataset_train +'.mat')
    data_test = loadmat(folder_name+ dataset_test +'.mat')

    inpPA_tr = data_train['in']
    outPA_tr = data_train['out']
    inpPA_te = data_test['in']
    outPA_te = data_test['out']

    normalizing_factor = [abs(1690.16376092 - 11735.22170986j)]

    inpPA_train = inpPA_tr/normalizing_factor
    outPA_train = outPA_tr/normalizing_factor

    inpPA_test = inpPA_te/normalizing_factor
    outPA_test = outPA_te/normalizing_factor

    paInputTrain = inpPA_train
    paOutputTrain = outPA_train
    paInputTest = inpPA_test
    paOutputTest = outPA_test


    return paInputTrain, paOutputTrain, paInputTest, paOutputTest
