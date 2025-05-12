import numpy as np
from Parameter import *

def process_data(inp_model, out_model):
    [no_samples,_] = np.shape(inp_model)

    inputVEC = np.zeros((no_samples-memLength,max_order+2,seqLength))
    outputVEC = np.zeros((no_samples - memLength, 2))
    absPA = max_order * [1]

    for v in range(max_order):
        absPA[v] = np.power(np.abs(inp_model), (v + 1))

    for i in range(no_samples-memLength):
        inputVEC[i][0] = np.real(inp_model[i:i + seqLength]).squeeze()
        inputVEC[i][1] = np.imag(inp_model[i:i + seqLength]).squeeze()
        inputVEC[i][2] = absPA[0][i:i + seqLength].squeeze()
        inputVEC[i][3] = absPA[1][i:i + seqLength].squeeze()
        outputVEC[i][0] = np.real(out_model[memLength+i])
        outputVEC[i][1] = np.imag(out_model[memLength+i])

    I_VEC = inputVEC[:, 0, :]
    I_VEC = I_VEC.reshape((no_samples-memLength,seqLength,1))
    Q_VEC = inputVEC[:, 1, :]
    Q_VEC = Q_VEC.reshape((no_samples-memLength,seqLength,1))
    absPA_1_VEC = inputVEC[:, 2, :]
    absPA_1_VEC = absPA_1_VEC.reshape((no_samples - memLength, seqLength, 1))
    absPA_2_VEC = inputVEC[:, 3, :]
    absPA_2_VEC = absPA_2_VEC.reshape((no_samples - memLength, seqLength, 1))

    inputVEC = np.array(inputVEC)
    inputVEC = np.expand_dims(inputVEC, axis=-1)

    X = inputVEC
    X1 = I_VEC
    X2 = Q_VEC
    X3 = absPA_1_VEC
    X4 = absPA_2_VEC

    Y = outputVEC

    return X, X1, X2, X3, X4, Y


def process_data_test_model(inp_model):

    [no_samples,_] = np.shape(inp_model)

    inputVEC = np.zeros((no_samples-memLength,max_order+2,seqLength))

    absPA = max_order * [1]

    for v in range(max_order):
        absPA[v] = np.power(np.abs(inp_model), (v + 1))

    for i in range(no_samples-memLength):
        inputVEC[i][0] = np.real(inp_model[i:i + seqLength]).squeeze()
        inputVEC[i][1] = np.imag(inp_model[i:i + seqLength]).squeeze()
        inputVEC[i][2] = absPA[0][i:i + seqLength].squeeze()
        inputVEC[i][3] = absPA[1][i:i + seqLength].squeeze()


    I_VEC = inputVEC[:, 0, :]
    I_VEC = I_VEC.reshape((no_samples-memLength,seqLength,1))
    Q_VEC = inputVEC[:, 1, :]
    Q_VEC = Q_VEC.reshape((no_samples-memLength,seqLength,1))
    absPA_1_VEC = inputVEC[:, 2, :]
    absPA_1_VEC = absPA_1_VEC.reshape((no_samples - memLength, seqLength, 1))
    absPA_2_VEC = inputVEC[:, 3, :]
    absPA_2_VEC = absPA_2_VEC.reshape((no_samples - memLength, seqLength, 1))

    inputVEC = np.array(inputVEC)
    inputVEC = np.expand_dims(inputVEC, axis=-1)

    X = inputVEC
    X1 = I_VEC
    X2 = Q_VEC
    X3 = absPA_1_VEC
    X4 = absPA_2_VEC

    return X, X1, X2, X3, X4



