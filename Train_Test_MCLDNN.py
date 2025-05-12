import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from scipy.io import loadmat,savemat
from tensorflow.keras.optimizers import Adam
from Model import MCLDNNmodel14 as MCLDNNmodel
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from data_processing import *
from Parameter import *
from load_dataset import *
import random
import tensorflow as tf
from Evaluation import *

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


def train_main():
    paInputTrainNorm, paOutputTrainNorm, _, _ = load_and_prepare_data(0)

    X_train, X1_train, X2_train, X3_train, X4_train, Y_train = process_data(paInputTrainNorm, paOutputTrainNorm)

    model = MCLDNNmodel.MCLmodel()

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mean_squared_error'])

    checkpoint = ModelCheckpoint(filepath=f'model_save/MCL.h5', monitor='loss', mode='min', save_best_only='True', save_weights_only=True,
                                 verbose=1)
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    csv_logger = CSVLogger(f'loss/val_loss_MCL.csv', separator=',', append=False)



    model.fit([X_train, X1_train, X2_train, X3_train, X4_train], Y_train, epochs=ep, batch_size=bs, verbose=2, callbacks=[checkpoint,csv_logger,early_stopping])

def main():
    set_seed(2024)
    train_main()

    model = MCLDNNmodel.MCLmodel()
    model.load_weights(f'model_save/MCL.h5')

    for Test_ID in range(5):
        dataset_list = ['DataTrain', 'DataTest1', 'DataTest2', 'DataTest3', 'DataTest4']
        dataset = dataset_list[Test_ID]

        print(dataset)

        _, _, paInputTestNorm, paOutputTestNorm = load_and_prepare_data(Test_ID)

        X_test, X1_test, X2_test, X3_test, X4_test = process_data_test_model(paInputTestNorm)

        [no_samples, _] = np.shape(paInputTestNorm)

        a = model.predict([X_test, X1_test, X2_test, X3_test, X4_test], verbose=2)

        Y_out = np.zeros((len(a), 1), dtype=np.complex128)

        for i in range(len(a)):
            Y_out[i, 0] = complex(a[i, 0], a[i, 1])

        Y_pred = Y_out * normalizing_factor

        outPA_whole = paOutputTestNorm * normalizing_factor
        inpPA_whole = paInputTestNorm * normalizing_factor

        out_ignore_first_mem = outPA_whole[seqLength - 1:no_samples]  # First M samples are neglected
        inp_ignore_first_mem = inpPA_whole[seqLength - 1:no_samples]
        error = Y_pred - out_ignore_first_mem

        NMSE(error, out_ignore_first_mem)
        ACEPR_cal(error, inp_ignore_first_mem)

        if Test_ID==1 or Test_ID==2:
            piece1_end = 2000
            piece1_pred = Y_pred[0:piece1_end -seqLength+1]
            piece1_out = outPA_whole[seqLength-1:piece1_end]
            piece1_in = inpPA_whole[seqLength-1:piece1_end]
            error = piece1_pred-piece1_out

            print('For piece1')

            NMSE(error, piece1_out)
            ACEPR_cal(error, piece1_in)

            piece2_pred = Y_pred[piece1_end:piece1_end*2-seqLength+1]
            piece2_out = outPA_whole[piece1_end+seqLength-1:piece1_end*2]

            piece2_in = inpPA_whole[piece1_end+seqLength-1:piece1_end*2]
            error = piece2_pred-piece2_out

            print('For piece2')

            NMSE(error, piece2_out)
            ACEPR_cal(error, piece2_in)




if __name__ == '__main__':
    main()