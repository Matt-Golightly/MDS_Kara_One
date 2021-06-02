'''
Simple vanilla LSTM multiclass classifier for raw EEG data
'''

import scipy.io as spio
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import gc
import h5py


def loadmat(filename):
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _has_struct(elem):
        """Determine if elem is an array and if any array item is a struct"""
        return isinstance(elem, np.ndarray) and any(isinstance(
                    e, spio.matlab.mio5_params.mat_struct) for e in elem)

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


"""Helper function to truncate dataframes to a specified shape - usefull to reduce all EEG trials to the same number
   of time stamps.
"""
def truncate(arr, shape):
    desired_size_factor = np.prod([n for n in shape if n != -1])
    if -1 in shape:  # implicit array size
        desired_size = arr.size // desired_size_factor * desired_size_factor
    else:
        desired_size = desired_size_factor
    return arr.flat[:desired_size].reshape(shape)

def main():
    PATH = "G:\\UWA_MDS\\2021SEM1\\Research_Project\\KARA_ONE_Data\\ImaginedSpeechData\\"
    subjects = ['MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02']

    for subject in subjects:
        print("Working on Subject: " + subject)

        print("Loading .set data")
        """ Load EEG data with loadmat() function"""
        SubjectData = loadmat(PATH + subject + '\\EEG_data.mat')

        print("Setting up dataframes")
        """ Setup target and EEG dataframes"""
        targets = pd.DataFrame(SubjectData['EEG_Data']['prompts'])
        targets.columns = ['prompt']

        sequences = pd.DataFrame(SubjectData['EEG_Data']['activeEEG'])
        sequences.columns = ['trials']

        EEG = pd.concat([sequences.reset_index(drop=True),targets.reset_index(drop=True)], axis=1)

        words = ['gnaw', 'pat', 'knew', 'pot']

        EEG = EEG.loc[EEG['prompt'].isin(words)]

        EEG = EEG.reset_index(drop=True)


        sequences = pd.DataFrame(EEG['trials'])
        targets = pd.DataFrame(EEG['prompt'])


        seq = np.asarray(sequences['trials'])
        for i in range(0,len(seq)):
            seq[i] = seq[i].transpose()
            i=i+1


        sequences['trials'] = seq

        print("Train / Test splitting data")
        #Stratified train test splits
        train_x, test_x, train_y, test_y = train_test_split(sequences, targets, stratify=targets, test_size=0.2, random_state=9)

        #Encode target prompts to 0/1
        train_y= pd.get_dummies(train_y['prompt'])
        test_y= pd.get_dummies(test_y['prompt'])

        #need train_x and test_x as arrays in order to truncate them down to the smallest time trial
        train_x = np.asarray(train_x['trials'])
        test_x = np.asarray(test_x['trials'])

        #find minimum length of all the trials present in both test and train trials
        min_ln = min(min(i.shape for i in train_x)[0], min(i.shape for i in test_x)[0])

        #reduce all trials down to common length set by min_ln
        for arr in [train_x, test_x]:
            i=0
            for trial in arr:
                arr[i] = truncate(trial, (min_ln, 62))
                i = i+1

        #for LSTM model we need data in a 3D array, (,
        train_x = np.rollaxis(np.dstack(train_x), -1)
        test_x = np.rollaxis(np.dstack(test_x), -1)


        #Make directories to store model results if not exit
        save_path = PATH + subject + '\\lstm_model'
        import os
        from os import path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_model = save_path + '\\lstm_vanilla_model'

        # Build and fit model
        from keras.callbacks import EarlyStopping
        with tf.device('/cpu:0'):
            print("Building LSTM")
            model = Sequential()
            model.add(LSTM(256, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))
            model.add(LSTM(256))
            model.add(Dropout(0.5))
            model.add(Dense(100, activation='relu'))
            model.add(Dense(4, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            print("Fitting Model")
            chk = ModelCheckpoint(save_model, monitor='loss', save_best_only=True, mode='min', verbose=1)
            es = EarlyStopping(monitor='loss', min_delta=0.01, verbose=1, patience=5, mode='auto')
            model.fit(train_x, train_y, epochs=30, batch_size=train_x.shape[0], verbose=1, callbacks=[chk, es])
            #history = model.fit(train_x, train_y, epochs=50, batch_size=train_x.shape[0], verbose=1, callbacks=[chk, es])
        print("Model successfully trained!")

        # Store a printout of the model summary
        model_sum = save_path + '\\lstm_summary.png'
        from keras.utils import plot_model
        plot_model(model, to_file=model_sum, show_shapes=True, show_layer_names=True)



        #Plots of model training
        #img_loc = PATH + subject + '\\lstm_training_loss.png'
        #plt.plot(history.history['loss'], label='train')
        #plt.plot(history.history['val_loss'], label='test')
        #plt.legend()
        #plt.savefig(img_loc)

        print("Performing model evaluation...")
        model2 = load_model(save_model)

        test_preds = model2.predict_classes(test_x)

        test_preds = pd.DataFrame(test_preds)
        test_preds.columns = ['prompts']

        test_preds = test_preds.replace({0 : 'gnaw', 1 : 'knew', 2 : 'pat', 3 : 'pot'})


        new_df = test_y.idxmax(axis=1)

        accuracy_score(new_df, test_preds['prompts'])

        from sklearn.metrics import confusion_matrix

        my_mat = confusion_matrix(new_df, test_preds['prompts'])

        my_mat = pd.DataFrame(my_mat, index=[i for i in ['gnaw', 'knew', 'pat', 'pot']],
                              columns=[i for i in ['gnaw', 'knew', 'pat', 'pot']])

        hdf_loc = PATH + subject + '\\lstm_conf_mat.h5'
        my_mat.to_hdf(hdf_loc, key='conf_mat', mode='w')


        import seaborn as sn
        sn.heatmap(my_mat, annot=True)
        img_loc = PATH + subject + '\\lstm_conf_mat.png'
        plt.savefig(img_loc)
        plt.clf()

        print("Model evaluation complete, results stored to subject folder, resetting Keras.")


        del model
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()  # TF graph isn't same as Keras graph

    #Compute subject model accuracies in dict, save to csv
    matrix = "\\lstm_conf_mat.h5"

    lstm_acc = {}
    for subject in subjects:
        file_path = PATH + subject + matrix

        with h5py.File(file_path, 'r') as f:

            # Get the HDF5 group
            group = f['conf_mat']

            acc = {}
            i = 0
            for block in group['block0_values'].value:
                acc[group['block0_items'].value[i]] = block[i] / sum(block)
                i += 1

            lstm_acc[subject] = np.array(list(acc.values())).mean()
            del group

    print("LSTM subject accuracies:")
    for k, v in lstm_acc.items():
        print(k, v)

    print("LSTM Cross Subject Accuracy:")
    np.array(list(lstm_acc.values())).mean()

    #Save lstm_acc to csv files
    (pd.DataFrame.from_dict(data=lstm_acc, orient='index').to_csv("G:\\UWA_MDS\\2021SEM1\\Research_Project\\KARA_ONE_Data\\ImaginedSpeechData\\lstm_acc.csv", header=False))

if __name__ == "__main__":
    main()
