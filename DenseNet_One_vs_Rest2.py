'''
This script runs a OvR classification for individual subjects, for the desired data set, image encoding,
 and target word.

 Run_DNet.py is the driver script.
'''

import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import csv

import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import gc
import os

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import applications
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Metric
from tensorflow.keras.metrics import Precision, Recall

from tensorflow.keras import applications
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers


def main(gaf, word, method, sub):

    PATH_TO_DATA = "PATH TO DATA"       #This should be the path to the root folder containing subject sub-folders with their Kara One data

    subject = sub

    sum_diff = gaf  # GADF_Images or GASF_Images

    im_type = method  # type of image method, ['DTWCT', 'FILTERED', 'RAW', 'ICA']

    target = word  # Word to target for classification

    # initilise subjects's instances.
    print("Working on Subject: " + subject)
    print("Setting up dataframes")

    """ Setup target and EEG dataframes"""

    df = pd.read_csv(PATH_TO_DATA + subject + "\\kinect_data\\labels.txt", delimiter="\n", header=None)

    df.columns = ['prompt']
    path = PATH_TO_DATA + "ImaginedSpeechData\\" + subject + "\\GAM\\" + sum_diff + "_Images\\" + im_type

    '''For DTCWT images, a single image represents all channels, requiring no manipulation
        All other image types need to be vertically stacked and then resized to an appropriate size (depending on the 
        gpu being used and it's memory size).'''
    if im_type == 'DTCWT':
        images = []
        for file in os.listdir(path):
            im = plt.imread(path + "\\" + file)
            images.append(im[:, :, :3])

        df['images'] = images
        del images, im
    else:
        from PIL import Image
        trial_ims = []
        for folder in os.listdir(path):
            images = []
            for file in os.listdir(path + '\\' + folder):
                im = Image.open(path + "\\" + folder + '\\' + file)
                im = np.asarray(im)
                images.append(im[:, :, :3])
            trial_ims.append(np.vstack((i for i in images)))
        del images, im
        images = []
        for img in trial_ims:
            im = Image.fromarray(img)
            im = im.resize((224, 1736))
            im = np.asarray(im)
            images.append(im)
        df['images'] = images
        del trial_ims, images, im

    #Sanity check of image shape
    print("image shape is: {}".format(df['images'][0].shape))

    words = ['gnaw', 'knew', 'pat', 'pot']

    #filter for desired words
    df = df.loc[df['prompt'].isin(words)]

    #Encode as 0/1 for else/target
    df['prompt'] = pd.Series(np.where(df['prompt'] == target, 1, 0), df.index)

    print("Train / Test splitting data")
    #Split into target word and 'else' dataframes
    target_df = df[df['prompt'] == 1]
    comp_df = df[df['prompt'] == 0]
    del df

    # Stratified train test splits, for target word and then the complement words
    # splitting them up into different sets allows for manipulation, like doubling of the target class instances for training
    # to handle imbalances.
    target_train_x, target_test_x, target_train_y, target_test_y = train_test_split(target_df['images'],
                                                                                    target_df['prompt'],
                                                                                    test_size=0.2, random_state=9)

    comp_train_x, comp_test_x, comp_train_y, comp_test_y = train_test_split(comp_df['images'], comp_df['prompt'],
                                                                            test_size=0.2, random_state=9)

    #double the target training instance to rebalance data set
    target_train_x = target_train_x.append(target_train_x.copy(), ignore_index=True)
    target_train_y = target_train_y.append(target_train_y.copy(), ignore_index=True)

    train_x = target_train_x.append(comp_train_x, ignore_index=True)
    train_y = target_train_y.append(comp_train_y, ignore_index=True)

    test_x = target_test_x.append(comp_test_x, ignore_index=True)
    test_y = target_test_y.append(comp_test_y, ignore_index=True)

    #Can comprise a val set of different sizes to aid in model evaluation during trianing (can be used for model selection)
    val_x = target_test_x.append(comp_test_x.sample(frac=1).reset_index(drop=True))
    val_y = target_test_y.append(comp_test_y.sample(frac=1).reset_index(drop=True))

    # train_x = target_train_x
    # train_y = target_train_y
    # test_x = target_test_x
    # test_y = target_test_y

    del target_train_x, target_test_x, target_train_y, target_test_y, comp_train_x
    del comp_test_x, comp_train_y, comp_test_y

    #set train/test sets, randomly shuffled
    train_x = train_x.sample(frac=1).reset_index(drop=True)
    train_y = train_y.sample(frac=1).reset_index(drop=True)
    test_x = test_x.sample(frac=1).reset_index(drop=True)
    test_y = test_y.sample(frac=1).reset_index(drop=True)

    train_x = np.asarray(train_x.tolist())
    test_x = np.asarray(test_x.tolist())
    val_x = np.asarray(val_x.tolist())

    #for multiclass
    # train_y = pd.get_dummies(train_y)
    # test_y = pd.get_dummies(test_y)

    # Make directories to store model results if not exit
    save_path = PATH_TO_DATA + "ImaginedSpeechData\\" + subject + '\\DenseNet'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = PATH_TO_DATA + "ImaginedSpeechData\\" + subject + '\\DenseNet\\' + im_type

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = PATH_TO_DATA + "ImaginedSpeechData\\" + subject + '\\DenseNet\\' + im_type + '\\' + sum_diff

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = PATH_TO_DATA + "ImaginedSpeechData\\" + subject + '\\DenseNet\\' + im_type + '\\' + sum_diff + "\\" + target + "_model"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    #Set model save path
    save_model = save_path + '\\' + target + '_model'

    # Build and fit model
    print("Building DenseNet Model")
    print("Building DenseNet 121 Model")
    # K.set_image_data_format('channels_first')
    INPUT_SHAPE = train_x.shape[1:]  # used to define the input size to the model
    n_output_units = 1

    if im_type == 'DTWCT':
        batch = 12
    else:
        batch = 12

    base_model = applications.densenet.DenseNet121(weights='imagenet', include_top=False,
                                                   input_shape=INPUT_SHAPE)

    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    #x = BatchNormalization()(x)                                        #These layers are left over from some exploratory trials
    #x = Dropout(0.2)(x)
    #x = Dense(1024, activation=LeakyReLU())(x)
    #x = Dense(512, activation=LeakyReLU())(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)

    output = Dense(n_output_units, activation='sigmoid')(x)  # FC-layer

    model = Model(inputs=base_model.input, outputs=output)

    # for layer in model.layers[:-300]:
    #    layer.trainable = False

    # for layer in model.layers[-300:]:
    #    layer.trainable = True

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse',
                                                                         tf.keras.metrics.Precision(),
                                                                         tf.keras.metrics.Recall()])
    model.summary()

    print("Fitting Model")
    anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-4)
    #chk = ModelCheckpoint(save_model, monitor='val_loss', save_best_only=True, mode='min',  save_freq='epoch', verbose=1)
    #chk1 = ModelCheckpoint(save_model, monitor='val_precision', save_best_only=True, mode='max',  save_freq='epoch', verbose=1)
    #chk2 = ModelCheckpoint(save_model, monitor='val_recall', save_best_only=True, mode='max',  save_freq='epoch', verbose=1)
    chk3 = ModelCheckpoint(save_model, monitor='loss', save_best_only=True, mode='min', save_freq='epoch', period=50, verbose=1)

    os.chdir(save_path)

    es = EarlyStopping(monitor='loss', min_delta=0.0001, verbose=1, patience=100, mode='auto')
    history = model.fit(train_x, train_y,
                        epochs=1000,
                        batch_size=batch,
                        verbose=1,
                        callbacks=[anne, chk3],
                        validation_data=(val_x, val_y))

    print("Model successfully trained!")

    # Store a printout of the model summary
    model_sum = save_path + '\\DenseNetOVR_summary.png'
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file=model_sum, show_shapes=True, show_layer_names=True)

    # Plots of model training
    img_loc = save_path + '\\Training_Accuracy_vs_Validation_Accuracy.png'
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['train', 'validation'])
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.savefig(img_loc)
    plt.clf()

    img_loc = save_path + '\\Training_Loss_vs_Validation_Loss.png'
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'])
    plt.title("Training Loss vs Validation Loss")
    plt.savefig(img_loc)
    plt.clf()

    img_loc = save_path + '\\Training_vs_Validation_Precision.png'
    plt.plot(history.history['precision'], label='train')
    plt.plot(history.history['val_precision'], label='validation')
    plt.xlabel("Num of Epochs")
    plt.ylabel("Precision")
    plt.legend(['train', 'validation'])
    plt.title("Training vs Validation Precision")
    plt.savefig(img_loc)
    plt.clf()

    img_loc = save_path + '\\Training_MSE_vs_Validation_MSE.png'
    plt.plot(history.history['mse'], label='train')
    plt.plot(history.history['val_mse'], label='validation')
    plt.xlabel("Num of Epochs")
    plt.ylabel("MSE")
    plt.legend(['train', 'validation'])
    plt.title("Training MSE vs Validation MSE")
    plt.savefig(img_loc)
    plt.clf()

    print("Performing model evaluation...")
    dependencies = {'accuracy': keras.metrics.Accuracy,
                    'loss': keras.losses.binary_crossentropy,
                    }

    del model, history
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    model = load_model(save_model)  # , custom_objects=dependencies, compile=True)

    #Model predictions
    pred_classes = model.predict(test_x, batch_size=batch)
    pred_classes = np.where(pred_classes > 0.5, 1, 0)

    pred_classes = pd.DataFrame(pred_classes)

    accuracy_score(test_y, pred_classes)

    from sklearn.metrics import confusion_matrix

    my_mat = confusion_matrix(test_y, pred_classes, labels=[0,1])
    print(my_mat)

    my_mat = pd.DataFrame(my_mat, index=[i for i in ['else', target]],
                          columns=[i for i in ['else', target]])

    my_mat.to_pickle(save_path + '\\conf_mat.pkl')

    sn.heatmap(my_mat, annot=True)
    img_loc = save_path + '\\conf_mat.png'
    plt.savefig(img_loc)
    plt.clf()

    print(sklearn.metrics.classification_report(test_y, pred_classes))

    report = sklearn.metrics.classification_report(test_y, pred_classes, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_pickle(save_path + '\\classification_report')

    print("Model evaluation complete, results stored to subject folder, resetting Keras.")

    del model
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()  # TF graph isn't same as Keras graph
    tf.keras.backend.clear_session()


    # Compute subject model accuracies in dict, save to csv

    dense_acc = {}

    file_path = PATH_TO_DATA + "ImaginedSpeechData\\" + subject + '\\DenseNet\\' + im_type + '\\' + sum_diff + '\\' + \
                target + '_model\\' + 'conf_mat.pkl'

    df = pd.read_pickle(file_path)

    # Get the HDF5 group

    acc = {}
    i = 0
    c = 0
    t = 0
    for block in df.values:
        acc[df.columns[i]] = block[i] / sum(block)
        c = c + block[i]
        t = t + sum(block)
        i += 1

    avrg_acc = c/t

    #dense_acc[subject] = np.array(list(acc.values())).mean()
    dense_acc['else'] = list(acc.values())[0]
    dense_acc[target] = list(acc.values())[1]
    dense_acc['total accuracy'] = avrg_acc

    print("DenseNet_L2 OvR accuracies:")
    for k, v in dense_acc.items():
        print(k, v)

    print("DenseNet Subject Accuracy:")
    #print(np.array(list(dense_acc.values())).mean())
    print(avrg_acc)

    # Save accuracies to csv files
    (pd.DataFrame.from_dict(data=dense_acc, orient='index').to_csv(
        "G:\\UWA_MDS\\2021SEM1\\Research_Project\\KARA_ONE_Data\\ImaginedSpeechData\\" + subject + "\\DenseNet\\" + im_type +
        '\\' + sum_diff + '\\' + target + "_DenseNet_acc.csv", header=False))

if __name__ == "__main__":
    print("subject: {}".format(sys.argv[4]))
    print("method: {}".format(sys.argv[3]))
    print("word: {}".format(sys.argv[2]))
    print("im_type: {}".format(sys.argv[1]))
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
