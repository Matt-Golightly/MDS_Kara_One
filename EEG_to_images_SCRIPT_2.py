'''
This script expands on EEG_to_Images_SCRIPT_1.py, and generates summation and difference
Grammian Angular Field images for each epoch in dataframes for raw, filtered, and ICA
component dataframes (outputs from EEG_to_Images_SCRIPT_1.py).

EEG to Images: (EEG_to_Images_SCRIPT_2.py)
- Conversion via GASF and GADF
- Map to 3 channel (colour map)
- Save the images
'''


import numpy as np
import MDS_SCRIPTS as mds
import mne
from mne.preprocessing import ICA
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import os
from mpl_toolkits.axes_grid1 import ImageGrid
import copy
import scipy
from pyts.image import GramianAngularField
from sklearn.pipeline import make_pipeline

def main(raw=True, filt=False, IC_A=False, beta=False):
    PATH_TO_DATA = "...\\KARA_ONE_Data\\ImaginedSpeechData\\"   #Set as appropriate
    #SUBJECTS = ['P02']
    SUBJECTS = ['MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02']

    #Suppress plotting of images
    plt.ioff()

    for subject in SUBJECTS:
        print("Working on Subject: " + subject)

        # Make directories to store model results if not exit
        paths = [PATH_TO_DATA + subject + '\\GAM\\GASF_Images',
                 PATH_TO_DATA + subject + '\\GAM\\GADF_Images',
                 PATH_TO_DATA + subject + '\\GAM\\GASF_Images\\DTCWT',
                 PATH_TO_DATA + subject + '\\GAM\\GADF_Images\\DTCWT',
                 PATH_TO_DATA + subject + '\\GAM\\GASF_Images\\RAW',
                 PATH_TO_DATA + subject + '\\GAM\\GADF_Images\\RAW',
                 PATH_TO_DATA + subject + '\\GAM\\GASF_Images\\ICA',
                 PATH_TO_DATA + subject + '\\GAM\\GADF_Images\\ICA',
                 PATH_TO_DATA + subject + '\\GAM\\GASF_Images\\FILTERED',
                 PATH_TO_DATA + subject + '\\GAM\\GADF_Images\\FILTERED']

        for path in paths:
            if not os.path.exists(path):
                print("Creating required folders")
                os.mkdir(path)

        os.chdir(PATH_TO_DATA + subject + '\\GAM')

        if raw:
            print("Computing RAW images")
            #Create Raw GAF
            df = pd.read_pickle('df_epochs_raw.pkl')
            folder = 'RAW'

        if filt or beta:
            print("Computing Filtered images")
            #Create Filtered GAF
            df = pd.read_pickle('df_epochs_filtered.pkl')
            folder = 'FILTERED'

        if IC_A:
            print("Computing filtered and ICA reduced images")
            #Create Filtered + ICA GAF
            df = pd.read_pickle('df_epochs_ica.pkl')
            folder = 'ICA'

        n_epochs = max(df['epoch']) + 1

        if beta:
            print("Computing images from DTCWT extracted beta band")
            '''Note: this uses the absolute values of the extracted highpass
            complex numbers for the beta band. Resulting in a single image representing
            the entire epoch, instead of 1 image per channel.'''
            import math
            import dtcwt
            from dtcwt import compat as cp
            for e in range(0, n_epochs):
                df_trial = df[df['epoch'] == e].drop(['time', 'condition', 'epoch'], axis=1)[10:5000]

                #Need an even number of rows for DTCWT with dtcwt package
                rows = len(df_trial)
                if (rows % 2 == 1):
                    rows = rows - 1 #If odd, subtract one

                # Performs a 5-level transform on the real image X using the 13,19-tap
                # filters for level 1 and the Q-shift 14-tap filters for levels >= 2.
                Yl, Yh, Yscale = cp.dtwavexfm2(df_trial[:rows], nlevels=5, biort='near_sym_b',
                                               qshift='qshift_b', include_scale=True)

                '''
                Use this block to reconstruct the original signal using only 2nd Order,
                or reformat as desired - reconstructing can de-noise the data.
                gmask = np.asarray([[0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]])

                Z = cp.dtwaveifm(Yl, Yh, 'near_sym_b', 'qshift_b', gain_mask=gmask)

                gasf = GramianAngularField(image_size=len(df_trial), method='summation')
                gadf = GramianAngularField(image_size=len(df_trial), method='difference')

                X_gasf = gasf.fit_transform(Z.transpose())
                X_gadf = gadf.fit_transform(Z.transpose())
                '''

                abss = []
                for a in range(0, Yh[2].shape[0]):
                    abss.append(abs(Yh[1][a, 0, 5]))

                plt.plot(abss)
                plt.title('{} DTCWT Deconstruction'.format(subject))
                plt.savefig('dtcwt_decomposition.png')
                plt.clf()

                Z = cp.dtwaveifm(Yl, Yh, 'near_sym_b', 'qshift_b', gain_mask=None)
                plt.figure()
                plt.plot(Z)
                plt.title('{} DTCWT Reconstruction'.format(subject))
                plt.clf()

                abss = np.asarray(abss)
                abss = abss.reshape(-1, 1)

                #abss.shape[0] #full size for images

                gasf = GramianAngularField(image_size=224, method='summation')
                gadf = GramianAngularField(image_size=224, method='difference')

                X_gasf = gasf.fit_transform(abss.transpose())
                X_gadf = gadf.fit_transform(abss.transpose())



                i = 0
                for gaf in X_gasf:
                    fpath = PATH_TO_DATA + subject + '\\GAM\\GASF_Images\\DTCWT\\gasf_trial_' + str(e) + '.png'
                    plt.imsave(fpath, gaf, cmap='rainbow', origin='lower')
                    i += 1

                i = 0
                for gad in X_gadf:
                    fpath = PATH_TO_DATA + subject + '\\GAM\\GADF_Images\\DTCWT\\gadf_trial_' + str(e) + '.png'
                    plt.imsave(fpath, gad, cmap='rainbow', origin='lower')
                    i += 1
                print("Trial {} of {} complete.".format(e, n_epochs))

        else:
            for e in range(0, n_epochs):
                df_trial = df[df['epoch'] == e].drop(['time', 'condition', 'epoch'], axis=1)[10:5000]

                gasf = GramianAngularField(image_size=224, method='summation')
                gadf = GramianAngularField(image_size=224, method='difference')

                X_gasf = gasf.fit_transform(df_trial.transpose())
                X_gadf = gadf.fit_transform(df_trial.transpose())

                fpath = [PATH_TO_DATA + subject + '\\GAM\\GASF_Images\\' + folder + '\\trial_' + str(e),
                         PATH_TO_DATA + subject + '\\GAM\\GADF_Images\\' + folder + '\\trial_' + str(e)]

                for path in fpath:
                    if not os.path.exists(path):
                        os.mkdir(path)

                i = 0
                for gaf in X_gasf:
                    fname = fpath[0] + '\\gasf_trial_' + str(e) + '_chan_' + str(i) + '.png'
                    plt.imsave(fname, gaf, cmap='rainbow', origin='lower')
                    i += 1

                i = 0
                for gad in X_gadf:
                    fname = fpath[1] + '\\gadf_trial_' + str(e) + '_chan_' + str(i) + '.png'
                    plt.imsave(fname, gad, cmap='rainbow', origin='lower')
                    i += 1

                print("Trial {} of {} complete.".format(e, n_epochs))

        print("Subject finished!")

if __name__ == "__main__":
    main(raw=False, beta=True)