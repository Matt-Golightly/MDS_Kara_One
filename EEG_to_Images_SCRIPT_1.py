'''
This script runs through EEG preprocessing for subjects of the KARA One dataset, then
 applies a time series to image process via Grammian Angular Field (both summation and
  difference) to produce image plots for each channel of each trial.

The workflow loosely follows Harvard's HAPPE workflow:
Pre-processing:
- Filtering
- Channel Selection
- Electrical Noise Removal (done in initial filtering for the EEG range worked with here)
- Bad channel rejection (already done by KARA One team, i.e. they are not imported when loading in)
- ICA
- ICA component rejection
- Channel interpolation (not performed here)
- Re-referencing (not performed here)
- Beta band extraction (via DTCWT, selection of subbands, optional step) (not performed here)

EEG to Images: (EEG_to_Images_SCRIPT_2.py)
- Conversion via GASF and GADF
- Map to 3 channel (colour map)
- Save the images

The output is a series of images visualising the workflow stages, along with pickled pandas
dataframes of the epoched subject data. This data can then be used in
 EEG_to_Images_SCRIPT_2.py to generate the Grammian Angular Field images.
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

def main():
    PATH_TO_DATA = "PATH TO DATA"       #This should be the path to the root folder containing subject sub-folders with their Kara One data
    SUBJECTS = ['MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02']

    #Suppress plotting of images
    plt.ioff()

    # initilise subjects's instances.
    for subject in SUBJECTS:
        mds.Dataset(subject)

    #Set larger fig size than default
    figure(figsize=(30, 20))

    for subject in mds.Dataset.registry:
        print("Working on Subject: " + subject.name)
        print('Loading data..')
        subject.load_data(PATH_TO_DATA, raw=True)

        print('Data Loaded.')
        print(subject.eeg_data.info)

        subject.eeg_data.load_data()

        # Make directories to store model results if not exit
        save_path = PATH_TO_DATA + "\\ImaginedSpeechData"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path = PATH_TO_DATA + "\\ImaginedSpeechData\\" + subject.name
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path = PATH_TO_DATA + "\\ImaginedSpeechData\\" + subject.name + '\\GAM'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        #Plot the raw eeg data
        plt.close(subject.eeg_data.plot(color='darkblue').savefig(save_path + '\\Raw_EEG'))

        #Plot the raw eeg psd
        plt.close(subject.eeg_data.plot_psd(area_mode='range', tmax=10.0).savefig(save_path + "\\Raw_EEG_PSD"))

        #Plot the raw eeg psd with Fmax of 50Hz
        plt.close(subject.eeg_data.plot_psd(fmax=50).savefig(save_path + "\\Raw_EEG_PSD_f50"))

        #Copy the data sets we need to maintain a raw and filtered copy of the data
        raw = subject.eeg_data.copy()
        filtered = subject.eeg_data.copy()
        ica_data = subject.eeg_data.copy()

        # Bandpass filter between 1Hz and 50Hz (also removes power line noise ~60Hz)
        filtered.filter(None, 45., fir_design='firwin')
        filtered.filter(2., None, fir_design='firwin')
        ica_data.filter(None, 45., fir_design='firwin')
        ica_data.filter(2., None, fir_design='firwin')

        #Plot the filtered eeg data PSD
        plt.close(filtered.plot_psd(area_mode='range', tmax=10.0).savefig(save_path + "\\Filtered_PSD"))

        # Create events matrix required to define epochs in mne.Epochs
        # needs to be of shape(n_epochs,3)
        # array([[epoch_timestamp0, 0, prompt0],
        #       [epoch_timestamp1, 0, prompt1],
        #       ...
        #     ])
        events = copy.deepcopy(subject.epoch_inds['thinking_inds'])
        events = np.reshape(events, (events.shape[1], 1))
        prompts = []
        for event in events:
            prompts.append(event[0][0])

        i = 0
        for prompt in prompts:
            prompt[1] = 0
            prompts[i] = np.append(prompt, np.array(subject.prompts[5][0][i][0]))
            i += 1

        prompts = np.asarray(prompts)

        # All prompts need to be int format
        prompts = np.where(prompts == '/iy/', 0, prompts)
        prompts = np.where(prompts == '/uw/', 1, prompts)
        prompts = np.where(prompts == '/piy/', 2, prompts)
        prompts = np.where(prompts == '/tiy/', 3, prompts)
        prompts = np.where(prompts == '/diy/', 4, prompts)
        prompts = np.where(prompts == '/m/', 5, prompts)
        prompts = np.where(prompts == '/n/', 6, prompts)
        prompts = np.where(prompts == 'pat', 7, prompts)
        prompts = np.where(prompts == 'pot', 8, prompts)
        prompts = np.where(prompts == 'knew', 9, prompts)
        prompts = np.where(prompts == 'gnaw', 10, prompts)

        #sanity check
        prompts = prompts.astype(int)
        print('MNE event_id array for epoching:')
        print(prompts[:5])

        # Create event identification dictionary
        event_id = {'/iy/': 0, '/uw/': 1, '/piy/': 2, '/tiy/': 3, '/diy/': 4, '/m/': 5,
                    '/n/': 6, 'pat': 7, 'pot': 8, 'knew': 9, 'gnaw': 10}

        # This epochs the data according the the array prompts defined above
        # Using the baseline parameter allows to 'baseline' the epoch against a specified window of eeg
        # - defaults to, for each channel, taking the mean over the entire sample space then subtracting
        #   the mean from the data.
        # Could be used to baseline against Rest epoch at the start of each trial

        epochs_filtered = mne.Epochs(filtered, prompts, event_id, tmin=-0.01, tmax=5.0, baseline=None)
        epochs_raw = mne.Epochs(raw, prompts, event_id, tmin=-0.01, tmax=5.0, baseline=None)
        epochs_ica = mne.Epochs(ica_data, prompts, event_id, tmin=-0.01, tmax=5.0, baseline=None)

        del events, prompts

        # Create ICA
        method = 'fastica'

        # Choose other parameters
        n_components = 0.998  # if float, select n_components by explained variance of PCA
        #decim = 3  #if needed, decimate the time points for efficiency
        random_state = 23

        ica = ICA(n_components=n_components, method=method, random_state=random_state)
        print(ica)

        # Typical threshold for rejection as it is undesireable to fit to these extreme values
        reject = dict(mag=5e-12, grad=4000e-13)

        # fit ICA
        ica.fit(epochs_ica, reject=reject)
        print(ica)

        # Plot ICA components
        i = 0
        for comp in ica.plot_components():
            plt.close(comp.savefig(save_path + '\\ICA_Components_' + str(i)))
            i += 1

        # Plot the reconstructed sources (no ICA exclusions)
        plt.close(ica.plot_sources(epochs_filtered).savefig(save_path + '\\Filtered_Epochs_ICAcomps'))

        # Plot the first 5 ICA component properties, and save each plot
        i = 0
        for fig in ica.plot_properties(epochs_filtered):
            name = save_path + '\\ica_plot_properties_ICA_' + str(i) + '.png'
            plt.close(fig.savefig(name))
            i += 1

        # Look for general artifacts
        ica.detect_artifacts(epochs_ica)

        # Look for EOG Artifacts (ocular) with generous frequency and threshold for detection
        # to add to the exclusion matrix, need to extend the matrix ica.exclude created above
        #eogs = mne.preprocessing.find_eog_events(filtered, event_id=998, l_freq=1, h_freq=100,
        #                                         ch_name=('FP1'), filter_length='10s', thresh=1)

        # Apply the ICA, with exclusions for components deemed to contain artifacts
        ica.apply(epochs_ica, n_pca_components=n_components, exclude=ica.exclude)

        # Plot of the filtered eeg data that corresponds to the ICA components
        plt.close(filtered.plot(color='darkblue').savefig(save_path + "\\Filtered_EEG_Data"))

        # Plot of the epoched filtered eeg data that corresponds to the ICA components
        plt.close(epochs_filtered.plot().savefig(save_path + "\\Epoch_Filtered_EEG_Data"))

        # Plot of the ICA components (with exclusions)
        plt.close(ica.plot_sources(epochs_ica).savefig(save_path + "\\ICA_sources_Filtered_excluded"))

        # Plot of the raw eeg data that corresponds to the ICA components
        plt.close(ica.plot_sources(raw).savefig(save_path + "ICA_sources_Raw"))

        # Plot of the epoched filtered eeg data that corresponds to the ICA components
        plt.close(epochs_ica.plot().savefig(save_path + "\\Epoch_ICA_recon_EEG_Data"))

        # Overlay plot of ICA components and filtered eeg data (no ICA)
        plt.close(ica.plot_overlay(ica_data, exclude=ica.exclude, title='ICA and Filtered EEG Overlay',
                         n_pca_components=n_components).savefig(save_path + "\\ICA_Filtered_Overlay"))

        # This allows for the computation of source scores with the provided scipy function
        # Must be a 1D array input for the score_func as the KARA One data imports with no reference electrode
        # such as MEG, hence no reference to compute scores against.

        # scores = ica.score_sources(epochs, score_func=scipy.stats.kurtosis)
        scores = ica.score_sources(epochs_ica, score_func=scipy.stats.skew)
        print("The ICA component scores with Pearson kurtosis as the scoring function:")
        print(scores)

        # Define the epoched ICA components
        epochs_ica = ica.get_sources(epochs_ica)

        # Create pandas dataframes of the epoch data (raw, filtered, and filtered with only ICA components)
        df_epochs_raw = epochs_raw.to_data_frame()
        df_epochs_filtered = epochs_filtered.to_data_frame()
        df_epochs_ica = epochs_ica.to_data_frame()

        #Save the data frames to pickles
        df_epochs_raw.to_pickle(save_path + "\\df_epochs_raw.pkl")
        df_epochs_filtered.to_pickle(save_path + "\\df_epochs_filtered.pkl")
        df_epochs_ica.to_pickle(save_path + "\\df_epochs_ica.pkl")



if __name__ == "__main__":
    main()