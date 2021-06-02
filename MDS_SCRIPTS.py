"""
 This is an adaptation of the work by Wojciech Błądek https://github.com/wjbladek/SilentSpeechClassifier, used for automated
  data handling for the KARA ONE open source EEG data on imagined speech.

  The original intent was to have this handle all data preprocessing, however this was separated out to speed up implementation.
  This code handles loading the desired EEG data from subjects.

  The methods filter_data(), ica(), and GAF() are not finished and not utilised in the study.

 """

import glob
import mne
import scipy.io as spio
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy import signal as sig
from time import time


class Dataset:
    """Loading and preprocessing of the KARA ONE database.

    Notes
    -----
    The class provides means for easily automated preproccessing
    and preparation of the KARA ONE dataset. Most of the data
    is kept within the class, only two methods
    (prepare_data() and find_best_features()) are not void.

    Attributes
    ---------
    registry : list
        list of class instances.
    subject : string
        subject name, in this case 'MM05', 'MM21' etc.
        Used for file navigation.
    figures_path : str
        By default it is /YOUR_SCRIPT/figures,
        folder is created if there is none.

    Methods
    -------
    load_data(path_to_data, raw=True)
        Load subject data from the Kara One dataset.
    select_channels(channels=5)
        Choose how many or which channels to use.
    filter_data(lp_freq=49, hpfreq=1, save_filtered_data=False, plot=False)
        Filter subject's EEG data.
    ica(ica='fast')
        Exclude components using Independent Component Analysis.
    prepare_data(mode=2)
        Organise subject's EEG data for machine learning.
    find_best_features(feature_limit=30, scale_data=True, statistic='Anova')
        Select n best features.

    """
    registry = []
    figures_path = os.path.dirname(os.path.abspath(__file__)) + '/figures'
    os.makedirs(figures_path, exist_ok=True)

    def __init__(self, subject):
        self.name = subject
        self.registry.append(self)
        # TODO is self in self.registry neccesary


    def load_data(self, path_to_data, raw=False, filtered=False):
        """Load subject data from the Kara One dataset.

        Notes
        -----
        By default, the function does not load all the channels, it excludes
        ['EKG', 'EMG', 'Trigger', 'STI 014'].
        It uses files the following original KARA ONE files:
            *.cnt                      raw EEG data
            all_features_simple.mat    epoch time intervals
            epoch_inds.mat             ordered list of prompts

        And the .set files generated from extracting the epoched data as per cfcooney's matlab script split_data.m
            *.set       historical matlab file that appears to capture the basic preprocessing performed by the original
                        team that gathered the KARA ONE data. Filtering between 1 - 50 Hz, mean values subtracted from
                        each channel, then laplacian filter applied to each channel using adjacent channels.
                        https://github.com/cfcooney/Imagined-Speech-EEG-Matlab

        Parameters
        ----------
        path_to_data : str
            path to the folder of the database, containing the individual subject folders with the above files.
            For a single subject the folder structure should resemble ...\\KARA_ONE\\MM05\\
        raw : bool
            If true, loads original data (*.cnt). Otherwise loads
            *-filtered.fif, filtered by filter_data(save_filtered_data=True) from previous runs.
        kara : bool
            If true loads from the .mat files created by using cfcooney's script to extract the EEG data from the .set
            KARA ONE files - this contains the preprocessed data from the original research team (as explained on their
            webpage - laplace filter and channel selection)
            Otherwise: does nothing
        """

        if not raw:
            print("Loading Kara One Pre-processed .set data")
            def loadmat(filename):
                '''
                this function should be called instead of direct spio.loadmat
                as it cures the problem of not properly recovering python dictionaries
                from mat files. It calls the function check keys to cure all entries
                which are still mat-objects

                This function requires the matlab engine python package to be installed and associated with the
                environment running this .py
                '''

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

            # the 'ImaginedSpeechData' is a sub folder containing folders by subject with the .m files containing the
            # epoched eeg data extracted from the .set files
            self.dataPath = path_to_data + 'ImaginedSpeechData\\' + self.name
            os.chdir(self.dataPath)
            self.eeg_data = loadmat('EEG_Data.mat')
            #self.eeg_rest = loadmat('EEG_Rest.mat')





        if raw:
            self.dataPath = path_to_data + self.name
            os.chdir(self.dataPath)
            print("Loading raw data from .cnt file.")
            for f in glob.glob("*.cnt"):
                self.eeg_data = mne.io.read_raw_cnt(f, preload=False)
                self.eeg_data.drop_channels(['M1', 'M2', 'VEO', 'HEO', 'EKG', 'EMG', 'Trigger'])
            for f in glob.glob("all_features_simple.mat"):
                prompts_to_extract = spio.loadmat(f)
            self.prompts = prompts_to_extract['all_features'][0, 0]
            for f in glob.glob("epoch_inds.mat"):
                self.epoch_inds = spio.loadmat(f, variable_names=('clearing_inds', 'thinking_inds'))

        elif raw and filtered:
            self.dataPath = path_to_data + self.name
            os.chdir(self.dataPath)
            print("Loading filtered data from .fif file (from a previous run).")
            for f in glob.glob("*-filtered.fif"):
                self.eeg_data = mne.io.read_raw_fif(f, preload=True)
            for f in glob.glob("all_features_simple.mat"):
                prompts_to_extract = spio.loadmat(f)
            self.prompts = prompts_to_extract['all_features'][0, 0]
            for f in glob.glob("epoch_inds.mat"):
                self.epoch_inds = spio.loadmat(f, variable_names=('clearing_inds', 'thinking_inds'))

    def select_channels(self, channels, raw=False):
        """Filter down to a select list of of channels from the KARA ONE headset:

        Notes
        -----
        List of available channels in KARA ONE:

        ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
        'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
        'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
         'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
        'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'OZ', 'O2',
        'CB2', 'O1']

        Excluded : ['M1', 'M2', 'VEO', 'HEO', 'EKG', 'EMG', 'Trigger']

        Parameters
        ----------
        channels : list
            List of strings, it treats them as chosen channels's names.

        raw : bool
            if raw is True then using the .cnt file EEG data, else using the .set EEG data

        Examples
        --------
        Dataset.select_channels(raw=True, channels=['FP1', 'FPZ', 'FP2', 'AF3'])

        """
        # TODO: check if passed channels exist.
        if type(channels) is list and raw:
            self.eeg_data = self.eeg_data.pick_channels(channels)
            print(self.eeg_data.info['ch_names'])
        elif type(channels) is list and not raw:
            indices = []
            for chan in channels:
                indices.append(np.where(np.all([self.eeg_data['EEG_Data']['chanlocs'] == chan], axis=0))[0][0])

            self.channelActive = []
            self.channelRest = []
            for i in range(0, len(self.eeg_data['EEG_Data']['activeEEG'])):
                self.channelActive.append(self.eeg_data['EEG_Data']['activeEEG'][i][indices])
                self.channelRest.append(self.eeg_data['EEG_Data']['restEEG'][i][indices])
            del indices

        else:
            raise AttributeError("Incorrect \"channels\" attribute type, should be a list or left empty.")
            # TODO: check if works


    def filter_data(self, lp_freq=49, hp_freq=1, save_filtered_data=False, plot=False):
        """Filter subject's EEG data.

        Notes
        -----
        Filter is Butterworth, order is 4.

        Parameters
        ----------
        lp_freq : int
            Frequency of a low-pass filter. Pass a false value (None,
            False, 0) to disable the filter.
        hp_freq : int
            Frequency of a high-pass filter. Pass a false value (None,
            False, 0) to disable the filter.
        save_filtered_data : bool
            Saves filtered data, so it can be loaded later by load_data().
            Data is stored in subject's data folder.
        plot : bool
            Plots results from before and after a filtration. The results
            are not shown during the runtime. Instead, the are saved, path
            is stored in self.figures_path, by default /YOUR_SCRIPT/figures.
        """
        print("Filtering data.")
        if plot:
            fig = self.eeg_data.plot_psd(tmax=np.inf, average=False, fmin=0., fmax=130., show=False)
            fig.savefig(os.path.join(self.figures_path, self.name + "_raw_signal.png"))
        if hp_freq:
            for idx, eeg_vector in enumerate(self.eeg_data[:][0]):
                [b, a] = sig.butter(4, hp_freq / self.eeg_data.info['sfreq'] / 2, btype='highpass')
                self.eeg_data[idx] = sig.filtfilt(b, a, eeg_vector)
        if lp_freq:
            for idx, eeg_vector in enumerate(self.eeg_data[:][0]):
                [b, a] = sig.butter(4, lp_freq / self.eeg_data.info['sfreq'] / 2, btype='lowpass')
                self.eeg_data[idx] = sig.filtfilt(b, a, eeg_vector)
        print("Filtering done.")
        if plot:
            fig = self.eeg_data.plot_psd(tmax=np.inf, average=False, fmin=0., fmax=130., show=False)
            fig.savefig(os.path.join(self.figures_path, self.name + "_filtered_signal.png"))
        if save_filtered_data:
            self.eeg_data.save(self.name + "-filtered.fif", overwrite=True)
            print("Filtered data saved as " + self.name + "-filtered.fif")

    def ica(self, ica_type='fast'):
        """Exclude components using Independent Component Analysis.

        Notes
        -----
        Requires a manual input concerning components to be excluded.
        Adequate, interactive plots are provided. Results are saved.

        Parameters
        ----------
        ica_type : {'fast', 'extensive', 'saved'}
            'fast'          fast ICA, but with reduced accuracy.
            'extensive'     slow, but more accurate (extended-infomax)
            'saved'         load previously computed.

        Examples
        --------
        Dataset.ica(ica_type('extensive'))
        Dataset.ica(ica_type('saved'))
        """
        if ica_type == 'fast':
            print("Computing fast ICA.")
            ica = mne.preprocessing.ICA(random_state=1)
        elif ica_type == 'saved':
            print("Loading previously computed ICA.")
            ica = mne.preprocessing.read_ica(self.name + "-ica.fif")
        elif ica_type == 'extensive':
            print("Computing extensive ICA.")
            ica = mne.preprocessing.ICA(method='extended-infomax', random_state=1)
        else:
            raise AttributeError("Incorrect \"ica_type\" attribute value.")
            # TODO: check if works
        ica.fit(self.eeg_data)
        ica.plot_components(inst=self.eeg_data)
        print("Write the components you want to exclude from the data. For instance \"14 28 66\"")
        ica.exclude = [int(x) for x in input().split()]
        ica.apply(self.eeg_data)
        if ica_type != "saved":
            ica.save(self.name + "-ica.fif")


    def GAF(self):
        from pyts.image import GramianAngularField
        from sklearn.pipeline import make_pipeline

        from PIL import Image
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid

        self.gasf = []
        self.gadf = []
        #i=1
        for trial in self.eeg_data['EEG_Data']['activeEEG']:

            # GAM sum and diff for full time trial
            gasf = GramianAngularField(image_size=trial.shape[1], method='summation')
            gadf = GramianAngularField(image_size=trial.shape[1], method='difference')

            # GAM sum and diff reduced to 224x224 (densenet size)
            #gasf = GramianAngularField(image_size=224, method='summation')
            #gadf = GramianAngularField(image_size=224, method='difference')

            X_gasf = gasf.fit_transform(trial)
            X_gadf = gadf.fit_transform(trial)

            GAMpath = self.dataPath + "\\GAM"
            Path(GAMpath).mkdir(parents=True, exist_ok=True)
            os.chdir(GAMpath)
            Path("\\gnaw").mkdir(parents=True, exist_ok=True)
            Path(GAMpath).mkdir(parents=True, exist_ok=True)
            #sum_file = "gasf_trial" + str(i) + ".npy"
            #diff_file = "gasd_trial" + str(i) + ".npy"
            #np.save(sum_file, X_gasf2)  # save
            #np.save(diff_file, X_gadf2)  # save
            #i += 1
            #del gasf, gadf, gasf2, gadf2








if __name__ == '__main__':
    pass