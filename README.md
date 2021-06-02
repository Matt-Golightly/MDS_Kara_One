# MDS_Kara_One
Imagined Speech Classification with the Kara One Database

This repository contains my approach to extraction, pre-processing, feature engineering, and classification of the imagined speech component of the EEG trials in the Kara One database (url: http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html).

To use the processed data from the original Kara One study, you will need to run split_data_.m that should extract the 'thinking' and 'rest' components of each subjects trials from the .set files from the Kara One data.

Then in using MDS_SCRIPTS, you can set raw=FALSE in loading the data to use the processed data, otherwise leave as raw=TRUE for the raw EEG data straight from the .cnt files (native format of the Neuroscan device).

MDS_SCRIPTS is only used for data loading, most of the methods are unfinished and not used.

EEG_to_Images_SCRIPT_1.py and EEG_to_Images_SCRIPT_2.py run through converting the raw data to images for each subject with EEG preprocessing to produce the following subject data sets:
- Raw EEG
- Filtered (between 1Hz - 45Hz)
- Filtered then ICA reconstructed
- Filtered, then DTCWT absolute values extracted

The settings in each script are left over from exploratory trials, and should be adjusted as required for the intended study. For example the DTWCT extraction in EEG_to_Images_SCRIPT_2.py was left as 5th order and the mask matrix using a single sub-band.

Run_DNet.py and Run_DNet_ALL.py runs DenseNet_One_vs_Rest2.py and DenseNet_One_vs_Rest_ALL.py respectively.

Vanilla_LSTM.py is a simple LSTM implementation for a baseline approach.

Folder structure images are included to give an idea of the required/resulting folder structures.



