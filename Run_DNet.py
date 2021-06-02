'''
This is used to repeatedly run DenseNet_One_vs_Rest2.py for the many variations of GAF, data sets, words, and subjects
'''

import os

subjects = ['MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02']

for subject in subjects:
    for gaf in ['GASF', 'GADF']:                       #['GASF', 'GADF']
        for method in ['DTCWT', 'FILTERED', 'RAW', 'ICA']:     # type of image method, ['DTCWT', 'FILTERED', 'RAW', 'ICA']
            for word in ['gnaw', 'knew', 'pat', 'pot']:                         #['gnaw', 'knew', 'pat', 'pot']
                os.system("python DenseNet_One_vs_Rest2.py {} {} {} {}".format(gaf, word, method, subject))

