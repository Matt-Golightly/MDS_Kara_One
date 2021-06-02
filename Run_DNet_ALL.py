'''
This is used to repeatedly run DenseNet_One_vs_Rest_ALL.py for the many variations of GAF, data sets, words
'''

import os


for gaf in ['GASF', 'GADF']:                       #['GASF', 'GADF']
    for method in ['DTCWT', 'FILTERED', 'RAW', 'ICA']:     # type of image method, ['DTCWT', 'FILTERED', 'RAW', 'ICA']
        for word in ['gnaw', 'knew', 'pat', 'pot']:                         #['gnaw', 'knew', 'pat', 'pot']
            os.system("python DenseNet_One_vs_Rest_ALL.py {} {} {}".format(gaf, word, method))

