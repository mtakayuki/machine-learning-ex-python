'''
reads the fixed vocabulary list in vocab.txt and returns a cell array
of the words
'''

import pandas as pd


def getVocabList():
    '''
    reads the fixed vocabulary list in vocab.txt and returns
    a cell array of the words in vocabList.
    '''

    return pd.read_table('vocab.txt', header=None, index_col=1)
