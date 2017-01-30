'''
preprocesses a the body of an email and returns a list of word_indices
'''

import re
import numpy as np
from nltk.stem import PorterStemmer
from getVocabList import getVocabList


def processEmail(email_contents):
    '''
    preprocesses the body of an email and returns a list of indices of
    the words contained in the email.
    '''

    # Load Vocabulary
    vocabList = getVocabList()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = strfind(email_contents, ([char(10) char(10)]));
    # email_contents = email_contents(hdrstart(1):end);

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub(r'<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub(r'[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr',
                            email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub(r'[$]+', 'dollar', email_contents)

    # ========================== Tokenize Email ===========================
    # Output the email to screen as well
    print('\n==== Processed Email ====\n')

    # Process file
    l = 0

    stemmer = PorterStemmer()
    for str in re.split(r'[ @$/#.-:&*+=\[\]?!(){},\'">_<;%\n\r]',
                        email_contents):
        # Remove any non alphanumeric characters
        str = re.sub(r'[^a-zA-Z0-9]', '', str)

        # Stem the word
        # (the porterStemmer sometimes has issues, so we use a try catch block)
        str = stemmer.stem(str)

        # Skip the word if it is too short
        if len(str) < 1:
            continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary. At this point
        #               of the code, you have a stemmed word from the email in
        #               the variable str. You should look up str in the
        #               vocabulary list (vocabList). If a match exists, you
        #               should add the index of the word to the word_indices
        #               vector. Concretely, if str = 'action', then you should
        #               look up the vocabulary list to find where in vocabList
        #               'action' appears. For example, if vocabList{18} =
        #               'action', then, you should add 18 to the word_indices
        #               vector (e.g., word_indices = [word_indices ; 18]; ).
        #
        # Note: vocabList{idx} returns a the word with index idx in the
        #       vocabulary list.
        #
        # Note: You can use strcmp(str1, str2) to compare two strings (str1 and
        #       str2). It will return 1 only if the two strings are equivalent.

        try:
            i = vocabList.loc[str][0]
            word_indices.append(i)
        except:
            pass

        # ============================================================

        # Print to screen, ensuring that the output lines are not too long
        if l + len(str) + 1 > 78:
            print()
            l = 0
        print('%s ' % str, end='')
        l += len(str) + 1

    # Print footer
    print('\n\n=========================')

    return np.array(word_indices)
