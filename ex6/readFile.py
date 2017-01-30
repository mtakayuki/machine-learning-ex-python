'''
reads a file and returns its entire contents
'''


def readFile(filename):
    '''
    reads a file and returns its entire contents in file_contents
    '''

    # Load File
    with open(filename, 'r') as f:
        file_contents = f.read()

    return file_contents
