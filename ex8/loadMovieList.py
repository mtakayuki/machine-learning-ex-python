'''
reads the fixed movie list in movie.txt and returns a cell array of the words
'''

import numpy as np


def loadMovieList():
    # Store all movies in cell array movie{}
    # n = 1682  # Total number of movies

    movieList = []

    # Store all movies in cell array movie{}
    with open('movie_ids.txt', 'r', encoding='latin-1') as lines:
        for line in lines:
            # Word Index (can ignore since it will be = i)
            idx, movieName = line.split(' ', 1)
            # Actual Word
            movieList.append(movieName.rstrip('\n'))

    return np.array(movieList)
