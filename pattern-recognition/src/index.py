import numpy as np
import time
from components.graphs.histograms import ratings_histogram,date_users_rates_histogram
from kmeansImpl.model import Matrix
from components.models.uniqueModels import UniqueElements

class Main():

    start = time.time()
    # Load the .npy file into a numpy array and parse it into obj[]
    data = np.load('Dataset.npy')
    #sorting the data by userId
    sortData = np.sort(data)
    actualData = sortData[:3000]

    def __init__(self) -> None:
        #finding unique elements
        unique = UniqueElements(self.sortData)
        unique_users = unique.unique_users();
        unique_movies = unique.unique_movies();
        print(len(unique_movies))
        print(len(unique_users))

        #creating the histograms
        ratings_histogram(self.actualData);
        date_users_rates_histogram(self.actualData);

        #creating the np matrix
        matrix = Matrix(self.actualData)
        matrix.createMatrix()
        end = time.time()
        print(end - self.start)

main = Main();

