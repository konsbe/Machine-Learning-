import numpy as np
import time
from components.graphs.histograms import ratings_histogram,date_users_rates_histogram
# from components.graphs.ex import Histograms
from components.models.uniqueModels import UniqueElements


class Main():

    start = time.time()
    # Load the .npy file into a numpy array and parse it into obj[]
    data = np.load('Dataset.npy')
    #sorting the data by userId
    sortData = np.sort(data)
    actualData = sortData[:3000]

    # plot = Histograms(sortData);
    # plot.ratings_histogram();
    def __init__(self) -> None:
        unique = UniqueElements(self.sortData)
        unique_users = unique.unique_users();
        unique_movies = unique.unique_movies();

        ratings_histogram(self.actualData);
        date_users_rates_histogram(self.actualData);

        print(unique_movies)
        print(unique_users)

        end = time.time()
        print(end - self.start)

main = Main();