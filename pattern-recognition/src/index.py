import numpy as np
import time
from components.graphs.histograms import ratings_histogram, date_users_rates_histogram
from kmeansImpl.model import Matrix
from kmeansImpl import create_model, create_3Dmodel, create_model_with_vector
from components.models.uniqueModels import UniqueElements


class Main():

    start: float = time.time()
    # Load the .npy file into a numpy array and parse it into obj[]
    data: np.array = np.load('Dataset.npy')
    # sorting the data by userId
    sortData = np.sort(data)
    actualData = sortData[:5000]

    def __init__(self) -> None:
        # finding unique elements
        unique: UniqueElements = UniqueElements(self.sortData)
        unique_users: int = unique.unique_users()
        unique_movies: int = unique.unique_movies()
        print(unique_movies)
        print(unique_users)

        # creating the histograms
        ratings_histogram(self.actualData)
        date_users_rates_histogram(self.actualData)

        # # creating the np matrix
        matrix: Matrix = Matrix(self.actualData)
        matrix.createMatrix()

        # applying kmeans to the matrix
        # matrix.blobs_for_Gaussian_distro()
        # matrix.kmeans()
        matrix.kameansAlgorithm()
        create_model(self.actualData)
        create_model_with_vector(self.actualData)
        # create_3Dmodel(self.actualData)
        end: float = time.time()
        print(end - self.start)


