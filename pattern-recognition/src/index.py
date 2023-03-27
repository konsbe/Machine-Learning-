import numpy as np
import time
from components.graphs.histograms import ratings_histogram,date_users_rates_histogram
from components.graphs.ex import Histograms
from components.models.uniqueModels import UniqueElements

start = time.time()
# Load the .npy file into a numpy array
# and parse it into obj[]
data = np.load('Dataset.npy')

sortData = np.sort(data)
lenData = len(data)

actualData = sortData[:3000]

# plot = Histograms(sortData);
# plot.ratings_histogram();


unique = UniqueElements(sortData)
unique_users = unique.unique_users();
unique_movies = unique.unique_movies();

ratings_histogram(actualData);
date_users_rates_histogram(actualData);

print(unique_movies)
print(unique_users)

end = time.time()
print(end - start)