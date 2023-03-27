import numpy as np
import time
from components.graphs.histograms import ratings_histogram,date_users_rates_histogram
from components.models.uniqueModels import UniqueElements

start = time.time()
# Load the .npy file into a numpy array
# and parse it into obj[]
data = np.load('Dataset.npy')

sortData = np.sort(data)
lenData = len(data)

actualData = sortData[:3000]

unique = UniqueElements(sortData)
unique_users = unique.unique_users();
unique_users = unique.unique_movies();

end = time.time()
print(end - start)