import numpy as np
#scripting ond data to process them
# Load the .npy file into a numpy array
data = np.load('Dataset.npy')


# #processing the data
sortData = np.sort(data)
lenData = len(data)


# # spactating the data
# # Print the first 10 rows of the array
# first_10_rows = data[:10]
# print(first_10_rows, "\ntype of row: ", type(data[0]))

objects = []

# Split each string in the data array and create an object with the values
for item in sortData:
    values = item.split(',')
    obj = {
        'userId': values[0],
        'movieId': values[1],
        'rating': values[2],
        'date': values[3]
    }
    objects.append(obj)

# Print unique movie IDs and user IDs
unique_movies = set(obj['movieId'] for obj in objects)
unique_users = set(obj['userId'] for obj in objects)



