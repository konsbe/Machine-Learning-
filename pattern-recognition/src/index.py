import numpy as np

# Load the .npy file into a numpy array
# and parse it into obj[]
data = np.load('Dataset.npy')

sortData = np.sort(data)
lenData = len(data)

actualData = sortData[:10]
"""
 Spactating & Processing the data


"""
# # Print the first 10 rows of the array
# first_10_rows = data[:10]
# print(first_10_rows, "\ntype of row: ", type(data[0]))

# objects = []

# # Split each string in the data array and create an object with the values
# for item in sortData:
#     values = item.split(',')
#     obj = {
#         'userId': values[0],
#         'movieId': values[1],
#         'rating': values[2],
#         'date': values[3]
#     }
#     objects.append(obj)

# # Print unique movie IDs and user IDs
# unique_movies = set(obj['movieId'] for obj in objects)
# unique_users = set(obj['userId'] for obj in objects)
user_data = {}
all_movies = set()
for item in actualData:
    values = item.split(',')
    userId = values[0]
    movieId = values[1]
    rating = int(values[2])
    all_movies.add(movieId)
    if userId in user_data:
        user_data[userId]['rating'][movieId] = rating
    else:
        user_data[userId] = {
            'rating': {movieId: rating},
            'movies': []
        }

# Create a list of objects with sorted movie IDs, all movie ratings, and user ID
user_list = []
for userId in user_data:
    user_movies = sorted(list(all_movies))
    user_rating = []
    for movie in user_movies:
        if movie in user_data[userId]['rating']:
            user_rating.append(user_data[userId]['rating'][movie])
        else:
            user_rating.append(0)
    user_list.append({
        'userId': userId,
        'movies': user_movies,
        'rating': user_rating
    })

# Print the resulting user list
print(len(user_list[0]["movies"]))
print(len(user_list[1]["movies"]))
# print(list(user_data.keys())[1]['movies'])

