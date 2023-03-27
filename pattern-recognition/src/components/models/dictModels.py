

class UsersModel():
    def __init__(self, data):
        self.user_list = []
        # Create a dictionary with unique user IDs as keys and all movie IDs as values
        user_data = {}
        all_movies = set()
        for item in data:
            values = item.split(',')
            userId = values[0]
            movieId = values[1]
            rating = int(values[2])
            date = values[3]
            all_movies.add(movieId)
            if userId in user_data:
                user_data[userId]['rating'][movieId] = {'rating': rating, 'date': date}
            else:
                user_data[userId] = {
                    'rating': {movieId: {'rating': rating, 'date': date}},
                    'movies': []
                }
        # # Create a list of objects with all movie IDs, corresponding ratings, and user ID
        all_movies_list = sorted(list(all_movies))
        for userId in user_data:
            user_movies = []
            for movie in all_movies_list:
                if movie in user_data[userId]['rating']:
                    user_movies.append({
                        'movie': movie,
                        'rating': user_data[userId]['rating'][movie]['rating'],
                        'date': user_data[userId]['rating'][movie]['date']
                    })
                else:
                    user_movies.append({
                        'movie': movie,
                        'rating': 0,
                        'date': None
                    })
            self.user_list.append({
                'userId': userId,
                'movies': user_movies
            })

# # Print the resulting user list
# print(user_list)
