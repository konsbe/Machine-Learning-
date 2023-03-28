import datetime
import numpy as np

class Matrix():
    filtered_data = []
    def __init__(self, data):
        # Parse date and find minimum and maximum dates
        dates = [datetime.datetime.strptime(row.split(',')[3], '%d %B %Y') for row in data]
        min_date = min(dates)
        max_date = max(dates)

        # Apply date filters
        start_date = datetime.datetime(2005, 1, 1)  # Example start date
        end_date = datetime.datetime(2005, 2, 25)  # Example end date
        self.filtered_data = [row for row in data 
                        if start_date <= datetime.datetime.strptime(row.split(',')[3], '%d %B %Y') <= end_date]

    def createMatrix(self) -> np.array:
        # Create matrix with ratings
        users = sorted(list(set([row.split(',')[0] for row in self.filtered_data])))
        movies = sorted(list(set([row.split(',')[1] for row in self.filtered_data])))
        matrix = np.zeros((len(users), len(movies)))

        for row in self.filtered_data:
            user_idx = users.index(row.split(',')[0])
            movie_idx = movies.index(row.split(',')[1])
            rating = int(row.split(',')[2])
            matrix[user_idx, movie_idx] = rating

        print(matrix)
        return matrix;

    


