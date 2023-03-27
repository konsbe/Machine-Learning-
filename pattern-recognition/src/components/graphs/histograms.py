import matplotlib.pyplot as plt
# Load the .npy file into a numpy array
# and parse it into obj[]

"""
 Spactating & Processing the data


"""
# # Print the first 10 rows of the array
# first_10_rows = data[:10]
# print(first_10_rows, "\ntype of row: ", type(data[0]))

# objects = []

user_data = {}

def ratings_histogram(data):
    for item in data:
        values = item.split(',')
        userId = values[0]
        movieId = values[1]
        rating = int(values[2])
        if userId in user_data:
            user_data[userId]['movies'].append(movieId)
            user_data[userId]['rating'].append(rating)
        else:
            user_data[userId] = {
                'movies': [movieId],
                'rating': [rating]
            }

    total_ratings_per_user = []
    # Sort movie arrays and fill in missing ratings with 0
    for userId in user_data:
        movies = len(user_data[userId]['movies'])
        total_ratings_per_user.append(movies)

    # Create a histogram with 20 bins
    plt.hist(total_ratings_per_user,  bins=range(1, 16), edgecolor='black')

    # Set the title and labels
    plt.title('Frequency of Total Ratings per User')
    plt.xlabel('Total Ratings')
    plt.ylabel('Frequency (Users)')
    plt.xlim(right=16)
    plt.savefig('ratings_histogram.png')

    # Display the histogram
    plt.show()


users = {}
def date_users_rates_histogram(data):
    for line in data:
        # split the line by comma
        values = line.split(',')
        
        # extract the user id, movie name, rating and date
        user_id = values[0]
        movie = values[1]
        rating = int(values[2])
        date = values[3].strip()
        
        # add user data to dictionary
        if user_id not in users:
            users[user_id] = {
                'movies': {},
                'ratings': {},
                'dates': {}
            }
        users[user_id]['movies'][movie] = 1
        users[user_id]['ratings'][movie] = rating
        users[user_id]['dates'][movie] = date
        
    # create a dictionary to hold user days data
    user_days = {}

    # process user data
    for user_id, user_data in users.items():
        for movie, rating in user_data['ratings'].items():
            date = user_data['dates'][movie]
            if date not in user_days:
                user_days[date] = set()
            user_days[date].add(user_id)

    # sort the dates
    sorted_dates = sorted(user_days.keys(), key=lambda x: x.split('-'))

    # group dates every 30 days
    grouped_dates = {}
    for i, date in enumerate(sorted_dates):
        group_index = i // 30
        if group_index not in grouped_dates:
            grouped_dates[group_index] = {
                'start_date': date,
                'end_date': date,
                'users': set()
            }
        else:
            grouped_dates[group_index]['end_date'] = date
        grouped_dates[group_index]['users'].update(user_days[date])

    # create a list to hold the frequencies
    frequencies = []
    for group_index in sorted(grouped_dates.keys()):
        frequencies.append(len(grouped_dates[group_index]['users']))

    # create a histogram
    plt.hist(frequencies, bins=range(min(frequencies), max(frequencies) + 2, 1))

    # set the x and y labels
    plt.xlabel('Days')
    plt.ylabel('Frequency')

    # set the title
    plt.title('User frequency per days')

    # display the histogram
    plt.show()
    plt.savefig('dates_histogram.png')
