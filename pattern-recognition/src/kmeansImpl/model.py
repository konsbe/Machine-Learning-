import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


class Matrix():
    filtered_data = []
    matrix = np.array([])
    num_users, num_movies = 0, 0;

    def __init__(self, data):
        # Parse date and find minimum and maximum dates
        dates = [datetime.datetime.strptime(
            row.split(',')[3], '%d %B %Y') for row in data]
        min_date = min(dates)
        max_date = max(dates)

        # Apply date filters
        start_date = datetime.datetime(2005, 1, 1)  # Example start date
        end_date = datetime.datetime(2005, 9, 25)  # Example end date
        self.filtered_data = [row for row in data
                              if start_date <= datetime.datetime.strptime(row.split(',')[3], '%d %B %Y') <= end_date]

    def createMatrix(self) -> np.array:
        # Create matrix with ratings
        users = sorted(list(set([row.split(',')[0]
                       for row in self.filtered_data])))
        movies = sorted(list(set([row.split(',')[1]
                        for row in self.filtered_data])))
        self.matrix = np.zeros((len(users), len(movies)))

        for row in self.filtered_data:
            user_idx = users.index(row.split(',')[0])
            movie_idx = movies.index(row.split(',')[1])
            rating = int(row.split(',')[2])
            self.matrix[user_idx, movie_idx] = rating

        #setting users-movies limits for our axis        
        self.num_users, self.num_movies = self.matrix.shape
        #return matrix
        return self.matrix

    def kameansAlgorithm(self):
        # Extract the number of users, movies, and ratings from the matrix
        num_users, num_movies = self.matrix.shape

        # Perform k-means clustering on the matrix
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        cluster_labels = kmeans.fit_predict(self.matrix)

        # Extract some information about the total number of users, movies, and ratings
        total_users = num_users
        total_movies = num_movies
        total_ratings = np.count_nonzero(self.matrix)

        # Create a 3D plot of the clustering results
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the movie ratings as dots
        for user in range(total_users):
            for movie in range(total_movies):
                if self.matrix[user, movie] != 0:
                    #red, blue, green, orange, pink
                    color = cluster_labels[user] == 0 and '#ab0000' or cluster_labels[user] == 1 and '#75e1ff' or cluster_labels[user] == 2 and '#37874d' or cluster_labels[user] == 3 and 'orange' or '#a35597';

                    ax.scatter(
                        movie, user, self.matrix[user, movie], c= color, alpha=0.6)

        # Plot the cluster centroids as small stars
        for i in range(num_clusters):
            centroid = kmeans.cluster_centers_[i]
            ax.scatter(centroid[0], centroid[1],
                       centroid[2], marker='*', s=100, c='red')

        self.plot_models(ax)

    def blobs_for_Gaussian_distro(self):
        plt.rcParams['figure.figsize'] = (16, 9)

        n_samples = self.matrix.shape[0]
        n_features = self.matrix.shape[1]
        centers = 5
        X, y = make_blobs(n_samples=n_samples-1,
                          n_features=n_features-1, centers=4)

       # perform k-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42).fit(X)

        # plot results
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.labels_,
                   cmap='viridis', marker='o', alpha=0.5)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                   :, 1], kmeans.cluster_centers_[:, 2], marker='*', s=500, c='r')
        
        self.plot_models(ax)

    def kmeans(self):
        plt.rcParams['figure.figsize'] = (16, 9)

        # # Apply KMeans clustering
        total_ratings_per_movie = np.sum(self.matrix, axis=1)
        total_ratings_per_user = np.sum(self.matrix, axis=0)

        # # apply KMeans clustering to the matrix data
        n_clusters = 5  # choose the number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.matrix)
        num_movies = self.matrix.shape[1]
        num_users = self.matrix.shape[0]
        num_ratings = np.count_nonzero(self.matrix)

        # Export a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        centers = kmeans.cluster_centers_
        # # plot the data points for each movie and user as a dot
        for i in range(num_movies):
            if (i == n_clusters):
                break
            for j in range(num_users):
                ax.scatter(
                    total_ratings_per_movie[i], total_ratings_per_user[j], self.matrix[i, j], c='blue', marker='.')
        # # plot the centroid for each cluster as a small star
        for i in range(n_clusters):
            centroid = kmeans.cluster_centers_[i]
            ax.scatter(np.sum(total_ratings_per_movie[i]), np.sum(
                total_ratings_per_user[i]), centroid, c='red', marker='*', s=50)

        self.plot_models(ax)


    def plot_models(self, ax):
        #setting labels and exporting our model
        ax.set_xlabel('Total Movies')
        ax.set_ylabel('Total Users')
        ax.set_zlabel('Ratings')
        ax.set_xlim([0, self.num_movies])
        ax.set_ylim([0, self.num_users])
        ax.set_zlim([0, 10])
        # Show the plot
        plt.show()
        plt.savefig('kmeans.png')
