import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def create_model(data):

    # # Split data into user, movie, and rating
    # users, movies, ratings, _ = zip(*[d.split(',') for d in data])

    # # Get unique users and movies
    # unique_users = np.unique(users)
    # unique_movies = np.unique(movies)

    # # Create matrix with user ratings
    # matrix = np.zeros((len(unique_users), len(unique_movies)))
    # for d in data:
    #     user, movie, rating, _ = d.split(',')
    #     matrix[np.where(unique_users == user)[0][0], np.where(unique_movies == movie)[0][0]] = rating

    # # Apply KMeans clustering
    # kmeans = KMeans(n_clusters=5)
    # kmeans.fit(matrix)

    # # Apply PCA to reduce dimensionality
    # pca = PCA(n_components=2)
    # transformed = pca.fit_transform(matrix)

    # # Scale the data for plotting
    # scaler = StandardScaler()
    # pca_data = scaler.fit_transform(transformed)

    # # Visualize clusters
    # # plt.figure(figsize=(10, 8))
    # # plt.scatter(transformed[:, 0], transformed[:, 1], c=kmeans.labels_, marker='.')

    # # Plot the data
    # colors = ['r', 'g', 'b', 'y', 'm']
    # for i in range(len(kmeans.labels_)):
    #     plt.scatter(pca_data[i, 0], pca_data[i, 1], marker='o', color=colors[kmeans.labels_[i]])

    # plt.xlabel('PCA 1')
    # plt.ylabel('PCA 2')
    # plt.title('KMeans Clustering with PCA (n_clusters=5)')
    # Parse data into numpy array
    ratings = []
    for line in data:
        line = line.split(',')
        user = int(line[0][6:])
        movie = int(line[1][7:])
        rating = int(line[2])
        ratings.append([user, movie, rating])
    ratings = np.array(ratings)

    # Create user-movie matrix
    matrix = np.zeros((np.max(ratings[:, 0]), np.max(ratings[:, 1])))
    for rating in ratings:
        matrix[rating[0] - 1, rating[1] - 1] = rating[2]

    # Standardize data
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)

    # Apply PCA
    pca = PCA(n_components=2)
    matrix_pca = pca.fit_transform(matrix)

    # Apply KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(matrix_pca)
    labels = kmeans.labels_

    # Plot data
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('KMeans Clustering')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    for i in range(matrix_pca.shape[0]):
        color = colors[labels[i] % len(colors)]
        ax.scatter(matrix_pca[i, 0], matrix_pca[i, 1], c=color, marker='.')

    plt.show()
    plt.savefig('kmeans-pca.png')