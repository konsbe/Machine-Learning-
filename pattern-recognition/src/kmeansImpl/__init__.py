import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def create_model(data) -> None:

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

def create_model_with_vector(data) -> None:

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

    # Apply Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(matrix_pca)

    # Plot scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans.labels_, cmap='viridis')

    # Add arrow for euclidean vector
    mean_vec = np.mean(X_scaled, axis=0)
    ax.quiver(mean_vec[0], mean_vec[1], 2, 2, angles='xy', scale_units='xy', scale=1, color='red')

    # Add colorbar
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    # Set axis limits and labels
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    plt.show()
    plt.savefig('kmeans-pca-vector.png')



def create_3Dmodel(data) -> None:
    # Load the data from the .npy file
    # Create a dictionary of movies and a list of unique users
    movies = {}
    users = set()
    for line in data:
        user, movie, rating, _ = line.split(',')
        rating = int(rating)
        users.add(user)
        if movie not in movies:
            movies[movie] = {}
        movies[movie][user] = rating

    # Create a matrix where each row represents a unique movie and each column represents a unique user
    matrix = np.zeros((len(movies), len(users)))
    for i, movie in enumerate(movies):
        for j, user in enumerate(users):
            if user in movies[movie]:
                matrix[i, j] = movies[movie][user]

    # Apply StandardScaler to scale the matrix
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)

    # Apply PCA to reduce the dimensionality to 3
    pca = PCA(n_components=3)
    pca_matrix = pca.fit_transform(scaled_matrix)

    # Apply KMeans to cluster the data
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pca_matrix)
    labels = kmeans.labels_

    # Create a 3D scatter plot of the data
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            color = labels[i] == 0 and '#ab0000' or labels[i] == 1 and '#75e1ff' or labels[i] == 2 and '#37874d' or labels[i] == 3 and 'orange' or '#a35597';

            if matrix[i][j] > 0:
                x = i
                y = j
                z = matrix[i][j]
                c = labels[i]
                ax.scatter(x, y, z, c=c, marker='o', s=50)

    ax.set_xlabel('Movies')
    ax.set_ylabel('Users')
    ax.set_zlabel('Ratings')
    ax.set_xlim([0, len(movies)])
    ax.set_ylim([0, len(users)])
    ax.set_zlim([0, 10])
    plt.show()
    plt.savefig('3D-kmeans-pca.png')
