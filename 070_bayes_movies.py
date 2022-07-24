from collections import defaultdict

from sklearn.naive_bayes import BernoulliNB
import numpy as np

# https://files.grouplens.org/datasets/movielens/ml-1m.zip


def load_rating_data_dat(data_path, n_users, n_movies):
    data = np.zeros([n_users, n_movies], dtype=np.float32)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines():
            user_id, movie_id, rating, _ = line.split('::')
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(rating)
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping


def load_rating_data_csv(data_path):
    ratings = np.genfromtxt(data_path, delimiter=",", skip_header=1, usecols=[0, 1, 2], dtype=int)
    n_users = np.unique(ratings[:, 0]).size
    n_movies = np.unique(ratings[:, 1]).size
    data = np.zeros([n_users, n_movies], dtype=np.float32)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    for line in ratings:
        user_id, movie_id, rating = line
        user_id = int(user_id) - 1
        if movie_id not in movie_id_mapping:
            movie_id_mapping[movie_id] = len(movie_id_mapping)
        rating = int(rating)
        data[user_id, movie_id_mapping[movie_id]] = rating
        if rating > 0:
            movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping


# data, movie_n_rating, movie_id_mapping = load_rating_data_csv('data/ml-latest-small/ratings.csv')
data, movie_n_rating, movie_id_mapping = load_rating_data_dat('data/ml-1m/ratings.dat', 6040, 3883)


def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print("Liczba ocen %d: %d" % (value, count))


display_distribution(data)

movie_id_most, n_rating_most = sorted(movie_n_rating.items(), key=lambda movie: movie[1], reverse=True)[0]
# movie_id_most_mapped = list(movie_id_mapping.keys())[list(movie_id_mapping.values()).index(movie_id_most)]
print(f"Film o ID {movie_id_most} uzyskał najwięcej ocen: {n_rating_most}.")