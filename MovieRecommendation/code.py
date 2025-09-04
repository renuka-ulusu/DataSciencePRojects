import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load datasets
ratings = pd.read_csv("ml-latest-small/ratings.csv")
print(ratings.head())

movies = pd.read_csv("ml-latest-small/movies.csv")
print(movies.head())

# Basic stats
n_ratings = len(ratings)
n_movies = len(ratings['movieId'].unique())
n_users = len(ratings['userId'].unique())

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average ratings per movie: {round(n_ratings/n_movies, 2)}")

# User frequency
user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
print(user_freq.head())

# Mean ratings
mean_rating = ratings.groupby('movieId')[['rating']].mean()
lowest_rated = mean_rating['rating'].idxmin()
print("Lowest rated movie:", movies.loc[movies['movieId'] == lowest_rated]['title'].values)

highest_rated = mean_rating['rating'].idxmax()
print("Highest rated movie:", movies.loc[movies['movieId'] == highest_rated]['title'].values)

# Movie stats (count + mean rating)
movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()

from scipy.sparse import csr_matrix

def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())
 
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
 
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper
    
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

from sklearn.neighbors import NearestNeighbors

def find_similar_movies(movie_id, X, k, metric='cosine'):
    if movie_id not in movie_mapper:
        print(f"Movie ID {movie_id} not found in movie_mapper!")
        return []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1  # include the movie itself
    
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    
    distances, indices = kNN.kneighbors(movie_vec, return_distance=True)
    
    neighbours = []
    for i in range(1, k):  # skip the first (itself)
        n = indices[0][i]
        sim = 1 - distances[0][i]  # cosine similarity
        mid = movie_inv_mapper[n]
        # Apply weighting with rating count
        count = movie_stats.loc[mid, 'count'] if mid in movie_stats.index else 1
        weighted_score = sim * np.log1p(count)  # log(1+count) reduces bias for very high counts
        neighbours.append((mid, sim, weighted_score, count))
    
    return neighbours

def recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10, top_n=3, show_plot=True):
    df1 = ratings[ratings['userId'] == user_id]

    if df1.empty:
        print(f"No ratings found for user {user_id}")
        return

    # Get top N rated movies for the user
    top_movies = df1.sort_values(by="rating", ascending=False).head(top_n)

    movie_titles = dict(zip(movies['movieId'], movies['title']))
    recommended = {}

    for _, row in top_movies.iterrows():
        movie_id = row['movieId']
        if movie_id in movie_titles:
            print(f"\nSince you loved '{movie_titles[movie_id]}', you might also like:")
        
        similar_movies = find_similar_movies(movie_id, X, k)
        
        for mid, sim, weighted_score, count in similar_movies:
            if mid in movie_titles:
                if mid not in recommended or weighted_score > recommended[mid][0]:
                    recommended[mid] = (weighted_score, sim, count)

    # Sort recommendations by weighted score
    sorted_recs = sorted(recommended.items(), key=lambda x: x[1][0], reverse=True)

    for mid, (wscore, sim, count) in sorted_recs:
        print(f"  - {movie_titles[mid]} (similarity: {sim:.2f}, ratings: {count}, weighted_score: {wscore:.2f})")

    # Visualization
    if show_plot and top_movies.shape[0] > 0 and len(sorted_recs) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        # User's top movies
        sns.barplot(
            x=top_movies['rating'], 
            y=[movie_titles[mid] for mid in top_movies['movieId']], 
            color="skyblue", label="User Top Rated", ax=ax
        )

        # Recommended movies with weighted similarity
        rec_df = pd.DataFrame({
            "Movie": [movie_titles[mid] for mid, _ in sorted_recs[:15]],  # top 15 recs
            "Weighted Score": [wscore for _, (wscore, sim, count) in sorted_recs[:15]]
        })

        sns.barplot(
            x=rec_df['Weighted Score'], 
            y=rec_df['Movie'], 
            color="lightgreen", label="Recommended", ax=ax
        )

        ax.set_title(f"User {user_id} – Top Rated vs Recommended Movies (Weighted by Rating Count)")
        ax.set_xlabel("Score (Rating / Weighted Similarity)")
        ax.legend()
        plt.tight_layout()
        plt.show()

# Example usage
user_id = 150 
recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10, top_n=3, show_plot=True)
