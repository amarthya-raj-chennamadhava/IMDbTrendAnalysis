import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
file_path = r'C:\Users\amart\Desktop\IMDbTrendAnalysis\data\cleaned_IMDB_Movie_Data.csv'
df = pd.read_csv(file_path)

# Convert all movie titles to lowercase for case-insensitive search
df["Title_lower"] = df["title"].str.lower()

# Fill missing values in Genre and Description
df["genre"] = df["genre"].fillna("")
df["description"] = df["description"].fillna("")

# Combine important features for similarity calculation
df["Combined"] = df["genre"] + " " + df["description"]

# Use TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Combined"])

# Compute similarity scores
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def recommend_movies(movie_title, df, min_rating=None):
    # Convert user input to lowercase
    movie_title = movie_title.lower()

    # Check if the movie exists in the dataset
    if movie_title not in df["Title_lower"].values:
        print("Movie not found. Please try again with a different title.")
        return []

    # Get the actual movie index
    movie_idx = df[df["Title_lower"] == movie_title].index[0]

    # Get similarity scores and sort them in descending order
    scores = list(enumerate(similarity_matrix[movie_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10 similar movies

    # Get movie recommendations
    recommended_movies = []
    for idx, score in scores:
        if min_rating is None or df.iloc[idx]["rating"] >= min_rating:
            recommended_movies.append(df.iloc[idx]["title"])

    return recommended_movies

# Example usage:
movie_name = "lion"  # You can change this to test
recommended_movies = recommend_movies(movie_name, df, min_rating=7.0)

print(f"recommended movies similar to '{movie_name}':")
for movie in recommended_movies:
    print("-", movie)
