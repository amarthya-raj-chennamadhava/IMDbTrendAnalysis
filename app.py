import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned dataset
file_path = r'C:\Users\amart\Desktop\IMDbTrendAnalysis\data\cleaned_IMDB_Movie_Data.csv'
df = pd.read_csv(file_path)

# TF-IDF Vectorization to find similar movies
vectorizer = TfidfVectorizer(stop_words='english')
df['genre'] = df['genre'].fillna('')
tfidf_matrix = vectorizer.fit_transform(df['genre'])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def recommend_movies(movie_title, min_rating=0):
    movie_title = movie_title.lower()
    if movie_title not in df['title'].str.lower().values:
        return ["Movie not found. Try a different one!"]
    
    movie_idx = df[df['title'].str.lower() == movie_title].index[0]
    similar_scores = list(enumerate(similarity_matrix[movie_idx]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)[1:6]
    recommended_movies = [df.iloc[i[0]] for i in similar_scores if df.iloc[i[0]]['rating'] >= min_rating]
    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie title to get similar recommendations!")

# User Input
user_movie = st.text_input("Enter Movie Name:")
min_rating = st.slider("Minimum Rating", 0.0, 10.0, 0.0)

if st.button("Get Recommendations"):
    if user_movie:
        recommendations = recommend_movies(user_movie, min_rating)
        if isinstance(recommendations[0], str):
            st.error(recommendations[0])
        else:
            for movie in recommendations:
                st.write(f"🎥 {movie['title']} ({movie['year']}), ⭐ {movie['rating']}")
    else:
        st.warning("Please enter a movie name!")
