import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = r'C:\Users\amart\Desktop\IMDbTrendAnalysis\data\cleaned_IMDB_Movie_Data.csv'
df = pd.read_csv(file_path)

def preprocess_data(df):
    df['genre'] = df['genre'].str.lower()
    df['combined_features'] = df['genre'] + ' ' + df['director'].str.lower() + ' ' + df['actors'].str.lower()
    return df

def compute_similarity(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_matrix = vectorizer.fit_transform(df['combined_features'])
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix

def recommend_movies(movie_name, df, similarity_matrix, min_rating=0):
    movie_name = movie_name.lower()
    if movie_name not in df['title'].str.lower().values:
        print("Movie not found. Please check the spelling.")
        return []
    
    movie_index = df[df['title'].str.lower() == movie_name].index[0]
    similar_movies = list(enumerate(similarity_matrix[movie_index]))
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:11]
    
    recommendations = []
    for index, score in sorted_movies:
        if df.loc[index, 'rating'] >= min_rating:
            recommendations.append((df.loc[index, 'title'], df.loc[index, 'rating']))
    
    return recommendations

if __name__ == "__main__":
    df = preprocess_data(df)
    similarity_matrix = compute_similarity(df)
    
    while True:
        user_movie = input("Enter a movie name (or type 'exit' to quit): ")
        if user_movie.lower() == 'exit':
            break
        
        min_rating = input("Enter minimum rating filter (or press enter to skip): ")
        min_rating = float(min_rating) if min_rating else 0
        
        recommendations = recommend_movies(user_movie, df, similarity_matrix, min_rating)
        if recommendations:
            print("\nRecommended Movies:")
            for title, rating in recommendations:
                print(f"- {title} (Rating: {rating})")
        else:
            print("No recommendations found with the given rating filter. Try again.")
        
        print("\n-----------------------------------\n")
