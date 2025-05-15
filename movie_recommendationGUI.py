import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
file_path = r'C:\Users\amart\Desktop\IMDbTrendAnalysis\data\cleaned_IMDB_Movie_Data.csv'
df = pd.read_csv(file_path)

# Function to recommend movies
def recommend_movies(movie_name):
    movie_name = movie_name.strip().lower()
    df["Title_lower"] = df["title"].str.lower()
    
    if movie_name not in df["Title_lower"].values:
        messagebox.showerror("Error", "Movie not found in dataset!")
        return
    
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["genre"] + " " + df["description"])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = df[df["Title_lower"] == movie_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    
    recommendations = df.iloc[movie_indices]["title"].tolist()
    result_label.config(text="\n".join(recommendations))

# GUI Setup
root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("500x400")

tk.Label(root, text="Enter a Movie Name:").pack(pady=5)
movie_entry = tk.Entry(root, width=40)
movie_entry.pack(pady=5)

tk.Button(root, text="Get Recommendations", command=lambda: recommend_movies(movie_entry.get())).pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12), justify="left")
result_label.pack(pady=10)

root.mainloop()
