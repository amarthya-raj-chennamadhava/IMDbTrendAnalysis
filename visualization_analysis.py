import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = r'C:\Users\amart\Desktop\IMDbTrendAnalysis\data\cleaned_IMDB_Movie_Data.csv'
df = pd.read_csv(file_path)
# Split multiple genres and count each genre separately
all_genres = df["genre"].str.split(", ").explode()

# Count the occurrences of each genre
genre_counts = all_genres.value_counts()

# Plot the bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="viridis")
plt.xticks(rotation=45)
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.title("Popularity of Different Movie Genres")
plt.show()