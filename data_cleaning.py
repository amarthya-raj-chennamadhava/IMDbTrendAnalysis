import pandas as pd

# Load the dataset
file_path = r'C:\Users\amart\Desktop\IMDbTrendAnalysis\data\IMDB-Movie-Data.csv'
df = pd.read_csv(file_path)

# Step 1: Handle Missing Data
# Fill missing 'Revenue (Millions)' with median value
df['Revenue (Millions)'].fillna(df['Revenue (Millions)'].median(), inplace=True)

# Fill missing 'Metascore' with median value
df['Metascore'].fillna(df['Metascore'].median(), inplace=True)

# Step 2: Check for Duplicates
if df.duplicated().sum() > 0:
    print(f"Duplicates Found: {df.duplicated().sum()} - Removing Duplicates")
    df.drop_duplicates(inplace=True)
else:
    print("No Duplicates Found")

# Step 3: Fix Data Types (if necessary)
# Ensure correct data types are used
df['Year'] = df['Year'].astype(int)
df['Runtime (Minutes)'] = df['Runtime (Minutes)'].astype(int)
df['Rating'] = df['Rating'].astype(float)
df['Votes'] = df['Votes'].astype(int)
df['Revenue (Millions)'] = df['Revenue (Millions)'].astype(float)
df['Metascore'] = df['Metascore'].astype(float)

# Step 4: Rename Columns for Consistency
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

print("Data Cleaning Completed Successfully!")

# Save the cleaned data to a new CSV file
df.to_csv(r'C:\Users\amart\Desktop\IMDbTrendAnalysis\data\cleaned_IMDB_Movie_Data.csv', index=False)
print("Cleaned data saved to cleaned_IMDB_Movie_Data.csv")
