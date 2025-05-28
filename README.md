# book-recommendation-system
This project implements a book recommendation system using Python, Pandas, and machine learning techniques including TF-IDF and cosine similarity.   It processes user ratings and book metadata to generate personalized recommendations, supporting data-driven decision-making and enhancing user engagement.
## Features
- Data cleaning and preprocessing
- Content-based filtering with TF-IDF vectorization
- Collaborative filtering using user-item interactions
- Clear, insightful visualizations for data exploration and model results

## Usage
Run the Jupyter Notebook to explore data, train models, and generate recommendations.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Simulated book data for visuals
book_titles = [
    "The Hobbit", "1984", "Harry Potter", "The Alchemist", "The Da Vinci Code",
    "Pride and Prejudice", "To Kill a Mockingbird", "The Great Gatsby", "The Catcher in the Rye", "Moby Dick"
]
ratings_count = [950, 870, 1200, 780, 920, 850, 890, 760, 800, 690]

# --- Plot 1: Top Rated Books Bar Chart ---
plt.figure(figsize=(10, 6))
plt.barh(book_titles[::-1], ratings_count[::-1], color='skyblue')
plt.xlabel('Number of Ratings')
plt.title('Top 10 Most Rated Books')
plt.tight_layout()
plt.savefig('top_books_chart.png')
plt.close()

# --- Plot 2: Heatmap of Simulated User-Book Ratings ---
np.random.seed(42)
user_ids = [f"User {i}" for i in range(1, 11)]
ratings_data = np.random.randint(1, 6, size=(10, 10)).astype(float)
mask = np.random.rand(*ratings_data.shape) < 0.3
ratings_data[mask] = np.nan
ratings_df = pd.DataFrame(ratings_data, index=user_ids, columns=book_titles)

plt.figure(figsize=(12, 6))
sns.heatmap(ratings_df, annot=True, cmap='YlGnBu', linewidths=.5, linecolor='gray')
plt.title('User-Book Ratings Heatmap (Simulated Sample)')
plt.tight_layout()
plt.savefig('heatmap_ratings.png')
plt.close()

# --- TF-IDF Similarity Table (Simulated) ---
tfidf_similar_books = pd.DataFrame({
    'Book': ["The Fellowship of the Ring", "The Two Towers", "Eragon", "Harry Potter and the Sorcerer\'s Stone", "Game of Thrones"],
    'Similarity Score': [0.91, 0.88, 0.85, 0.82, 0.80]
})

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off')
tbl = ax.table(cellText=tfidf_similar_books.values,
               colLabels=tfidf_similar_books.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2)
plt.title("Top 5 Similar Books to 'The Hobbit' (TF-IDF Similarity)", pad=20)
plt.savefig('tfidf_similarity_table.png', bbox_inches='tight')
plt.close()

# --- Final Recommendation Output (Simulated) ---
final_recs = pd.DataFrame({
    'User ID': ['User 101', 'User 202', 'User 303'],
    'Recommended Books': [
        'The Alchemist, The Prophet, Siddhartha',
        '1984, Brave New World, Fahrenheit 451',
        'Harry Potter, Eragon, Percy Jackson'
    ],
    'Confidence Score': [0.92, 0.88, 0.85]
})

fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
tbl = ax.table(cellText=final_recs.values,
               colLabels=final_recs.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2)
plt.title("Sample Final Recommendation Output", pad=20)
plt.savefig('final_recommendation_output.png', bbox_inches='tight')
plt.close()
