### Course Recommender System
This repository contains a Python program that builds a Coursera course recommendation system. The system leverages both content-based and collaborative filtering techniques to recommend courses to users.

## Features
Exploratory Data Analysis (EDA): Visualizes course enrollments, popularity, and generates word clouds for course titles using matplotlib and WordCloud.
Content-Based Recommender: Uses TF-IDF vectorization and cosine similarity to recommend courses based on course descriptions.
Collaborative Filtering Recommender: Implements a collaborative filtering system using the SVD algorithm from the surprise library and evaluates the model with cross-validation.

## Libraries Used
pandas
numpy
matplotlib
wordcloud
scikit-learn
surprise

## Usage
Load the datasets: Ensure coursea_data.csv is in the same directory as the script.
Run the script: Execute the Python file to perform EDA, generate recommendations, and train the collaborative filtering model.

## Code Overview

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Load datasets
coursea_data = pd.read_csv('coursea_data.csv')

# Column names based on the provided dataset
course_id_col = 'course_id'
course_title_col = 'course_title'
course_desc_col = 'course_organization'
user_id_col = 'course_Certificate_type'
rating_col = 'course_rating'
course_difficulty_col = 'course_difficulty'
course_students_enrolled_col = 'course_students_enrolled'

# Exploratory Data Analysis
def eda(coursea_data):
    # Enrollment distribution
    coursea_data[course_id_col].value_counts().plot(kind='bar', figsize=(10, 6))
    plt.title('Course Enrollment Distribution')
    plt.xlabel('Course ID')
    plt.ylabel('Number of Enrollments')
    plt.show()

    # Most popular courses
    top_20_courses = coursea_data[course_students_enrolled_col].str.replace('k', 'e3').str.replace('M', 'e6').astype(float).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    top_20_courses.plot(kind='bar')
    plt.title('Top 20 Most Popular Courses')
    plt.xlabel('Course ID')
    plt.ylabel('Number of Students Enrolled')
    plt.show()

    # Word cloud of course titles
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(coursea_data[course_title_col]))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Course Titles')
    plt.show()

# Content-based Recommender System
def content_based_recommender(user_profile, threshold=0.1):
    vectorizer = TfidfVectorizer(stop_words='english')
    course_matrix = vectorizer.fit_transform(coursea_data[course_desc_col])
    user_vector = vectorizer.transform([user_profile])
    
    cosine_sim = cosine_similarity(user_vector, course_matrix)
    recommendations = cosine_sim.argsort()[0, -10:][::-1]
    
    recommended_courses = coursea_data.iloc[recommendations]
    return recommended_courses

# Collaborative Filtering Recommender System
def collaborative_filtering_recommender():
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(coursea_data[[user_id_col, course_id_col, rating_col]], reader)
    trainset = data.build_full_trainset()
    
    # Using SVD algorithm
    algo = SVD()
    cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)
    
    return algo

def main():
    # Perform EDA
    eda(coursea_data)

    # Content-based recommendation
    user_profile = "interested in data science and machine learning"
    recommendations = content_based_recommender(user_profile)
    print("Content-based Recommendations:")
    print(recommendations)

    # Collaborative filtering recommendation
    algo = collaborative_filtering_recommender()
    print("Collaborative Filtering Model Trained.")

if __name__ == "__main__":
    main()
    
## Getting Started
1. Clone this repository to your local machine.
2. Ensure you have the necessary libraries installed. You can install them using pip:

pip install pandas numpy matplotlib wordcloud scikit-learn surprise

3. Place your course data (here, it's coursea_data.csv) in the same directory as the script.

4. Run the script:

python Course\ Recommender.py

## License
This project is licensed under the MIT License.
