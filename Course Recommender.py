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
course_desc_col = 'course_organization'  # Using course_organization as a placeholder for course descriptions
user_id_col = 'course_Certificate_type'  # Assuming this column could serve as user_id in ratings context
rating_col = 'course_rating'
course_difficulty_col = 'course_difficulty'
course_students_enrolled_col = 'course_students_enrolled'

# Display the first few rows to understand the structure and columns of the dataset
print(coursea_data.head())

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

# Content-based Recommender System using Unsupervised Learning
def content_based_recommender(user_profile, threshold=0.1):
    vectorizer = TfidfVectorizer(stop_words='english')
    course_matrix = vectorizer.fit_transform(coursea_data[course_desc_col])
    user_vector = vectorizer.transform
def content_based_recommender(user_profile, threshold=0.1):
    vectorizer = TfidfVectorizer(stop_words='english')
    course_matrix = vectorizer.fit_transform(coursea_data[course_desc_col])
    user_vector = vectorizer.transform([user_profile])
    
    cosine_sim = cosine_similarity(user_vector, course_matrix)
    recommendations = cosine_sim.argsort()[0, -10:][::-1]
    
    recommended_courses = coursea_data.iloc[recommendations]
    return recommended_courses

# Collaborative Filtering Recommender System using Supervised Learning
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