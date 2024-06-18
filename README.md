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

## Getting Started
1. Clone this repository to your local machine.
   
2. Ensure you have the necessary libraries installed. You can install them using pip:

pip install pandas numpy matplotlib wordcloud scikit-learn surprise

3. Place your course data (here, it's coursera_data.csv) in the same directory as the script.

4. Run the script:

python Course\ Recommender.py

## License
This project is licensed under the MIT License. For more info, see the [LICENSE](https://github.com/JaCar-868/Course-Recommender/blob/main/LICENSE) file.
