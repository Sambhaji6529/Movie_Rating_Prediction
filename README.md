🎬 Movie Rating Prediction - IMDb India Dataset

📖 Project Overview
This project focuses on building a predictive model to estimate movie ratings based on various attributes of movies, such as cast, director, genre, duration, votes, and more. The model can help predict how a new movie might perform based on its characteristics.

✅ Key Objectives

* Perform data preprocessing, including handling missing values and encoding categorical variables.

* Engineer features like director success rate, genre popularity, and cast average rating.

* Train a machine learning model (RandomForestRegressor) to predict IMDb ratings.

* Evaluate the model performance.

* Save the trained model for future use.

📊 Dataset Description

The dataset used for this project contains the following columns:

| Column Name                                                      | Description

Name:-                                                           Title of the movie

Year:-                                                           Release year of the movie

Duration:-                                                       Duration of the movie (in minutes)

Genre:-                                                          Genre of the movie (Drama, Comedy, Romance, etc.)

Rating:-                                                         IMDb rating of the movie
 
Votes:-                                                          Number of user votes received  

Director:-                                                       Name of the movie's director

Actor 1:-                                                        Lead actor/actress in the movie

Actor 2:-                                                        Second lead actor/actress in the movie

Actor 3:-                                                        Third lead actor/actress in the movie


🔎 Preprocessing Steps

* Handled missing values by filling with either mean (numeric columns) or 'Unknown' (categorical columns).

* Converted columns such as Votes to numeric.

* Applied Label Encoding to categorical columns (Genre, Director, Actor 1, Actor 2, Actor 3).

🛠 Feature Engineering

* Director Success Rate: Average rating of each director's movies.

* Genre Popularity: Average rating for each genre.

* Movie Age: Calculated using 2024 - Year.

* Cast Average Rating: Average rating of movies by each cast member.

✅ Model Development

* Used RandomForestRegressor with n_estimators=200 and random_state=42.

* Split data into training (80%) and testing (20%).

* Evaluated model using metrics: R² Score, MAE (Mean Absolute Error), and RMSE (Root Mean Squared Error).

📈 Evaluation Results

* Example results from the model:

* R² Score: ~0.62

* MAE: ~0.32

* RMSE: ~0.60

📂 Project Structure

movie-rating-prediction/

│
├── data/

│   └── imdb_india_movies.csv

│
├── notebooks/

│   └── movie_rating_prediction.ipynb

│
├── model/

│   └── movie_rating_predictor.pkl

│
├── README.md
│

✅ How to Run

* Clone the repository.
* Install Libraries.
* Run the Jupyter notebook or script.
* The trained model will be saved as movie_rating_predictor.pkl.

🛠 Tools & Libraries Used

* Python
* Pandas, Numpy
* Matplotlib, Seaborn
* Scikit-learn
* Joblib

🙌 Conclusion

This project successfully demonstrates a machine learning workflow for predicting movie ratings with feature engineering, preprocessing, model training, evaluation, and deployment.

