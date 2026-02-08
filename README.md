🚗 Car-Price-Prediction-Recommendation-System-AWS-ML-
An end-to-end machine learning web application that predicts vehicle prices and recommends similar cars based on user input.

📌 Project Overview
This system uses an XGBoost regression model trained on car listing data to estimate vehicle prices. The model is deployed using AWS SageMaker, and a Django web application interacts with the deployed endpoint to generate real-time predictions.
The system also recommends cars within a similar price range and stores prediction history in a database.

User inputs vehicle details through a Django web interface. The application sends the data to a deployed XGBoost model on AWS SageMaker for real-time price prediction. Based on the predicted price, the system retrieves similar vehicles from a dataset stored in Amazon S3 and displays the top recommendations. All predictions are stored in PostgreSQL for tracking and analysis.

⚙️ Features

Real-time car price prediction

Top 5 similar car recommendations

Prediction history storage

Cloud-based model deployment

Full-stack ML integration

🛠 Tech Stack

Python

XGBoost

AWS SageMaker

Amazon S3

Django

PostgreSQL

HTML/CSS

🧠 Workflow

Data preprocessing and feature engineering

Model training using XGBoost

Model deployment to SageMaker endpoint

Django app sends user input to endpoint

Predictions and recommendations are displayed and saved

🚀 Future Improvements

Add user authentication

Improve recommendation algorithm

Deploy full application to cloud hosting

Add model monitoring and retraining pipeline
