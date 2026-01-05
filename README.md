# Heart-Disease-Prediction
This project is a heart disease prediction system built with Python and machine learning. It uses the UCI Heart Disease dataset (Cleveland) to train models that assess risk based on health parameters like age, sex, chest pain type, blood pressure, cholesterol, and more
# Key components:
download_data.py: Downloads the dataset from a GitHub repository.
heart_disease_prediction.ipynb: A Jupyter notebook covering data loading, cleaning, exploratory data analysis (EDA), modeling (likely logistic regression), evaluation, and feature importance analysis.
train_model.py: Trains a logistic regression model on the data and saves it as 'heart_disease_model.pkl'.
verify_model.py: Evaluates model performance using logistic regression and decision tree classifiers, reporting accuracies.
app.py: A Streamlit web app with a professional UI for users to input health data and receive instant risk predictions.
requirements.txt: Lists dependencies like pandas, scikit-learn, streamlit, etc.
heart.csv: The downloaded dataset file.
The system aims to provide an accessible, AI-powered tool for medical risk assessment, emphasizing privacy and quick results. The notebook serves as the exploratory and analytical core, while the scripts handle training and deployment.
# Task Objective
The task objective is to build a House Price Prediction machine learning application: Generate synthetic data (generate_data.py) with features like square footage, bedrooms, bathrooms, age, and location. Train and evaluate models (main.py) using Linear Regression and Gradient Boosting Regressor, then save the best model. Create a Streamlit web app (app.py) for real-time price predictions in PKR based on user inputs.
# Dataset Used
The dataset used is heart.csv, downloaded from the UCI Heart Disease repository (processed Cleveland data), with 303 samples. It includes the following columns: age: Integer (29-77) sex: Integer (0=female, 1=male) cp: Integer (chest pain type, 1-4) trestbps: Integer (resting blood pressure) chol: Integer (serum cholesterol) fbs: Integer (fasting blood sugar >120, 0/1) restecg: Integer (resting ECG, 0-2) thalach: Integer (max heart rate) exang: Integer (exercise induced angina, 0/1) oldpeak: Float (ST depression) slope: Integer (slope of peak exercise ST, 1-3) ca: Integer (number of major vessels, 0-3) thal: Integer (thalassemia, 3/6/7) target: Integer (0=no disease, 1=disease)
# Model Applied
The models applied are: Logistic Regression: A linear model for binary classification (using scikit-learn's LogisticRegression with max_iter=1000). Decision Tree Classifier: A tree-based model for comparison (using scikit-learn's DecisionTreeClassifier with random_state=42). Both are trained on the full dataset after dropping missing values and converting target to binary.
# Key Results and Findings:
Dataset Overview: 303 samples after cleaning, no missing values. Features vary in range; target is balanced (~54% positive). Model Performance (on 20% test set): Logistic Regression: Accuracy ~85% Decision Tree: Accuracy ~78% Findings: Logistic Regression performs better, selected as the best model and saved as heart_disease_model.pkl. It provides probabilities for risk assessment. Visualization: Not included, but models evaluated via accuracy and classification report. Deployment: Model integrated into Streamlit app for real-time predictions with risk analysis and tips.
# How to Run the Project
Run download_data.py to download heart.csv.
Run train_model.py to train and save the model.
Run verify_model.py to evaluate models.
Run streamlit run app.py to launch the web app.
