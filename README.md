# ML-Classification-with-FLAML-LightGBM-Optuna

## Introduction

This repository is an advanced Machine Learning Classification Studio that utilizes the FLAML, LightGBM, and Optuna libraries throughout the entire machine learning product life cycle. The code is organized into Jupyter notebooks, with each notebook focusing on a specific technique or dataset.

Our goal is to provide a comprehensive set of examples that can help you effectively tackle classification problems and guide you through each step of the process, from data analysis to deploying and monitoring the final machine learning model. We believe that understanding the entire process of developing a machine learning model, from data cleaning and preprocessing to deploying and maintaining the model in a production environment, is crucial for creating successful machine learning products.

Furthermore, we aim to provide a solution for resolving classification problems for tabular datasets using advanced techniques while streamlining the workflow through automation of common tasks in the machine learning product life cycle. The FLAML library provides automatic machine learning (AutoML) capabilities to help you efficiently explore the search space of various machine learning algorithms and hyperparameters. The LightGBM library is a high-performance gradient boosting framework that can efficiently handle large-scale data and supports various learning objectives and evaluation metrics. The Optuna library provides a flexible and scalable framework for hyperparameter optimization.

Whether you're a beginner or an experienced practitioner in machine learning, we believe that this repository can provide you with valuable insights and guidance on how to effectively apply these advanced tools in solving classification problems. We encourage you to explore the notebooks, experiment with the code, and let us know if you have any questions or feedback.

## Dataset

The dataset used in this project is the "Airline Passenger Satisfaction" dataset from Kaggle. This dataset contains information about airline passengers' satisfaction levels with various aspects of their travel experience, such as in-flight service, cleanliness, online support, and departure/arrival time satisfaction.

With over 129,000 rows and 24 columns, this dataset provides a rich source of data for in-depth analysis and modeling. It also presents a variety of challenges related to data preprocessing, such as handling missing values and encoding categorical variables. However, with careful preprocessing, this dataset can yield valuable insights into factors that influence airline passenger satisfaction levels and enable airlines to make data-driven decisions to improve their customers' experiences.

## 1 - Preprocessing dataset

The first step in solving a machine learning classification problem is to preprocess the dataset. Using the Pandas library, we load the "airline_passenger_satisfaction.csv" file and then analyze it to determine which features are numerical, which are categorical, and which is the target variable. It's essential to identify and handle any missing values or NaNs in the dataset, either by replacing or removing them.

The next critical step is to transform categorical features into numerical ones using OneHotEncoder and scale numerical features using Standarscaler. Additionally, we need to transform the classes present in the target variable into numerical labels using LabelEncoder. After completing these preprocessing steps, we can save the datasets ready for use by the Sklearn machine learning libraries.

## 2 - ML Model Lab

Here, we explore the performance of gradient boosting libraries in machine learning, such as XGBoost, CATboost, and LightGBM, on tabular datasets. We use FLAML to automatically optimize the hyperparameters of these models for our dataset, which allows us to obtain the best possible configurations without manual tuning.

After the hyperparameters are optimized, we instantiate multiple classification models and fit them to our training set. Once the models are trained, we evaluate their performance on the testing set using metrics such as precision, recall, accuracy, and F1 score. To compare the performance of different models, we plot precision and recall curves at different thresholds.

By going through this process, we can identify the most promising machine learning models for our dataset and select the one that performs best in terms of precision, recall, and overall accuracy. With the best-performing model selected, we can build a reliable and effective machine learning classifier that can make predictions in production environments.

## 3- Tuning LightGBM hyperparameters with Optuna

We fine-tune the hyperparameters of LightGBM, a gradient boosting library of machine learning, using Optuna. After identifying LightGBM as the best performer, we create a hyperparameter space with a range of values centered on those obtained automatically by FLAML to make the search more efficient. In this case, we ask Optuna to find the best hyperparameters to maximize the F1 score. Once we obtain the best set of hyperparameters, we apply them to the previous notebook "ML Model Lab" and evaluate the improvement in classification performance. This approach can help us optimize our model and achieve better results for our machine learning classification problem.

## 4 - Pipeline for production

In the final step, we reconstruct the pipeline of our model by including the necessary input data transformations and the previously optimized model with the best hyperparameters. We can then train the pipeline one last time on the X and y data and save it to disk using the joblib library.

At this point, our classification predictive model is ready to be integrated into any production application. The pipeline can be easily deployed to a cloud environment, such as Amazon Web Services (AWS), Google Cloud Platform (GCP) or Microsoft Azure, or to an on-premises server. Once deployed, the pipeline can receive input data and provide predictions in real-time, enabling us to make accurate decisions and automate processes based on the predictions made by our machine learning model.

## Final notes

You can follow the four-step approach outlined in the Jupyter Notebook worksheets for any binary or multi-classification work using the best gradient boosting models like XGBoost, CATBoost, and LightGBM. By following these steps, you can achieve highly satisfactory results for your machine learning classification problem.

Happy classifying!




