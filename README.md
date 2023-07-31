# Introduction
In an increasingly connected world, telecommunications companies play a vital role in people's lives. However, competition is fierce and retaining customers has become a difficult task. Churn or customer abandonment is a common problem in the industry, and can be costly for the company in terms of lost revenue and reputation. Therefore, it is crucial for telecommunications companies to identify customers who are at risk of leaving and take proactive measures to keep them satisfied. In this project, machine learning techniques will be used to predict churn and help companies retain their customers.


## Installing the dependencies
pip -r requirements.txt


## Executing the Script
## Console python
 streamlit run ChurnPrediction.py

## Viewing Your Streamlit App
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.23:8501


 ## Description
 ## Streamlit
The project focuses on predicting customer churn in a telecommunications company. It utilizes the Streamlit framework to create an interactive user interface that allows users to input customer information and obtain a prediction of whether the customer will churn or not.
The project starts by loading a pre-trained model and a CSV data file containing customer information. It then performs some data transformations, such as converting the target variable "Churn" into binary values.
The user interface displays some general metrics about customer retention and churn. It also provides a sidebar where users can explore different statistics and visualizations related to customer churn. Some of the available visualizations include bar charts and pie charts showing the distribution of churned customers based on different features such as gender, age, partnership status, dependents, phone and internet services, among others.
In summary, this project utilizes machine learning and interactive visualizations to predict and analyze customer churn in a telecommunications company.


## Descripcion
## Jupyter
The project consists of a Machine Learning model to predict whether a telecommunications company's customer will churn or not. To do this, a dataset is used that contains information about customers, such as their gender, age, whether they have a partner or dependents, whether they use phone and internet services, among others.
An exploratory data analysis is performed to better understand the characteristics of customers who churn and coding and vectorization techniques are used to convert categorical variables into numerical ones and train the model.
A logistic regression model is used to make the prediction and its performance is evaluated on a validation dataset. Finally, a web interface is implemented in which the user can enter a customer's data and obtain a prediction on whether they are likely to churn or not.