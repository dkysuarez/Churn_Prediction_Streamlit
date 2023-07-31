import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import plotly.express as px


model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

st.set_page_config(page_title="Churn Model",
                   page_icon=":bar_chart:",
                   layout='wide'
                   )

# Remove menu
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)


# data preparation
missing_value = ["N/a", "na", np.nan]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv',
                 na_values=missing_value)
df.dropna(inplace=True)

dfEst = df
def statics():
 dfEst['Churn'].replace({'Yes': 1, 'No': 0}, inplace=True)

 col1, col2 = st.columns(2)
 num_retained = dfEst[df.Churn == 0.0].shape[0]
 num_churned = dfEst[df.Churn == 1.0].shape[0]
 retined = num_retained / (num_retained + num_churned)*100
 churned = num_churned / (num_retained + num_churned) * 100
 col1.metric("Customers Stayed with the company:", retined, "%")
 col2.metric("Customers Left with the company:",
            churned, "%", delta_color='inverse')

statics()
st.sidebar.info('This app is created to predict Customer Churn')



# Sidebar Stadistics
if st.sidebar.checkbox("Statics"):
    if st.sidebar.button("Head"):
        st.write(df.head(60))
    if st.sidebar.button("Tail"):
        st.write(df.tail(60))
    if st.sidebar.button("Columns"):
        st.write(df.columns)
    if st.sidebar.button("Describe"):
        st.write(df.describe(include='all'))
    if st.sidebar.button("Shape"):
        st.write("Number of Rows", df.shape[0])
        st.write("Number of Columns", df.shape[1])


# plot churned
churnedCharts = st.sidebar.checkbox('Churned Charts')
if churnedCharts:
    opt = st.sidebar.radio('Churned:', [
        'Bar', 'Pie'
    ])
    if opt == 'Bar':
        plotchurned = df['Churn'].value_counts()
        st.bar_chart(plotchurned)
    if opt == 'Pie':

        # pie churn
        piechar = px.pie(df, names="Churn", title="Pie Churn")
        st.plotly_chart(piechar)


load = st.sidebar.checkbox('Load Charts')
if load:
    opt = st.sidebar.radio('Churned by type:', [
                           "Gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                           "InternetService", "DeviceProtection", "TechSupport", "StreamingTV"])
    if opt == 'Gender':
        st.subheader("Gender Analysis:")
        st.write("-There are 3,488 women in the sample, of which 2,549 do not churn and 939 do.")
        st.write("-The average number of women who do not churn is 73.08%.")
        st.write("-There are 3555 men in the sample, of which 2625 do not churn and 930 do.")
        st.write("-The average male churn rate is 73.83%")     
        # plot churned clients by gender
        fig = plt.figure(figsize=(9, 7))
        sns.countplot(x='gender', hue='Churn', data=df)
        st.pyplot(plt)
        

    if opt == 'SeniorCitizen':
        st.subheader("Senior Citizen Analysis:")
        st.write("-Based on the data provided, it can be observed that the churn rate of customers churn rate of non-seniorcitizen clients is lower than that of non-seniorcitizen clients.")
        st.write("-Of the 1142 seniorcitizen clients there is an average of 58.31% who do not churn, in contrast to the 5901 non-seniorcitizen clients who do not churn, who have an average of 76.39%.")
        st.write("-This suggests that the company could focus its efforts on improving the retention of seniorcitizen customers, as the seniorcitizen customers, as they have a higher risk of churn")
        # plot churned clients by SeniorCitizen
        plt.figure(figsize=(15, 8))
        sns.countplot(x='SeniorCitizen', hue='Churn', data=df)
        st.pyplot(plt)


    if opt == 'Partner':
        st.subheader("Partner Analysis:")
        st.write("-There are currently 3402 customers who are partners of the company, of which 669 have churned, giving an average no churn rate of average no churn rate of 80.33%.")
        st.write("-On the other hand, the 3641 customers who are not partners have an average no churn rate of 67.04%.")
        st.write("-It can be seen that the churn rate of non-partner customers is much higher than that of partner customers.")
        # plot churned clients by Partner
        plt.figure(figsize=(18, 10))
        sns.countplot(x='Partner', hue='Churn', data=df)
        st.pyplot(plt)

    if opt == 'Dependents':
        st.subheader("Dependents Analysis:")
        st.write("-Of the 2110 customers who make depents, only 326 churn, giving an average no churn rate of 84.54%.")
        st.write("-Of the 4933 customers who have no dependents, the average no churn rate is 68.72%.")
        # plot churned clients by Dependents
        plt.figure(figsize=(15, 10))
        sns.countplot(x='Dependents', hue='Churn', data=df)
        st.pyplot(plt)

    if opt == 'PhoneService':
        st.subheader("PhoneService Analysis:")
        st.write("-Of the 682 customers without PhoneService, 170 customers have churned, giving an average of of 75.07%.")
        st.write("-Of the 6361 customers who have PhoneService, 4662 have no churn, giving an average of 73.29% no churn")
        # plot churned clients by phoneservice
        plt.figure(figsize=(15, 10))
        sns.countplot(x='PhoneService', hue='Churn', data=df)
        st.pyplot(plt)

    if opt == 'InternetService':
        st.subheader("InternetService Analysis:")
        st.write("-Based on the data provided, there are a total of 7043 customers, of which 2421 have DSL services, 3096 customers have fibre and 1526 have no internet service.")
        st.write("-Of the customers who have DSL service, the average no churn rate is 81.04%, while of those who do not have Fibre Optic service, the average no churn rate is 58.10%. On the other hand, 92.59% of those who do not have Internet service do so.")
        st.write("-Clearly, more attention should be paid to fibre optic subscribers, as they have a higher average churn rate.")
        # plot churned clients by InternetService
        plt.figure(figsize=(15, 10))
        sns.countplot(x='InternetService', hue='Churn', data=df)
        st.pyplot(plt)

    if opt == 'DeviceProtection':
        st.subheader("Device Protection Analysis:")
        st.write("-Regarding the device protection factor, there are 3095 devices that have no DeviceProtection, giving an average no churn rate of 60.87%. do not churn and 939 do.")
        st.write("-1,526 customers have no Internet service, of which 113 have churned, giving an average no churn rate of 92.59% churn rate.")
        st.write("-Of the 2422 customers with device protection, 1877 have not churned and 545 have churned, giving an average no churn rate of 77.49%.")
        st.write("-The highest average churn rate belongs to those who do not have device protection, the company should focus on this area to avoid losing customers. ")     
        # plot churned clients by DeviceProtection
        plt.figure(figsize=(15, 10))
        sns.countplot(x='DeviceProtection', hue='Churn', data=df)
        st.pyplot(plt)

    if opt == 'TechSupport':
        st.subheader("TechSupport Analysis:")
        st.write("-Of the 7043 customers, 2044 subscribed to the TechSupport service, 1734 did not churn, 310 have churned, giving an average churn rate of 84.83%.")
        st.write("-1526 of the company's customers do not have Internet service, of which 1413 have not churned and 113 have churned, giving an average no churn rate of 92%.")
        st.write("-3473 customers have not taken up TechSupport, 1446 customers have churn on record giving an average no churn rate of 58.36%.")
        st.write("-Customers who have not subscribed to TechSupport have a high average churn rate, so they are more likely to churn")     
        # plot churned clients by TechSupport
        plt.figure(figsize=(15, 10))
        sns.countplot(x='TechSupport', hue='Churn', data=df)
        st.pyplot(plt)

    if opt == 'StreamingTV':
        st.subheader("StreamingTV Analysis:")
        st.write("-Regarding the StreamingTV service, 2810 customers have not subscribed to this service. Of these, 1868 have no churn, giving an average of 66.47% no churn.")
        st.write("-On the other hand, the 1526 customers who do not have an Internet service have an average of 92.59% of no churn.")
        st.write("-Finally, the 2707 customers who have subscribed to the StreamingTV service have an average no churn rate of 69.92%.")
        st.write("-Both those who have subscribed to the StreamingTV service and those who have not have a very high average churn rate, so strategies to retain these customers should be considered")    
        # plot churned clients by StreamingTV
        plt.figure(figsize=(15, 10))
        sns.countplot(x='StreamingTV', hue='Churn', data=df)
        st.pyplot(plt)


def main():
    st.title("Predicting Customer Churn")
    gender = st.selectbox('Gender:', ['male', 'female'])
    seniorcitizen = st.selectbox(' Customer is a senior citizen:', [0, 1])
    partner = st.selectbox(' Customer has a partner:', ['yes', 'no'])
    dependents = st.selectbox(' Customer has  dependents:', ['yes', 'no'])
    phoneservice = st.selectbox(' Customer has phoneservice:', ['yes', 'no'])
    multiplelines = st.selectbox(' Customer has multiplelines:', [
                                 'yes', 'no', 'no_phone_service'])
    internetservice = st.selectbox(' Customer has internetservice:', [
                                   'dsl', 'no', 'fiber_optic'])
    onlinesecurity = st.selectbox(' Customer has onlinesecurity:', [
                                  'yes', 'no', 'no_internet_service'])
    onlinebackup = st.selectbox(' Customer has onlinebackup:', [
                                'yes', 'no', 'no_internet_service'])
    deviceprotection = st.selectbox(' Customer has deviceprotection:', [
                                    'yes', 'no', 'no_internet_service'])
    techsupport = st.selectbox(' Customer has techsupport:', [
                               'yes', 'no', 'no_internet_service'])
    streamingtv = st.selectbox(' Customer has streamingtv:', [
                               'yes', 'no', 'no_internet_service'])
    streamingmovies = st.selectbox(' Customer has streamingmovies:', [
                                   'yes', 'no', 'no_internet_service'])
    contract = st.selectbox(' Customer has a contract:', [
                            'month-to-month', 'one_year', 'two_year'])
    paperlessbilling = st.selectbox(
        ' Customer has a paperlessbilling:', ['yes', 'no'])
    paymentmethod = st.selectbox('Payment Option:', [
                                 'bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check', 'mailed_check'])
    tenure = st.number_input(
        'Number of months the customer has been with the current telco provider :', min_value=0, max_value=240, value=0)
    monthlycharges = st.number_input(
        'Monthly charges :', min_value=0, max_value=240, value=0)
    totalcharges = tenure*monthlycharges
    output = ""
    output_prob = ""
    input_dict = {
        "gender": gender,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "tenure": tenure,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges
    }

    if st.button("Predict"):
        X = dv.transform([input_dict])
        y_pred = model.predict_proba(X)[0, 1]
        churn = y_pred >= 0.5
        output_prob = float(y_pred)
        output = bool(churn)
    st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))


if __name__ == '__main__':
    main()
