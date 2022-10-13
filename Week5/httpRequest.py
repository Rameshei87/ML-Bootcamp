#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

user = {
    "tenure": 1,
    "monthlycharges": 700.75,
    "totalcharges": 700.75,
    "seniorcitizen": 0,
    "gender": "male",
    "partner": "no",
    "dependents": "yes",
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "no",
    "onlinesecurity": "no_internet_service",
    "onlinebackup": "no_internet_service",
    "deviceprotection": "no_internet_service",
    "techsupport": "no_internet_service",
    "streamingtv": "no_internet_service",
    "streamingmovies": "no_internet_service",
    "contract": "month-to-month",
    "paperlessbilling": "no",
    "paymentmethod": "mailed_check"
    }


res = requests.post(url, json=user).json()

print(f'Probability of churning: {res}')

if res['churn'] == True:
    print('Sending promo email to customer')
else:
    int('Not sending promo email to customer')




