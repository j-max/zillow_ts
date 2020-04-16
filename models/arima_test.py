import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import itertools
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

with open('../data/processed/chi_mil_mnsp.p','rb') as read_file:
    chi, mil, mnsp = pickle.load(read_file)

# establish test set as the last 7 instances
size = 7
train, test = chi[:-size], chi[-size:]
test = [x for x in test.values]
history = [x for x in train.values]

def predict_ar1(coef, history):
    '''
    For an AR1 model, the prediction is the coeficient minus times the lagged value
    History is either actual history or the residuals, if using MA model

    '''
    y_hat = coef*history[-1]

# fit an AR model using ARIMA and calculate the prediction:

y_hats = []
for _ in range(size):
    
    model = ARIMA(history, order=(1,0,1))
    model_fit = model.fit()
    ar_coef = model_fit.arparams
    ma_coef = model_fit.maparams

    residuals = model_fit.resid 

    y_hat = ar_coef*history[-1] + ma_coef*residuals[-1]
    y_hats.append(y_hat)

    # extend history to include the new test for the next step ahead and new prediction
    history.append(test[_])

# convert y_hats/test to array to make calcuations broadcastable
y_hats = np.array(y_hats)
test = np.array(test)
rmse = np.sqrt(mean_squared_error(test, y_hats))
print(rmse)
