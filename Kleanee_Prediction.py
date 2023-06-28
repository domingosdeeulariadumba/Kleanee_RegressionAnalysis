# -*- coding: utf-8 -*-
"""
Created on Fri Jun 2 23:38:08 2023

@author: domingosdeeulariadumba
"""

%pwd



""" Importing the required libraries """


# For EDA and Plotting

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as mpl
mpl.style.use('ggplot')


# For selecting ML candidate models, and evaluate the chosen one

from sklearn.model_selection import train_test_split as tts, cross_validate as cv
from sklearn.linear_model import LinearRegression as lreg
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.pipeline import make_pipeline as mpp
from sklearn.metrics import mean_squared_error, r2_score
import sklearn as skl

# To summarize the chosen model (especially to check the p_value)

import statsmodels.api as sm

# To save and load the trained/final model

import joblib


# To ignore warnings about compactibility issues and so on

import warnings
warnings.filterwarnings('ignore')



"""" EXPLORATORY DATA ANALYSIS """


# Loading the dataset

df_kln= pd.read_excel("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/kleanee_Prediction/Kleanee_dataset.xlsx")

# Displaying the first and last ten records

df_kln.head(10)

df_kln.tail(10)


# Dropping the ID column, since it's irrelevant for our analysis

df_kln=df_kln.drop('ID', axis=1)


# Displaying and saving the scatter plot of the data relationship
sb.scatterplot(data=df_kln, x='Age', y='Spending Score (1-100)')
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/kleanee_Prediction/SSvs.Age_ScatterPlot.png")
mpl.close()

# Defining the independent and dependent variables

x=df_kln['Age'].to_numpy()

y=df_kln['Spending Score (1-100)']


# Splitting the data into train and test sets

X=x.reshape(-1,1)

X_train, X_test, y_train, y_test = tts(X,y, test_size=0.2, random_state=1)


# Checking the shape of the train and test sets

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Evaluation of the candidate models

"""
Looking to the scatter plot, we may firstly choose two models to estimate the
relationship between the Spending Score and the Customers Age. To avoid overfitting
or strange behaviors on predicting unseen data, we'll evaluate and then select one
from two candidate models: Linear and Polynomial (quadratic)
""" 

poly_reg=mpp(pf(), lreg(fit_intercept=False))
lin_reg=lreg()


# K- folds cross validation for selecting the model with the best average performance

scoring = "neg_root_mean_squared_error"
poly_reg_scores=cv(poly_reg, X_train, y_train, scoring=scoring, return_estimator=(True))
lin_reg_scores= cv(lin_reg, X_train, y_train, scoring=scoring, return_estimator=(True))


# Printing the coefficient of Linear Regression

poly_reg_scores['estimator'][0].steps[1][1].coef_
lin_reg_scores['estimator'][0].intercept_, lin_reg_scores['estimator'][-1].coef_

    """
    Then, the fitted polynomial (1) and linear (2) models may be written as below:
    
        (1) Spending Score = 86.061-1.191*Age+0.0057*Age^2

        (2) Spending Score = 76.942-0.618*Age
    """

# Finding which model performed best on average

    """
    As we are using Negative Root Mean Squared Error as the performance 
    comparison metric, the best model has to be the one with the highest score
    (which is closest to 0)
    """
    
models=['Quadratic','Linear']
avgscores= [poly_reg_scores['test_score'].mean(), lin_reg_scores['test_score'].mean()]
bestmodel=''
greater=-1000

for i,j in zip (range (len(avgscores)), range (len(models))):
  if avgscores[i]>greater:
      greater=avgscores[i]
      bestmodel=models[j]
print('The best model is the %s Line with an average negative RMSE score of %r' %(bestmodel,-greater))
      
 
# Plotting and saving the regression models line graph
    
    """
    Before (re)training the chosen model we can observe the two lines on the
    scatter plot to better understand how each one fits the data points
    """

sb.scatterplot(data=df_kln, x='Age', y='Spending Score (1-100)')
mpl.plot(X,poly_reg_scores['estimator'][0].predict(X))
mpl.plot(X, lin_reg_scores['estimator'][0].predict(X))
mpl.ylabel('Spending Score')
mpl.xlabel('Age')
mpl.xlim(15,75)
mpl.ylim(0,100)
mpl.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/kleanee_Prediction/SSvs.Age_ScatterPlot_withmodels.png")
mpl.close()

# Retraining the chosen model
    
    """
    Since the model changes in each cross validation iterations, it remains 
    untrained, so it is reasonable to reset the model (base) and clone a
    new one to be trained
    """
LINREG=skl.base.clone(lin_reg)
LINREG.fit(X_train, y_train)


# Evaluating the chosen model performance on test_set ('unseen data')

y_pred=LINREG.predict(X_test)

print('Test set (RMSE):', mean_squared_error(y_test, y_pred, squared=False))
print('Mean Validation (RMSE):', -lin_reg_scores['test_score'].mean())

    """
    The first result printed above shows how the model will perform if we deploy it.
    The second, comparing to the first, shows how well the model as been traind by
    looking to the RMSE difference
    """

print('NRMSE:', (mean_squared_error(y_test, y_pred, squared=False)/(y.max()-y.min())))

    """
    Above we print the normalized RMSE to have a better comprehension regarding the 
    dimension of this value on its own scale
    """

# Addicional analysis

print('Coefficient of determination (R^2):', r2_score(y_test, y_pred))
    
      """
    Conversely, before we decide to deploy the model it is also a good practice
    to find out the statistcal significance of the estimation discussed above.
    It will provide usefull information to ponder if it is better having or not
    the trained model. We'll check the p-value.
    Concerning this point, since SciKit_learn does not provide an exhaustive summary
    of some crucial metrics like p-value, additionally, we'll use the statsmodel
    module to do so.
    
    (we should've also apply this approach to fit the model as was done using 
    SciKit-learn and reduce some steps)
    """

X_train_=X_train
y_train_=y_train
X_train_=sm.add_constant(X_train_)
Reg=sm.OLS(y_train_, X_train_).fit()
Reg.summary()


# Rewriting the equation model

LINREG.intercept_, LINREG.coef_

    """
    Spending Score = 75.098-0.612*Age
    
    Based on the model summary, we notice that there are 95% of confidence that
    the mean change in the Spending Score related to a customer average variation
    of ten (10) years, coming to the mall, results in a decrease of the Spending
    Score by around eight (8) and three (3) points approximately. 
    """

# Saving the model

joblib.dump(LINREG,"C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/kleanee_Prediction/SS_Vs_Age_Prediction.sav")


# Loading the model for operational use (first printing )

loaded_model=joblib.load("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/kleanee_Prediction/SS_Vs_Age_Prediction.sav")
result=(loaded_model.score(X_test, y_test))
print(result)
    

# Making predictions with the saved model

Xnewval=[[76],[19],[23],[75], [18]]

ynewval=loaded_model.predict(Xnewval)

for j in range (len(Xnewval)):
    print(Xnewval[j], ynewval[j])

___________________________________ end _______________________________________