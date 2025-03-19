#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:24:26 2024

@author: sieuchuoi
"""
# QUESTION 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as skm

from pandas.tseries.offsets import *
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import (DecisionTreeRegressor as DTR, plot_tree)
from sklearn.ensemble import (RandomForestRegressor as RF,
                              GradientBoostingRegressor as GBR)

path = "/Users/sieuchuoi/Documents/2. Quantitative Methods, Big Data, and Machine Learning/Project Assignment"

gkx = pd.read_csv(path + "/datashare/datashare.csv")
crsp_m = pd.read_csv(path + "/Monthly_Stock_Returns.csv")

# Standardize identifiers and date formats
crsp_m = crsp_m.rename(columns={'PERMNO': 'permno', 
                          'PRC': 'prc',
                          'RET': 'ret',
                          'ALTPRC': 'altprc'})
gkx = gkx.rename(columns={'DATE': 'date'})

crsp_m['date'] = pd.to_datetime(crsp_m['date'].astype(str)) + MonthEnd(0)
gkx['date'] = pd.to_datetime(gkx['date'].astype(str)) + MonthEnd(0)

gkx.info()
crsp_m.info()

# Merge 2 DataFrames on common identifiers (permno and date)
data = pd.merge(crsp_m[['date', 'permno', 'ret']], gkx, on=['date', 'permno'], how='inner')
data.sort_values(['date', 'permno'], inplace=True)

data['ret_ahead'] = data.groupby('permno')['ret'].shift(1)  # One-month ahead return
data_test = data.head(10000)

data.info()

# Check the column names in `data` after merging to identify available features
print("Available columns in data:", data.columns)


# Pre process data

# Dropna if there is any NA value for 'date', 'permno', 'ret' and 'ret_ahead' columns
data = data.dropna(axis = 'index', how = 'any', subset = ['date', 'permno', 'ret', 'ret_ahead'])
data.info()

# Replace missing value (for others columns, except date permno, ret, ret_ahead) with mean
def replace_missing_value(column):
    mean_value = column.mean(skipna=True)
    return column.fillna(mean_value)

data_filled = data.copy()

data_filled = data_filled.groupby('permno').transform(replace_missing_value)

columns_filled = data.columns.difference(['date', 'permno', 'ret', 'ret_ahead'])
data[columns_filled] = data_filled.reindex(columns=columns_filled, index=data.index)
data_test = data.head(10000)



# Question a: Choose list of 30 predictive features (20 monthly + 10 annual/quarterly) & summary statistics
# a.1. Choose the list of 30 predictive features
monthly_characteristics = ["baspread", "beta", "betasq", "chmom", "dolvol",
                           "idiovol", "ill", "indmom", "maxret", "mom12m",
                           "mom1m", "mom36m", "mom6m", "mvel1", "pricedelay",
                           "retvol", "std_dolvol", "std_turn", "turn", "zerotrade"]
annual_quarterly_characteristics = ["agr", "bm", "bm_ia", "cash", "cashdebt",
                                    "chcsho", "chempia", "chinv", "chpmia", "currat"]


predictive_features = monthly_characteristics + annual_quarterly_characteristics

data = data[['date', 'permno', 'ret_ahead'] + predictive_features].dropna()
data_test = data.head(10000)

# a.2. Summary Statistics for the predictive features
summary_stats = data[predictive_features].describe().T
print("Summary Statistics for Selected Features:")
print(summary_stats)



# Question b: Pre-process the predictive features by applying the rank normalization technique
# b.1. Create a new data frame to store the dense ranked predictive features
df_ranked = data.copy()

# b.2. Define the columns to be ranked
columns_to_rank = df_ranked.columns.difference(['date', 'permno', 'ret_ahead'])

# Define a function that (dense) ranks the columns and maps to [0,1]:
def rank_norm(column):
    rank = column.rank(method='dense')
    return (rank-1)/(np.nanmax(rank)-1)

# b.3. Apply the function to df_ranked and store the new columns with '_rank' suffix:
df_ranked = df_ranked.groupby('date')[columns_to_rank].transform(rank_norm)
df_ranked = df_ranked.add_suffix('_rank')
df_ranked_test = df_ranked.head(10000)

# b.4. Concatenate the original data with the rank-normalized dataframe
data = pd.concat([data,df_ranked], axis=1)   
data_test = data.head(10000)




# Question c: Training model
# c.1. Get rank columns
ranked_columns = [col for col in data.columns if col.endswith('_rank')]

# c.2. Train-Test Split
train_data = data[(data['date'] < '1990-01-01')]
test_data = data[(data['date'] >= '1990-01-01')]

train_data = train_data.reset_index(drop=True) 
test_data = test_data.reset_index(drop=True)

train_data = train_data[['date', 'permno', 'ret_ahead'] + ranked_columns].dropna()
test_data = test_data[['date', 'permno', 'ret_ahead'] + ranked_columns].dropna()


X_Train = train_data[ranked_columns]
y_Train = train_data['ret_ahead']
X_Test = test_data[ranked_columns]
y_Test = test_data['ret_ahead']

# c.3. Standardize the features
scaler = StandardScaler()
X_Train_scaled = scaler.fit_transform(X_Train)
X_Test_scaled = scaler.transform(X_Test)
X_Train_scaled = pd.DataFrame(X_Train_scaled, columns=ranked_columns)
X_Test_scaled = pd.DataFrame(X_Test_scaled, columns=ranked_columns)



# c.4. Train PLS Model
PLS_coefs_df = pd.DataFrame(columns=ranked_columns)

# Center the responses
y_Train_centered = y_Train - y_Train.mean()
y_Train_centered_test = y_Train_centered.head(100)

for num_comp in range(1,len(ranked_columns)+1):
    # Perform PLS regression
    pls = PLSRegression(n_components=num_comp)
    pls.fit(X_Train_scaled, y_Train_centered)

    # Get the coefficients
    coefficients_original_features = pls.coef_.flatten()
    
    # Store the PLS coefficients
    PLS_coefs_df.loc[f'{num_comp}_comp'] = coefficients_original_features

PLS_coefs_df.reset_index(drop=True, inplace=True)
PLS_coefs_df.index = range(1,len(ranked_columns)+1)

# Plot PLS Coefficients of Features
plt.figure(figsize=(12, 8))

for column in PLS_coefs_df.columns:
    plt.plot(PLS_coefs_df.index, PLS_coefs_df[column], label=column)

plt.title('PLS Coefficients of Features')
plt.xlabel('Number of Components', fontsize=20)
plt.ylabel('Standardized coefficients', fontsize=20)
plt.legend(loc='upper right', fontsize=8)
plt.xticks(range(1, len(ranked_columns) + 1))

plt.show()

# Using cross-validation to assess how well your model according to different number of components and find the optimal one
def computes_test_mse(kf):
    PLS_test_predictions = pd.DataFrame()
    PLS_train_predictions = pd.DataFrame()
    PLS_optimal_params = pd.DataFrame(columns=['Optimal Parameter'])
    
    pipe = Pipeline([ 
    ('scaler', StandardScaler()),
    ('pls', PLSRegression())
    ])

    param_grid = {'pls__n_components': range(1,len(ranked_columns)+1)}
    grid = skm.GridSearchCV(pipe,\
                        param_grid,
                        cv=kf,
                        scoring='neg_mean_squared_error')
    grid.fit(X_Train, y_Train)

    print(f"Optimal PLS number of components: {grid.best_params_['pls__n_components']}")

    PLS_model = grid.best_estimator_.fit(X_Train,y_Train)

    PLS_optimal_params.loc['PLS'] = grid.best_params_['pls__n_components']

    PLS_test_predictions['PLS'] = PLS_model.predict(X_Test)
    PLS_train_predictions['PLS'] = PLS_model.predict(X_Train)
    
    PLS_test_error = PLS_test_predictions.sub(y_Test, axis=0)
    PLS_train_error = PLS_train_predictions.sub(y_Train, axis=0)
    
    PLS_test_mse = (PLS_test_error**2).mean(0)
    PLS_train_mse = (PLS_train_error**2).mean(0)
    
    return((PLS_model, PLS_optimal_params, PLS_test_mse, PLS_train_mse))


n_folds = 5

kf = skm.KFold(n_splits=n_folds, shuffle=True, random_state=100)

PLS_model, PLS_optimal_params, PLS_test_mse, PLS_train_mse = computes_test_mse(kf)

print('\nPLS Optimal parameters:\n')
print(PLS_optimal_params)

print("PLS Out-of-sample MSE:", PLS_test_mse)

print("PLS In-sample MSE:", PLS_train_mse)

# Compute the TSS using the mean of y_Train to prevent information leakage
PLS_TSS_test = ((y_Test - y_Train.mean())**2).mean()
PLS_TSS_train = ((y_Train - y_Train.mean())**2).mean()

PLS_R2_test = 1 - PLS_test_mse/PLS_TSS_test
PLS_R2_train = 1 - PLS_train_mse/PLS_TSS_train

print("PLS Out-of-sample R2:", PLS_R2_test)
print("PLS In-sample R2:", PLS_R2_train)




# c.5. Train RF Model
RF_TSS_test = ((y_Test - y_Train.mean())**2).mean()
RF_TSS_train = ((y_Train - y_Train.mean())**2).mean()

random_forest = RF(max_features=3,
                   n_estimators=100,
                   max_depth=5,
                   random_state=100).fit(X_Train , y_Train)

y_test_predicted = random_forest.predict(X_Test)
y_train_predicted = random_forest.predict(X_Train)

print("RF Out-of-sample MSE:",np.mean((y_Test - y_test_predicted)**2))
print("RF Out-of-sample R2:",1-np.mean((y_Test - y_test_predicted)**2)/RF_TSS_test)

print("RF In-sample MSE:",np.mean((y_Train - y_train_predicted)**2))
print("RF In-sample R2:",1-np.mean((y_Train - y_train_predicted)**2)/RF_TSS_train)


# Tuning the model
n_folds = 5

kf = skm.KFold(n_splits=n_folds, shuffle=True, random_state=100)
   
def tree_models(model, param_grid, val_type, X, y):
    grid = skm.GridSearchCV(model,
                            param_grid,
                            refit=True,
                            cv=val_type,
                            scoring='neg_mean_squared_error')
    grid.fit(X, y)
    RF_best_model = grid.best_estimator_
    RF_best_params = grid.best_params_
    return((RF_best_model, RF_best_params))

param_grid = {
        'n_estimators': [10], # Number of trees
        'random_state': [100], # Seed to initiate random sampling
        'max_features': [1, 3],  # Maximum # of features to sample
        'max_depth': [10],   # Maximum depth of the tree
        }

RF_best_model, RF_best_params = tree_models(RF(), param_grid, kf, X_Train, y_Train)

RF_MSE_test = np.mean((y_Test - RF_best_model.predict(X_Test))**2)
RF_R2_test = 1 - RF_MSE_test/RF_TSS_test

RF_MSE_train = np.mean((y_Train - RF_best_model.predict(X_Train))**2)
RF_R2_train = 1 - RF_MSE_train/RF_TSS_train
 
print("RF Out-of-sample MSE - best model:", RF_MSE_test)
print("RF Out-of-sample R2 - best model:", RF_R2_test)

print("RF In-sample MSE - best model:",RF_MSE_train)
print("RF In-sample R2 - best model:", RF_R2_train)

# Try another set of params, increase the n_estimators to 30 and 100 -> Result gets worse



# Question d: Compare the out-of-sample performance between 1990-1999 to the 2000-2021
# d.1. Split the test data into two periods: 1990-1999 and 2000-2021
test_data_1990s = test_data[(test_data['date'] >= '1990-01-01') & (test_data['date'] <= '1999-12-31')].reset_index(drop=True)
test_data_2000s = test_data[(test_data['date'] >= '2000-01-01')].reset_index(drop=True)

X_Test_1990s = test_data_1990s[ranked_columns]
y_Test_1990s = test_data_1990s['ret_ahead']

X_Test_2000s = test_data_2000s[ranked_columns]
y_Test_2000s = test_data_2000s['ret_ahead']

# d.2. Compute TSS for both periods to prevent information leakage
TSS_1990s = ((y_Test_1990s - y_Train.mean())**2).mean()
TSS_2000s = ((y_Test_2000s - y_Train.mean())**2).mean()

# d.3. Evaluate PLS model for 1990-1999
PLS_test_predictions_1990s = PLS_model.predict(X_Test_1990s)
MSE_1990s_PLS = mean_squared_error(y_Test_1990s, PLS_test_predictions_1990s)
R2_1990s_PLS = 1 - MSE_1990s_PLS / TSS_1990s

# d.4. Evaluate PLS model for 2000-2021
PLS_test_predictions_2000s = PLS_model.predict(X_Test_2000s)
MSE_2000s_PLS = mean_squared_error(y_Test_2000s, PLS_test_predictions_2000s)
R2_2000s_PLS = 1 - MSE_2000s_PLS / TSS_2000s

# d.5. Evaluate RF model for 1990-1999
RF_test_predictions_1990s = RF_best_model.predict(X_Test_1990s)
MSE_1990s_RF = mean_squared_error(y_Test_1990s, RF_test_predictions_1990s)
R2_1990s_RF = 1 - MSE_1990s_RF / TSS_1990s

# d.6. Evaluate RF model for 2000-2021
RF_test_predictions_2000s = RF_best_model.predict(X_Test_2000s)
MSE_2000s_RF = mean_squared_error(y_Test_2000s, RF_test_predictions_2000s)
R2_2000s_RF = 1 - MSE_2000s_RF / TSS_2000s

# d.7. Display the results
print("Out-of-sample performance comparison:")
print("\n1990-1999:")
print(f"PLS Model - MSE: {MSE_1990s_PLS:.4f}, R2: {R2_1990s_PLS:.4f}")
print(f"RF Model  - MSE: {MSE_1990s_RF:.4f}, R2: {R2_1990s_RF:.4f}")

print("\n2000-2021:")
print(f"PLS Model - MSE: {MSE_2000s_PLS:.4f}, R2: {R2_2000s_PLS:.4f}")
print(f"RF Model  - MSE: {MSE_2000s_RF:.4f}, R2: {R2_2000s_RF:.4f}")





# Question e: Measure varible importance (using in Panel Predictive R^2)
# e.1. Measure the variable importance for PLS model
# e.1.1. Create a dictionary to store importance index
pls_measure_variables_importances = {}

# e.1.2. Loop through all 30 features to calculate the difference between original R2 and modified R2
for i in range(len(ranked_columns)):
    #Pick the variable that needed to be measured
    measured_variable = ranked_columns[i]

    # Modify the variable's value that needed to be measured to 0
    test_data_modified = test_data.copy()
    test_data_modified[measured_variable] = 0
    
    X_Test_modified = test_data_modified[ranked_columns]
    y_Test_modified = test_data_modified['ret_ahead']
    
    #Calculate R2 with new modified data
    PLS_y_test_predictions_modified = PLS_model.predict(X_Test_modified)

    PLS_TSS_modified = ((y_Test_modified - y_Train.mean())**2).mean()
    PLS_MSE_modified = mean_squared_error(y_Test_modified, PLS_y_test_predictions_modified)
    PLS_R2_modified = 1 - PLS_MSE_modified / PLS_TSS_modified
    
    # Calculate the changes in R2 after excluding particular variable
    importance = PLS_R2_test.iloc[0] - PLS_R2_modified
    pls_measure_variables_importances[measured_variable] = importance

# e.1.3. Convert dictionary to DataFrame for final output
pls_measure_variables_importances = pd.DataFrame.from_dict(pls_measure_variables_importances, orient='index', columns=['Importance'])

# e.1.4. Print the result
print(pls_measure_variables_importances)

# e.1.5. Show result on chart
pls_measure_variables_importances.sort_values(by='Importance', ascending=True, inplace=True)
     
ax = pls_measure_variables_importances.plot.barh(legend=False, color='skyblue')
ax.set_xlabel('Values')
ax.set_ylabel('Features')
ax.set_title('PLS - R2-Based Variable Importance')

plt.show()




# e.2. Measure the variable importance for Random Forest model (similar to PLS model above)
# e.2.1. Create a dictionary to store importance index
rf_measure_variables_importances = {}

# e.2.2. Loop through all 30 features to calculate the difference between original R2 and modified R2
for i in range(len(ranked_columns)):
    #Pick the variable that needed to be measured
    measured_variable = ranked_columns[i]

    # Modify the variable's value that needed to be measured to 0
    test_data_modified = test_data.copy()
    test_data_modified[measured_variable] = 0
    
    X_Test_modified = test_data_modified[ranked_columns]
    y_Test_modified = test_data_modified['ret_ahead']

    # Calculate R2 with new modified data
    RF_y_test_predictions_modified = RF_best_model.predict(X_Test_modified)
    
    RF_TSS_modified = ((y_Test_modified - y_Train.mean())**2).mean()
    RF_MSE_modified = mean_squared_error(y_Test_modified, RF_y_test_predictions_modified)
    RF_R2_modified = 1 - RF_MSE_modified / RF_TSS_modified
    
    
    # Calculate the changes in R2 after excluding particular variable
    importance = RF_R2_test - RF_R2_modified
    rf_measure_variables_importances[measured_variable] = importance

# e.2.3. Convert dictionary to DataFrame for final output
rf_measure_variables_importances = pd.DataFrame.from_dict(rf_measure_variables_importances, orient='index', columns=['Importance'])

# e.2.4. Print the result
print(rf_measure_variables_importances)

# e.2.5. Show result on chart
rf_measure_variables_importances.sort_values(by='Importance', ascending=True, inplace=True)
     
ax = rf_measure_variables_importances.plot.barh(legend=False, color='skyblue')
ax.set_xlabel('Values')
ax.set_ylabel('Features')
ax.set_title('RF - R2-Based Variable Importance')

plt.show()



