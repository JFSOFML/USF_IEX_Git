# %%
# imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

#models
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve


# %%
#Load Data
columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']

df = pd.read_csv('AmesHousing.txt',
                 sep='\t',
                 usecols=columns)

# %%
# Clean data
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})

df.isnull().sum()

df = df.dropna(axis=0)

# %%
#Train Test Split
y = df["SalePrice"]
X = df.drop(["Central Air", "SalePrice"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
#Initialize Models
sc = StandardScaler()
forest_model = RandomForestRegressor(n_estimators=50, min_samples_split=4)
lr_model = LinearRegression()
SVR_model = SVR(C=10, epsilon= 0.2, kernel="sigmoid")

pipeline_forest = Pipeline([
    ('scaler', sc),
    ('forest', forest_model)
])

pipeline_lr = Pipeline([
    ('scaler', sc),
    ('lr', lr_model)
])

pipeline_SVR = Pipeline([
    ('scaler', sc),
    ('SVR', SVR_model)
])


# %%
# Fit models and print accuracy
pipeline_forest.fit(X_train, y_train)
pipeline_lr.fit(X_train, y_train)
pipeline_SVR.fit(X_train, y_train)


ypred_forest = pipeline_forest.predict(X_test)
r2_forest = r2_score(y_test, ypred_forest)
ypred_lr = pipeline_lr.predict(X_test)
r2_lr = r2_score(y_test, ypred_lr)
ypred_svr = pipeline_SVR.predict(X_test)
r2_svr = r2_score(y_test, ypred_svr)

print(f"r2_forest: {r2_forest}")
print(f"r2_lr: {r2_lr}")
print(f"r2_svr: {r2_svr}")





# %%
# Parameter grids

# %%
#Retrain models with Hyperparameter tuning

# %%
#Bagging
regr = BaggingRegressor(estimator=forest_model,
                        n_estimators=10, random_state=0).fit(X, y)
ypred = regr.predict(X_test)
r2 = r2_score(y_test, ypred)
r2

# %%
# Boosting
gboost = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,
   max_depth=1, random_state=0).fit(X_train, y_train)
gboost.score(X_test, y_test)

# %%
#Stacking
level1_models = [
    ('lr', lr_model),
    ('svr', SVR_model)
]
# Define the final estimator
final_estimator = forest_model

stacking_model = StackingRegressor(estimators=level1_models, final_estimator=final_estimator, cv=5)
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f'Stacking Model Accuracy: {r2:.4f}')

# %%
#voting
voting_model = VotingRegressor(estimators=level1_models)
voting_model.fit(X_train, y_train)
y_pred = voting_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f'Majority Voting Model R2: {r2:.4f}')


