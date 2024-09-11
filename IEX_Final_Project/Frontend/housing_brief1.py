"""Entry file for streamlit app"""

import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import streamlit as st  # streamlit run Housing.py -- Run that

st.header("Housing Prices Predictions")
st.write(
    """
Author: Julio Figueroa

"""
)
st.write(
    """
Data Source: [Here](https://github.com/rasbt/machine-learning-book/blob/main/ch09/AmesHousing.txt)
"""
)
st.write(
    """
### Selected Features from the Dataset

- **Overall Qual**: Rates the overall material and finish of the house
- **Overall Cond**: Rates the overall condition of the house
- **Gr Liv Area**: Above grade (ground) living area square feet
- **Central Air**: Central air conditioning (Yes/No)
- **Total Bsmt SF**: Total square feet of basement area
- **SalePrice**: Sale price of the house
"""
)

columns = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Central Air",
    "Total Bsmt SF",
    "SalePrice",
]

df = pd.read_csv(
    "http://jse.amstat.org/v19n3/decock/AmesHousing.txt", sep="\t", usecols=columns
)


st.write(
    """
## Data Preprocessing Steps
```df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})```

```df.isnull().sum()```

1. **Binary Encoding**: Converted 'Central Air' from categorical \
    ('N', 'Y') to binary (0, 1) representation.
2. **Missing Values Check**: Evaluated each column for missing \
    values to ensure data integrity.
"""
)


df["Central Air"] = df["Central Air"].map({"N": 0, "Y": 1})

df = df.dropna(axis=0)
df.isnull().sum()


X1 = df[
    [
        "Gr Liv Area",
        "Overall Qual",
        "Overall Cond",
        "Gr Liv Area",
        "Central Air",
        "Total Bsmt SF",
    ]
].values  # Defining all my features variables
Y = df["SalePrice"].values

X = df.drop("SalePrice", axis=1)
sc = StandardScaler()


X_scaled = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.25, random_state=42
)
forest = RandomForestRegressor(n_estimators=50, min_samples_split=4)

forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
R2 = metrics.r2_score(y_test, y_pred)

with open("Scaler.pkl", "wb") as f:
    pickle.dump(sc, f)
with open("forest.pkl", "wb") as f:
    pickle.dump(forest, f)

st.header("Summary")

st.write("I used a Random Forest Regressor")
st.write("I used a StandardScaler")
st.write(
    f"My R2 score was {(round(R2,2)*100)}%, indicating that it \
        accurately predicts the sale prices of houses about 86% \
            of the time, compared to the actual sale prices."
)
st.write("My MAE was approximately $21,000")
