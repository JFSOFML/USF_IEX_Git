"""Convert ipynb for ensemble learning"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

column_names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

# Load the 'adult.data' file with named columns
data_df = pd.read_csv("adult.data", names=column_names)

# Load the 'adult.test' file with named columns
test_df = pd.read_csv("adult.test", names=column_names)
# Combine the two DataFrames
df = pd.concat([data_df, test_df], ignore_index=True)

# Save the combined DataFrame to a new CSV file (optional)
df.to_csv("adult_combined.csv", index=False)

print("Combined dataset with named columns created successfully!")

df.head()

df.describe()

columns_to_encode = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]

label_encoders = {}

for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

X = df.drop(
    columns=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
)
y = df["income"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cls = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42
).fit(X_train, y_train)
cls.score(X_test, y_test)

gnb_classifier = GaussianNB()
svc_classifier = SVC()

clf = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0).fit(
    X_train, y_train
)
clf.predict(X_test)

clf = BaggingClassifier(estimator=GaussianNB(), n_estimators=10, random_state=0).fit(
    X_test, y_test
)
clf.predict(X_test)