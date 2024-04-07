import pickle
import streamlit as st
import app
# seaborn offers customizable visualizations
import seaborn as sns
# this allows you to plot charts 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

with open("SVCModel.pkl", "rb") as f: 
    linear_SVC_model = pickle.load(f)
    

st.header("Visualization")

app.train

app.test

app.gender_submission





selected_variables = app.train[['Age','Pclass', 'Survived', 'Fare','SibSp','Sex_binary']] #inner list, outer brackets are how I reference columns in train

# Drop rows with missing values to ensure proper correlation calculation
selected_variables = selected_variables.dropna()

# Calculate the correlation matrix
corr_matrix = selected_variables.corr()

# Create the heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')# annot=true means insert the values visually withing the squares. 
st.pyplot(plt.gcf())
# The below shows the correlation highest positive correlation

app.Titanic


# Copy the DataFrame to avoid modifying the original data
titanic_df = app.Titanic.copy()

# I Mapped numerical values to strings
titanic_df['Survived'] = titanic_df['Survived'].map({0: 'Died', 1: 'Survived'})


fig = px.histogram(titanic_df, x='Survived', 
                   color='Survived', 
                   category_orders={"Survived": ['Died', 'Survived']},
                   labels={"Survived": "Outcome"})  

fig.update_layout(
    title='Survival Histogram',
    xaxis_title='Outcome',
    yaxis_title='Count',
    barmode='group',
    bargap=0.2,  # Adjusted gap 
    xaxis_title_font_size=14,
    yaxis_title_font_size=14,
    title_font_size=16,
    legend_title_text='Outcome',
    legend=dict(x=0.8, y=0.9, font_size=12),
    margin=dict(l=50, r=50, t=50, b=50)
)

# Displays in my Streamlit app
st.plotly_chart(fig)

# Display a header in your Streamlit app
st.header('Feature Importance Analysis')

# Print the feature ranking
st.write("Feature ranking:")
for f in range(train_features_norm.shape[1]):  # using train_features_norm
    st.write("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(10, 6))  # Optionally adjust the figure size
plt.title('Feature Importance')
plt.bar(range(train_features_norm.shape[1]), importances[indices], color='skyblue', align='center')
plt.xticks(range(train_features_norm.shape[1]), [feat_labels[i] for i in indices], rotation=90)
plt.xlim([-1, train_features_norm.shape[1]])
plt.tight_layout()

# Display the plot in your Streamlit app
st.pyplot(plt)