import pandas as pd # Data manipulation and analysis
import seaborn as sns # Data visualization
import matplotlib.pyplot as plt # Plotting library

# Load dataset
df = pd.read_csv("titanic_data.csv")

# Fill missing age values with mean
df['Age'] = df['Age'].fillna(round(df['Age'].mean()))

# Display the number of male to female ratio
male_count = df[df['Sex'] == 'male'].shape[0]
female_count = df[df['Sex'] == 'female'].shape[0]
print(f'Male:Female ratio is {male_count}:{female_count}')

# Show the lowest and the highest fare paid
lowest_fare = df['Fare'].min()
highest_fare = df['Fare'].max()
print(f'Lowest fare paid: {lowest_fare}')
print(f'Highest fare paid: {highest_fare}')

# Create a visualization for first, second, and third class passengers
class_counts = df['Pclass'].value_counts()
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Number of Passengers in Each Class')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers')
plt.show()

# Show an age distribution of the passengers
sns.histplot(df['Age'], kde=True, bins=30, color='skyblue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

# Calculate and display survival rate by gender
survival_rate_gender = df.groupby('Sex')['Survived'].mean() * 100
print(f'Survival Rate by Gender:\n{survival_rate_gender}')

# Create a bar plot for survival rate by gender
sns.barplot(x=survival_rate_gender.index, y=survival_rate_gender.values, palette='coolwarm')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate (%)')
plt.show()

# Calculate and display survival rate by class
survival_rate_class = df.groupby('Pclass')['Survived'].mean() * 100
print(f'Survival Rate by Class:\n{survival_rate_class}')

# Create a bar plot for survival rate by class
sns.barplot(x=survival_rate_class.index, y=survival_rate_class.values, palette='viridis')
plt.title('Survival Rate by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate (%)')
plt.show()

# Categorize passengers into different age groups
age_bins = [0, 12, 18, 60, 80]
age_labels = ['Children', 'Teenagers', 'Adults', 'Seniors']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Calculate and display survival rate by age group
survival_rate_age_group = df.groupby('AgeGroup')['Survived'].mean() * 100
print(f'Survival Rate by Age Group:\n{survival_rate_age_group}')

# Create a bar plot for survival rate by age group
sns.barplot(x=survival_rate_age_group.index, y=survival_rate_age_group.values, palette='pastel')
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate (%)')
plt.show()

# Create a correlation heatmap
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
