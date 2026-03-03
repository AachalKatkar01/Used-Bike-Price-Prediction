import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset

df = pd.read_csv("bikes.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())



# 2. Data Cleaning


# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)


# Feature Engineering


current_year = 2026

# Bike Age
df['Bike_Age'] = current_year - df['model_year']

# Clean kms_driven column
df['kms_driven'] = df['kms_driven'].str.extract('(\d+)')
df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')

# Clean owner column
df['owner'] = df['owner'].str.extract('(\d+)')
df['owner'] = pd.to_numeric(df['owner'], errors='coerce')

# Fill missing owner as 1 (assume first owner if missing)
df['owner'] = df['owner'].fillna(1)

# Drop remaining NaN rows
df = df.dropna(subset=['kms_driven', 'Bike_Age', 'owner', 'price'])


# 4. Exploratory Data Analysis


plt.figure()
sns.histplot(df['price'], kde=True)
plt.title("Price Distribution")
plt.show()

plt.figure()
sns.scatterplot(x='kms_driven', y='price', data=df)
plt.title("KMs Driven vs Price")
plt.show()

plt.figure()
sns.boxplot(x='owner', y='price', data=df)
plt.title("Owner Type vs Price")
plt.show()



# 5. Model Building


# Select numerical features
X = df[['kms_driven', 'Bike_Age', 'owner']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# 6. Model Evaluation


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# 7. Prediction Example


sample = [[20000, 3, 1]]  # kms_driven, Bike_Age, owner
predicted_price = model.predict(sample)

print("\nPredicted Price for Sample Bike:", predicted_price[0])