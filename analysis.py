import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the data
df = pd.read_csv('data/product-sales.csv')

# Convert 'Month' to datetime
df['Month'] = pd.to_datetime(df['Month'], format='%d-%b')

# Reshape the 'ProductSales' column to 2D array as required by the model
sales = df['ProductSales'].values.reshape(-1,1)

# Define the model
model = IsolationForest(contamination=0.05)

# Fit the model
df['Anomaly'] = model.fit_predict(sales)

# Print the DataFrame with anomalies
print(df)