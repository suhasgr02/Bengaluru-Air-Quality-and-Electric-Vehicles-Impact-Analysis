import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

# Load the CSV file
file_path = 'Bengaluru AQI.csv'
data = pd.read_csv(r"C:\Users\suhas\Desktop\mini project\Master Thesis\Data\Main_Air_vehicle_emission_dataset.csv\Bengaluru AQI.csv")

# Display the first few rows to understand the data
data.head(), data.info()

# Step 1: Data Preprocessing

# 1.1 Handling Missing Values
# Check for missing values
missing_values = data.isnull().sum()

# Fill missing values with the column's mean for numeric columns only
numeric_cols = data.select_dtypes(include=['float64']).columns  # Select only numeric columns
data_filled = data.copy()
data_filled[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())


# 1.2 Date Parsing
# Convert the Date column to datetime format
data_filled['Date'] = pd.to_datetime(data_filled['Date'], format='%d-%m-%Y')

# 1.3 Outlier Detection
# Define a function to detect outliers using the IQR method
def detect_outliers(df, columns):
    outliers = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
    return outliers

# Detect outliers in pollutant columns and AQI
outlier_columns = ['PM2.5', 'PM10', 'Nox', 'CO2', 'SO2', 'O3', 'AQI']
outliers_detected = detect_outliers(data_filled, outlier_columns)

missing_values, outliers_detected

# Step 2: Descriptive Statistics

# 2.1 Calculate basic descriptive statistics (mean, median, standard deviation) for pollutants and AQI
descriptive_stats = data_filled[numeric_cols].describe()

# 2.2 Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing the distribution of AQI and pollutants using histograms
plt.figure(figsize=(14, 10))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data_filled[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()

# Visualizing correlations between variables using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data_filled[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Pollutants and AQI')
plt.show()

descriptive_stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset (replace 'your_file.csv' with the actual file name)
data_filled = pd.read_csv(r"C:\Users\suhas\Desktop\mini project\Master Thesis\Data\Main_Air_vehicle_emission_dataset.csv\Bengaluru AQI.csv")  # Adjust the path as needed

# Display the first few rows and the columns of the DataFrame to inspect its structure
print("Columns in the DataFrame:")
print(data_filled.columns)  # Check column names

print("\nFirst few rows of the DataFrame:")
print(data_filled.head())  # Check the data

# Check if 'Date' column exists after inspecting
if 'Date' not in data_filled.columns:
    raise KeyError("The 'Date' column is missing from the dataset.")

# If the 'Date' column exists, proceed with further processing
data_filled['Date'] = pd.to_datetime(data_filled['Date'], format='%d-%m-%Y', errors='coerce')

# Drop rows where Date is NaT (Not a Time)
data_filled = data_filled.dropna(subset=['Date'])

# Set the Date column as the index for resampling
data_filled.set_index('Date', inplace=True)

# Select only numeric columns for resampling
numeric_cols = ['PM2.5', 'PM10', 'Nox', 'CO2', 'SO2', 'O3', 'AQI']

# Ensure numeric columns exist
for col in numeric_cols:
    if col not in data_filled.columns:
        raise KeyError(f"The '{col}' column is missing from the dataset.")

# Proceed with resampling and analysis as before
monthly_data = data_filled[numeric_cols].resample('M').mean()

# Check for NaN values and clean the data if necessary
monthly_data = monthly_data.dropna()

# Plotting time-series trends for AQI and pollutants
plt.figure(figsize=(14, 8))

# Create a function to plot trends with linear regression
def plot_with_trend(data, label, color):
    plt.plot(data.index, data, label=label, color=color)
    z = np.polyfit(data.index.to_julian_date(), data, 1)
    p = np.poly1d(z)
    plt.plot(data.index, p(data.index.to_julian_date()), color=color, linestyle='--', label=f"{label} Trend")

# Plot each pollutant with trend lines
plot_with_trend(monthly_data['AQI'], 'AQI', 'blue')
plot_with_trend(monthly_data['PM2.5'], 'PM2.5', 'green')
plot_with_trend(monthly_data['PM10'], 'PM10', 'orange')
plot_with_trend(monthly_data['Nox'], 'Nox', 'red')

plt.xlabel('Date')
plt.ylabel('Levels')
plt.title('Trend of AQI and Pollutants Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Comparing periods before and after EV implementation
ev_implementation_date = pd.to_datetime('2020-01-01')

pre_ev_data = monthly_data[monthly_data.index < ev_implementation_date]
post_ev_data = monthly_data[monthly_data.index >= ev_implementation_date]

# Remove rows with all NaN values in numeric columns
pre_ev_data_cleaned = pre_ev_data.dropna(how='all', subset=numeric_cols)
post_ev_data_cleaned = post_ev_data.dropna(how='all', subset=numeric_cols)

# Calculate the means after removing rows with all NaN values
pre_ev_means = pre_ev_data_cleaned.mean()
post_ev_means = post_ev_data_cleaned.mean()

print("Average levels before EV implementation:\n", pre_ev_means)
print("\nAverage levels after EV implementation:\n", post_ev_means)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess the data (if not already done)
data_filled = pd.read_csv(r"C:\Users\suhas\Desktop\mini project\Master Thesis\Data\Main_Air_vehicle_emission_dataset.csv\Bengaluru AQI.csv")
data_filled['Date'] = pd.to_datetime(data_filled['Date'], format='%d-%m-%Y', errors='coerce')
data_filled = data_filled.dropna(subset=['Date'])
data_filled.set_index('Date', inplace=True)

# Select only numeric columns for analysis
numeric_cols = ['PM2.5', 'PM10', 'Nox', 'CO2', 'SO2', 'O3', 'AQI']
monthly_data = data_filled[numeric_cols].resample('M').mean()

# Ensure there are no NaN values before correlation analysis
monthly_data = monthly_data.dropna()

# 1. Correlation Analysis
correlation_matrix = monthly_data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Pollutants and AQI')
plt.show()

# 2. Analyze the impact of EV implementation
ev_implementation_date = pd.to_datetime('2020-01-01')
pre_ev_data = monthly_data[monthly_data.index < ev_implementation_date]
post_ev_data = monthly_data[monthly_data.index >= ev_implementation_date]

# Calculate means before and after EV implementation
pre_ev_means = pre_ev_data.mean()
post_ev_means = post_ev_data.mean()

# Create a DataFrame to compare pre and post EV implementation means
comparison_df = pd.DataFrame({'Pre EV Implementation': pre_ev_means,
                               'Post EV Implementation': post_ev_means})

print("\nComparison of Pollutant Levels Before and After EV Implementation:")
print(comparison_df)

# Calculate the percentage change for better analysis
comparison_df['Percentage Change'] = ((comparison_df['Post EV Implementation'] -
                                        comparison_df['Pre EV Implementation']) /
                                        comparison_df['Pre EV Implementation']) * 100

print("\nComparison with Percentage Change:")
print(comparison_df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load and preprocess the data
data_filled = pd.read_csv(r"C:\Users\suhas\Desktop\mini project\Master Thesis\Data\Main_Air_vehicle_emission_dataset.csv\Bengaluru AQI.csv")
data_filled['Date'] = pd.to_datetime(data_filled['Date'], format='%d-%m-%Y', errors='coerce')
data_filled = data_filled.dropna(subset=['Date'])
data_filled.set_index('Date', inplace=True)

# Select only numeric columns for analysis
numeric_cols = ['PM2.5', 'PM10', 'Nox', 'CO2', 'SO2', 'O3', 'AQI']
monthly_data = data_filled[numeric_cols].resample('M').mean()

# Ensure there are no NaN values
monthly_data = monthly_data.dropna()

# 1. Data Preparation
X = monthly_data[['PM2.5', 'PM10', 'Nox', 'CO2', 'SO2', 'O3']]  # Features
y = monthly_data['AQI']  # Target variable

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Selection
model = LinearRegression()

# 4. Model Training
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

# 6. Forecasting future AQI based on hypothetical pollutant levels
# Example: Predict AQI for future levels of pollutants
future_pollutants = np.array([[30, 40, 10, 350, 20, 50],  # Hypothetical pollutant levels
                               [20, 30, 5, 300, 10, 40],   # Different scenario
                               [15, 25, 2, 250, 5, 35]])  # Another scenario

future_predictions = model.predict(future_pollutants)
print("\nPredicted future AQI levels based on hypothetical pollutant levels:")
for i, prediction in enumerate(future_predictions):
    print(f"Scenario {i + 1}: AQI = {prediction:.2f}")

# 7. Estimate improvements due to increased EV usage
# Example scenario: Assume a 20% reduction in each pollutant due to EVs
reduced_pollutants = future_pollutants * 0.8  # 20% reduction
improved_predictions = model.predict(reduced_pollutants)
print("\nPredicted AQI levels after 20% reduction in pollutants due to EVs:")
for i, prediction in enumerate(improved_predictions):
    print(f"Scenario {i + 1}: AQI = {prediction:.2f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
data_filled = pd.read_csv(r"C:\Users\suhas\Desktop\mini project\Master Thesis\Data\Main_Air_vehicle_emission_dataset.csv\Bengaluru AQI.csv")
data_filled['Date'] = pd.to_datetime(data_filled['Date'], format='%d-%m-%Y', errors='coerce')
data_filled = data_filled.dropna(subset=['Date'])
data_filled.set_index('Date', inplace=True)

# Select only numeric columns for analysis
numeric_cols = ['PM2.5', 'PM10', 'Nox', 'CO2', 'SO2', 'O3', 'AQI']
monthly_data = data_filled[numeric_cols].resample('M').mean()

# Ensure there are no NaN values
monthly_data = monthly_data.dropna()

# 1. Line Chart for Trends Over Time
plt.figure(figsize=(14, 8))
for col in numeric_cols:
    plt.plot(monthly_data.index, monthly_data[col], label=col)
plt.title('Trends of AQI and Pollutants Over Time')
plt.xlabel('Date')
plt.ylabel('Levels')
plt.legend()
plt.grid()
plt.show()

# 2. Bar Chart for Average Pollutant Levels Before and After EV Implementation
ev_implementation_date = pd.to_datetime('2020-01-01')
pre_ev_data = monthly_data[monthly_data.index < ev_implementation_date].mean()
post_ev_data = monthly_data[monthly_data.index >= ev_implementation_date].mean()

# Create a DataFrame for bar chart
comparison_df = pd.DataFrame({
    'Pre EV Implementation': pre_ev_data,
    'Post EV Implementation': post_ev_data
})

comparison_df.plot(kind='bar', figsize=(10, 6))
plt.title('Average Pollutant Levels Before and After EV Implementation')
plt.ylabel('Levels')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# 3. Heatmap for Correlation Matrix
correlation_matrix = monthly_data.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Pollutants and AQI')
plt.show()