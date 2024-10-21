# Bengaluru Air Quality and Electric Vehicles Impact Analysis

## Overview
This project analyzes air quality data in Bengaluru, focusing on pollutants and the potential impact of Electric Vehicle (EV) implementation on air quality. The dataset includes values for various air pollutants and AQI (Air Quality Index) from the city over time. The project also explores trends, correlations, and the effect of EV adoption on pollutant levels.

## Files
- `Bengaluru AQI.csv`: The dataset containing air quality measurements, including pollutant levels (PM2.5, PM10, NOx, CO2, SO2, O3) and AQI values.
- Python scripts to preprocess the data, analyze trends, model pollutant impacts, and predict future air quality based on increased EV usage.

## Prerequisites
- Python 3.x
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Preprocessing
1. **Missing Values**: Missing values in numeric columns are handled by filling them with the column mean.
2. **Date Parsing**: The 'Date' column is converted to a datetime format.
3. **Outlier Detection**: Outliers are detected using the Interquartile Range (IQR) method.

## Analysis
- **Descriptive Statistics**: Mean, median, and standard deviations are calculated for pollutants and AQI.
- **Visualization**: 
  - Histograms and distribution plots for pollutant levels.
  - Correlation heatmap between pollutants and AQI to understand relationships.
  - Time-series trends of pollutant levels and AQI.
  
- **Impact of EV Implementation**: 
  - The average levels of pollutants before and after EV adoption (from January 2020) are compared to estimate the impact of EVs on air quality.

## Model Training
A **Linear Regression** model is trained using historical pollutant data to predict AQI values. The model is evaluated using:
- **Mean Absolute Error (MAE)**
- **R-Squared (R²)**

Predictions are made for future AQI values based on hypothetical pollutant levels and scenarios with reduced emissions due to increased EV usage.

## Results
- **Trends**: 
  - The analysis shows how pollutant levels and AQI have evolved over time.
  - A significant reduction in specific pollutants is observed after EV adoption.
  
- **Prediction**: 
  - Future AQI predictions are made based on current trends.
  - A scenario with 20% reduction in pollutants due to EVs shows improvement in AQI levels.

## Interpretation of Findings
- **Correlation**: Strong correlations between certain pollutants (e.g., PM2.5, PM10) and AQI suggest that controlling these emissions can significantly improve air quality.
- **Impact of EVs**: A comparison of pollutant levels before and after January 2020 indicates that increased adoption of EVs correlates with reduced levels of CO2 and NOx, highlighting the positive impact of EVs on air quality.
- **Model Prediction**: The regression model effectively predicts AQI based on pollutant levels, with an R² value suggesting a strong fit. Scenario analysis further confirms the potential of EVs to lower AQI through reduced emissions.

## Future Work
Further studies could:
- Include more detailed data on traffic patterns and other sources of pollution.
- Explore machine learning models beyond linear regression for improved AQI prediction accuracy.

## Conclusion
This analysis provides insights into how EV adoption in Bengaluru impacts air quality. The results show that reducing vehicle emissions, especially through EVs, is a critical factor in improving air quality.

