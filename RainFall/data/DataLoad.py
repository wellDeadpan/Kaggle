import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load CSV files
rain = pd.read_csv('./data/Rainfall.csv')
print(rain.head())
# Clean column names (remove extra spaces)
rain.columns = rain.columns.str.strip().str.lower()
# Assume the data is already ordered from Jan 1 to Dec 31 (including Feb 29)
# Define number of days in each month for a leap year
days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Create a list with the month for each day
month_labels = [month for month, days in zip(months, days_in_month) for _ in range(days)]
rain['Month'] = month_labels
# Plot daily values colored by month for each feature
features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
# Convert to rain flag
rain['rain_flag'] = rain['rainfall'].map({'yes': 1, 'no': 0})


# EDA
rain['rainfall'].value_counts()
# Check for missing or non-finite values
print(rain[features].isna().sum())  # Check NaNs
print(rain[features].applymap(lambda x: not pd.api.types.is_number(x)).sum())  # Non-numeric check
print(~np.isfinite(rain[features]).all().all())  # Check for inf/-inf

cleaned_rain = rain.replace([np.inf, -np.inf], np.nan).dropna()
cleaned_rain['rainfall'].value_counts()

# Create summary statistics grouped by rainfall
summary_table = cleaned_rain.groupby('rainfall')[features].describe().transpose()
# Show the summary
summary_table

# Set seaborn style
sns.set_theme(style="whitegrid")
for feature in features:
    # Example: Plot daily temperature in each month
    g = sns.relplot(
        data=rain,
        x='day', y=feature,
        col='Month', kind='line',
        col_wrap=4, height=3, aspect=1.2,
        facet_kws={'sharey': True}
    )

    g.fig.suptitle(f'Daily {feature.capitalize()} by Month', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.show()

sorted_rain = cleaned_rain.sort_values(by='rain_flag', ascending=False).reset_index(drop=True)
# Standardize the values for better visual comparison
scaler = StandardScaler()
# Standardize only the feature values (after sorting)
scaled_data = scaler.fit_transform(sorted_rain[features])
scaled_df = pd.DataFrame(scaled_data, columns=features)
row_colors = sorted_rain['rainfall'].map({'yes': 'green', 'no': 'yellow'})

# Count how many Rain days we have
rainy_days = sorted_rain['rain_flag'].sum()

plt.figure(figsize=(14, 10))
sns.heatmap(
    scaled_df.T,
    cmap='coolwarm',
    cbar_kws={'label': 'Standardized Value'},
    xticklabels=False
)
plt.title('Daily Weather Heatmap (Grouped by Rainfall)', fontsize=16)
plt.ylabel('Features')
plt.xlabel('Day (Rain → No Rain)')
plt.tight_layout()
plt.show()


# Create DataFrame for heatmap
scaled_df = pd.DataFrame(scaled_data, columns=features)
scaled_df['rainfall'] = cleaned_rain['rainfall'].values  # keep raw rainfall for clustering

# Optional: cluster only by rainfall values
clustered_df = scaled_df.sort_values(by='rainfall', ascending=False).reset_index(drop=True)
heatmap_data = clustered_df[features]
labels = clustered_df.loc[clustered_df.index, 'rainfall']  # Align rain labels to cleaned rows
# Map 'Rain' and 'No Rain' to colors
row_colors = labels.map({'yes': 'green', 'no': 'yellow'})

plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data.T, cmap='coolwarm', cbar_kws={'label': 'Standardized Value'})
plt.title('Daily Weather Feature Heatmap (Sorted by Rainfall)', fontsize=16)
plt.ylabel('Features')
plt.xlabel('Day (sorted by rainfall)')
plt.tight_layout()
plt.show()

sns.clustermap(heatmap_data, row_colors=row_colors, cmap='coolwarm', figsize=(12, 10), z_score=0)
plt.suptitle("Clustered Heatmap of Daily Weather Features", y=1.02)
plt.show()
