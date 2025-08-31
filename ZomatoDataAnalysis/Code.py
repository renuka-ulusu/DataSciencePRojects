# =========================
# Module 1: Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Module 2: Data Loading
# =========================
# Update the path as per your local environment
DATA_PATH = "Zomato-data-.csv"  # Use relative path for portability

def load_data(path):
    """
    Loads the dataset from the given CSV file path.
    """
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

dataframe = load_data(DATA_PATH)
print(dataframe.head())

# =========================
# Module 3: Data Cleaning & Preprocessing
# =========================

def handle_rate(value):
    """
    Cleans and converts the 'rate' column to float.
    Handles missing or malformed values gracefully.
    """
    try:
        value = str(value).split('/')[0].strip()
        if value in ['NEW', '-', 'nan']:
            return np.nan
        return float(value)
    except:
        return np.nan

def clean_data(df):
    """
    Applies cleaning functions to the dataframe.
    """
    df['rate'] = df['rate'].apply(handle_rate)
    # Clean 'approx_cost(for two people)' column: remove commas and convert to int
    df['approx_cost(for two people)'] = (
        df['approx_cost(for two people)']
        .astype(str)
        .str.replace(',', '', regex=False)
        .replace('nan', np.nan)
        .astype(float)
    )
    # Fill missing values if needed
    df['rate'].fillna(df['rate'].mean(), inplace=True)
    df['approx_cost(for two people)'].fillna(df['approx_cost(for two people)'].median(), inplace=True)
    return df

dataframe = clean_data(dataframe)
print(dataframe.head())
dataframe.info()
print("Missing values per column:\n", dataframe.isnull().sum())

# =========================
# Module 4: Exploratory Data Analysis (EDA) & Visualization
# =========================

def plot_restaurant_types(df):
    """
    Plots the count of each restaurant type.
    """
    plt.figure(figsize=(10,6))
    sns.countplot(y=df['listed_in(type)'], order=df['listed_in(type)'].value_counts().index)
    plt.xlabel("Count")
    plt.ylabel("Type of Restaurant")
    plt.title("Distribution of Restaurant Types")
    plt.tight_layout()
    plt.show()

def plot_votes_by_type(df):
    """
    Plots total votes by restaurant type.
    """
    grouped_data = df.groupby('listed_in(type)')['votes'].sum().sort_values()
    plt.figure(figsize=(10,6))
    grouped_data.plot(kind='barh', color='green')
    plt.xlabel('Total Votes')
    plt.ylabel('Type of Restaurant')
    plt.title('Total Votes by Restaurant Type')
    plt.tight_layout()
    plt.show()

def plot_rating_distribution(df):
    """
    Plots the distribution of ratings.
    """
    plt.figure(figsize=(8,5))
    plt.hist(df['rate'], bins=10, color='skyblue', edgecolor='black')
    plt.title('Ratings Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_online_order_count(df):
    """
    Plots the count of restaurants offering online order.
    """
    plt.figure(figsize=(6,4))
    sns.countplot(x=df['online_order'])
    plt.title('Online Order Availability')
    plt.xlabel('Online Order')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_cost_distribution(df):
    """
    Plots the distribution of approximate cost for two people.
    """
    plt.figure(figsize=(8,5))
    sns.histplot(df['approx_cost(for two people)'], bins=20, kde=True)
    plt.title('Approximate Cost for Two People Distribution')
    plt.xlabel('Approximate Cost (INR)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_boxplot_online_order_vs_rate(df):
    """
    Plots a boxplot of ratings based on online order availability.
    """
    plt.figure(figsize=(6,6))
    sns.boxplot(x='online_order', y='rate', data=df)
    plt.title('Online Order vs Rating')
    plt.xlabel('Online Order')
    plt.ylabel('Rating')
    plt.tight_layout()
    plt.show()

def plot_heatmap_online_order_vs_type(df):
    """
    Plots a heatmap of restaurant types vs online order availability.
    """
    pivot_table = df.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='d')
    plt.title('Heatmap: Restaurant Type vs Online Order')
    plt.xlabel('Online Order')
    plt.ylabel('Restaurant Type')
    plt.tight_layout()
    plt.show()

# =========================
# Module 5: Insights & Analysis
# =========================

def restaurant_with_max_votes(df):
    """
    Finds and prints the restaurant(s) with the maximum votes.
    """
    max_votes = df['votes'].max()
    restaurants = df.loc[df['votes'] == max_votes, 'name']
    print('Restaurant(s) with the maximum votes:')
    print(restaurants.values)

# =========================
# Module 6: Main Execution
# =========================

if __name__ == "__main__":
    plot_restaurant_types(dataframe)
    plot_votes_by_type(dataframe)
    restaurant_with_max_votes(dataframe)
    plot_online_order_count(dataframe)
    plot_rating_distribution(dataframe)
    plot_cost_distribution(dataframe)
    plot_boxplot_online_order_vs_rate(dataframe)
    plot_heatmap_online_order_vs_type(dataframe)

# =========================
# End of Script
# =========================