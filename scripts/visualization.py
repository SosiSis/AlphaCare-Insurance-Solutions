import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
          
def univariate_analysis_histogram_plot(data):
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:  # Numerical columns
            print(f"{column}")
            plt.figure(figsize=(8, 6))
            data[column].plot(kind='hist', bins=10, edgecolor='black')
            plt.title(f'Histogram for {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

def univariate_analysis_bar_chart(data):
    for column in data.columns:
        if data[column].dtype.name == 'object':  # Categorical columns
            print(f"{column}")

            plt.figure(figsize=(8, 6))
            data[column].value_counts().plot(kind='bar')
            plt.title(f'Bar Chart for {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

def plot_scatter_by_group(data, x, y, group):

    plt.figure(figsize=(10, 6))
    for grp in data[group].unique():
        subset = data[data[group] == grp]
        plt.scatter(subset[x], subset[y], label=f"{group}: {grp}", alpha=0.7)

    plt.title(f"Scatter Plot of {y} vs {x} by {group}")
    plt.xlabel(x)
    plt.ylabel(y)
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, title=group, frameon=False)
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(data, columns=None):

    if columns is None:
        columns = data.select_dtypes(include=['int64', 'float64']).columns

    corr_matrix = data[columns].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


def plot_grouped_bar_chart(data, geography):
    """
    Plot a grouped bar chart comparing TotalPremium and TotalClaims across regions.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        geography (str): The column name for geographical grouping.

    Returns:
        None: Displays the bar chart.
    """
    grouped_data = data.groupby(geography)[['TotalPremium', 'TotalClaims']].mean()

    grouped_data.plot(kind='bar', figsize=(12, 6))
    plt.title(f"Comparison of TotalPremium and TotalClaims Across {geography}")
    plt.xlabel(geography)
    plt.ylabel("Average Value")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()
    
    
def plot_correlation_heatmap(data, numeric_columns):
    """
    Plot a heatmap of correlation across numerical features.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        numeric_columns (list): List of numerical columns to include in the correlation matrix.

    Returns:
        None: Displays the heatmap.
    """
    correlation_matrix = data[numeric_columns].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.show()
    
def plot_stacked_area(data, time_column, category_column):
    """
    Plot a stacked area chart showing changes in category distribution over time.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        time_column (str): The column representing time (e.g., 'Year').
        category_column (str): The column representing the categorical variable (e.g., 'CoverType').

    Returns:
        None: Displays the stacked area chart.
    """
    # Count the occurrences of each category over time
    grouped_data = data.groupby([time_column, category_column]).size().unstack().fillna(0)

    grouped_data.plot(kind='area', stacked=True, figsize=(12, 6), alpha=0.7)
    plt.title(f"Changes in {category_column} Over {time_column}")
    plt.xlabel(time_column)
    plt.ylabel("Count")
    plt.legend(title=category_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()