import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

def clean_data(df):
    # Remove columns/rows with all missing values
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    # Fill missing values with column mean (numeric) or mode (categorical)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '')
    return df

def summarize(df, columns):
    summary = []
    for col in columns:
        if col in df.columns:
            col_data = df[col]
            mean = col_data.mean() if np.issubdtype(col_data.dtype, np.number) else None
            median = col_data.median() if np.issubdtype(col_data.dtype, np.number) else None
            missing = col_data.isnull().sum()
            summary.append({
                "column": col,
                "mean": mean,
                "median": median,
                "missing": int(missing)
            })
    return summary

def correlation(df, columns):
    corr = df[columns].corr().to_dict() if columns else {}
    return corr

def plot_histogram(df, columns, out_path):
    plt.figure(figsize=(8, 6))
    for col in columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col].plot(kind='hist', alpha=0.5, label=col)
    plt.legend()
    plt.title('Histogram')
    plt.savefig(out_path)
    plt.close()

def plot_heatmap(df, columns, out_path):
    corr = df[columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(out_path)
    plt.close()

def main():
    # Load input
    with open('key_value_stores/default/INPUT.json', 'r') as f:
        input_data = json.load(f)
    file_path = input_data.get('dataFile')
    columns = input_data.get('columnsToAnalyze', [])
    summary_type = input_data.get('summaryType', 'stats')

    # Load data with error handling
    if not file_path or not Path(file_path).exists():
        print(f"Input file '{file_path}' not found. Please check your INPUT.json and file location.")
        return
    ext = Path(file_path).suffix.lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file type: {ext}")
            return
    except Exception as e:
        print(f"Error loading file '{file_path}': {e}")
        return

    # Clean data
    cleaned_df = clean_data(df)
    cleaned_path = 'key_value_stores/default/cleaned.csv'
    cleaned_df.to_csv(cleaned_path, index=False)
    cleaned_df.to_excel('key_value_stores/default/cleaned.xlsx', index=False)

    # Summary
    columns_to_use = columns if columns else cleaned_df.columns.tolist()
    summary = summarize(cleaned_df, columns_to_use)
    # Only use numeric columns for correlation and plots
    numeric_columns = [col for col in columns_to_use if pd.api.types.is_numeric_dtype(cleaned_df[col])]
    corr = correlation(cleaned_df, numeric_columns) if summary_type in ['correlations', 'charts'] and numeric_columns else {}
    def nan_to_none(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [nan_to_none(v) for v in obj]
        return obj

    summary_json = {
        "summary": nan_to_none(summary),
        "correlation": nan_to_none(corr)
    }
    with open('key_value_stores/default/summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    # Outlier Detection (IQR method for numeric columns)
    outliers = {}
    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (cleaned_df[col] < (Q1 - 1.5 * IQR)) | (cleaned_df[col] > (Q3 + 1.5 * IQR))
        outliers[col] = cleaned_df.loc[mask, col].tolist()
    with open('key_value_stores/default/outliers.json', 'w') as f:
        json.dump(outliers, f, indent=2)

    # Data Type Summary
    data_types = {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()}
    with open('key_value_stores/default/data_types.json', 'w') as f:
        json.dump(data_types, f, indent=2)

    # Custom Plots
    # Boxplot
    boxplot_path = 'key_value_stores/default/boxplot.png'
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=cleaned_df.select_dtypes(include=[np.number]))
    plt.title('Boxplot')
    plt.savefig(boxplot_path)
    plt.close()

    # Scatter Plot (first two numeric columns)
    scatter_path = 'key_value_stores/default/scatter.png'
    num_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=cleaned_df[num_cols[0]], y=cleaned_df[num_cols[1]])
        plt.xlabel(num_cols[0])
        plt.ylabel(num_cols[1])
        plt.title('Scatter Plot')
        plt.savefig(scatter_path)
        plt.close()

    # Bar Chart (first categorical column)
    barchart_path = 'key_value_stores/default/barchart.png'
    cat_cols = cleaned_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        plt.figure(figsize=(8, 6))
        cleaned_df[cat_cols[0]].value_counts().plot(kind='bar')
        plt.title(f'Bar Chart: {cat_cols[0]}')
        plt.xlabel(cat_cols[0])
        plt.ylabel('Count')
        plt.savefig(barchart_path)
        plt.close()

    # Plot (histogram/correlation heatmap)
    plot_path = 'key_value_stores/default/plot.png'
    if summary_type == 'charts' and numeric_columns:
        plot_histogram(cleaned_df, numeric_columns, plot_path)
    elif summary_type == 'correlations' and numeric_columns:
        plot_heatmap(cleaned_df, numeric_columns, plot_path)
    # If stats, skip plot

if __name__ == '__main__':
    main()
