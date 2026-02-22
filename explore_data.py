import pandas as pd
import matplotlib.pyplot as plt


#load data
df = pd.read_csv('github_prs_raw.csv')


#first portion of data
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())


# Check correlations with merged
print("\n=== CORRELATIONS WITH MERGED ===")
numeric_cols = ['additions', 'deletions', 'changed_files', 'commits', 'comments']
correlations = df[numeric_cols].corrwith(df['merged'])
print(correlations.sort_values())