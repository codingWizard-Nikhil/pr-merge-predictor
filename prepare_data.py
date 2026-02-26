import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('github_prs_processed.csv')


print("Loaded data shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop columns we don't need for training
print("\n=== Dropping Unnecessary Columns ===")
columns_to_drop = [
    'repo',              
    'pr_number',         
    'state',             # Redundant with 'merged'
    'created_at',        
    'closed_at',        
    'title',             # Already extracted description_length
    'body',              # Already extracted description_length
    'user_login',        # Not using author features
    'hour_created',      # Weak correlation 
    'day_of_week',       # Weak correlation 
    'add_del_ratio'      # Weak correlation 
]

df_clean = df.drop(columns=columns_to_drop)
print(f"Dropped {len(columns_to_drop)} columns")
print(f"Remaining columns: {df_clean.columns.tolist()}")

# Handle missing values
print("\n=== Handling Missing Values ===")
print(f"How many PRs have not been approved: {df_clean['time_open_hours'].isnull().sum()}")
df_clean['description_length'] = df_clean['description_length'].fillna(0)


# Drop rows with missing time_open_hours (open PRs)
df_clean = df_clean.dropna(subset=['time_open_hours'])

print(f"Rows after dropping open PRs: {len(df_clean)}")
print(f"Missing values remaining:\n{df_clean.isnull().sum()}")

# Verify correlations on clean data
print("\n=== Correlations on Closed PRs Only ===")
features = ['additions', 'deletions', 'changed_files', 'commits', 'comments',
            'total_lines', 'lines_per_commit', 'description_length', 'time_open_hours']
correlations = df_clean[features].corrwith(df_clean['merged'])
print(correlations.sort_values())



# Create train/test split
print("\n=== Creating Train/Test Split ===")

X = df_clean.drop('merged', axis=1)
y = df_clean['merged']

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Features: {X.columns.tolist()}")

# Save processed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\n✓ Saved train/test sets to CSV files")