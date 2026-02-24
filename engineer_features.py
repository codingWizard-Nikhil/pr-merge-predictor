import pandas as pd

#load data
df = pd.read_csv('github_prs_raw.csv')

print("Original shape:", df.shape)
print("\nFirst few rows")
print(df.head())


print("\n=== Creating Time Features ===")

# Convert created_at from string to datetime

df['created_at'] = pd.to_datetime(df['created_at'])
df['hour_created'] = df['created_at'].dt.hour
df['day_of_week'] = df['created_at'].dt.dayofweek

print("Sample of new time features:")
print(df[['created_at', 'hour_created', 'day_of_week']].head())


#--- DERIVED FEATURES ---

print("\n=== Creating Derived Features ===")

df['total_lines'] = df['additions'] + df['deletions']
df['lines_per_commit'] = df['total_lines'] / df['commits'].replace(0, 1)
df['description_length'] = df['title'].str.len() + df['body'].str.len()


print("Sample of derived features:")
print(df[['additions', 'deletions', 'total_lines', 'lines_per_commit', 'description_length']].head())


#--- additional features ---

#how long PR was open
print("=== Creating Time-Open Feature ===")
df['closed_at'] = pd.to_datetime(df['closed_at'])
df['time_open_hours'] = (df['closed_at'] - df['created_at']).dt.total_seconds()/3600 


# Handle PRs still open (closed_at is NaN)
print(f"Open PRs: {df['closed_at'].isnull().sum()}")
print("Sample time_open_hours:")
print(df[['created_at', 'closed_at', 'time_open_hours', 'merged']].head(10))




# Additions/deletions ratio
print("\n=== Creating Add/Delete Ratio Feature ===")
df['add_del_ratio'] = df['additions'] / df['deletions'].replace(0, 1)
df.loc[(df['additions'] == 0) & (df['deletions'] == 0), 'add_del_ratio'] = 1

print("Sample add_del_ratio:")
print(df[['additions', 'deletions', 'add_del_ratio', 'merged']].head(10))


# Check correlations including new features
print("\n=== UPDATED CORRELATIONS (With New Features) ===")
all_features = ['additions', 'deletions', 'changed_files', 'commits', 'comments',
                'hour_created', 'day_of_week', 'total_lines', 'lines_per_commit', 
                'description_length', 'time_open_hours', 'add_del_ratio']
correlations = df[all_features].corrwith(df['merged'])
print(correlations.sort_values())


# Save processed data
df.to_csv('github_prs_processed.csv', index=False)
print("\n✓ Saved processed data to github_prs_processed.csv")