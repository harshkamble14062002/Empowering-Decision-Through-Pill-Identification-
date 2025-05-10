import pandas as pd

# Load the dataset (adjust the path as necessary)
df = pd.read_csv('Medicine_Details.csv')

# Convert the 'Medicine Name' column to lowercase to make the search case-insensitive
df['Medicine Name'] = df['Medicine Name'].str.lower()

# Keep only the rows that contain the words 'tablet', 'pill', or 'capsule' in the 'Medicine Name' column
filtered_df = df[df['Medicine Name'].str.contains('tablet|pill|capsule')]

# Save the updated dataset (adjust the path as necessary)
filtered_df.to_csv('Medicine_Details_Updated3.csv', index=False)

# Print the number of rows removed
rows_removed = len(df) - len(filtered_df)
print(f"Number of rows removed: {rows_removed}")