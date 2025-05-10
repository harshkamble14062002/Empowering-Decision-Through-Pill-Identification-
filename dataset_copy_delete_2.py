import pandas as pd

# Load your dataset (Specify the path to your dataset here)
df = pd.read_csv('Medicine_Details.csv')  # Adjust the path as necessary

# Number of rows before removing duplicates
initial_count = len(df)

# Drop duplicates row based on 'Medicine Name' and 'Manufacturer', keeping only the first occurrence
df_unique = df.drop_duplicates(subset=['Medicine Name', 'Manufacturer'], keep='first')

# Number of rows after removing duplicates
final_count = len(df_unique)

# Calculate the number of rows deleted
deleted_count = initial_count - final_count

# Save the updated dataset back to the file
df_unique.to_csv('Medicine_Details_Updated.csv', index=False)  # Adjust the path as necessary

# Display the number of rows deleted
print(f"Number of duplicated rows removed: {deleted_count}")
