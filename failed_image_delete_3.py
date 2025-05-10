import pandas as pd  # Ensure pandas is imported
import requests


# Load your dataset (Specify the path to your dataset here)
df = pd.read_csv('Medicine_Details_Updated.csv')  # Adjust the path as necessary

# Function to check image link accessibility
def check_image_link(row):
    image_url = row['Image URL']  # Adjust if your column name is different
    try:
        response = requests.head(image_url)  # Use HEAD request to check link
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking link for {row['Medicine Name']}: {e}")
        return False

# Apply the check to each row and filter out rows with inaccessible links
df['link_accessible'] = df.apply(check_image_link, axis=1)
# remove those all row which are store as false 
df_filtered = df[df['link_accessible'] == True].drop(columns='link_accessible')

# Save the updated dataset with accessible links
df_filtered.to_csv('Medicine_Details_Updated2.csv', index=False)  # Adjust the path as necessary

# Display the number of removed rows
removed_count = len(df) - len(df_filtered)
print(f"Number of rows with inaccessible image links removed: {removed_count}")