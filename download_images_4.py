import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load your dataset (Specify the path to your dataset here)
df = pd.read_csv(r'C:\\Medicine_Details_Updated3.csv')  # Adjust the path as necessary

# Directory to save the images (Specify the folder path where images will be saved)
image_dir = Path(r'C:\\Medicine Images')  # Adjust the path as necessary
image_dir.mkdir(parents=True, exist_ok=True)

# Function to download an image
def download_image(row):
    image_url = row['Image URL']  # Adjust if your column name is different
    medicine_name = row['Medicine Name'].replace(' ', '_').replace('/', '_')
    manufacturer = row['Manufacturer'].replace(' ', '_').replace('/', '_')
    image_name = f"{medicine_name}_{manufacturer}.jpg"
    image_path = image_dir / image_name

    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
            return f"Downloaded {image_name}"
        else:
            return f"Failed to download {image_name} (status code: {response.status_code})"
    except Exception as e:
        return f"Error downloading {image_name}: {e}"

# Download images using threading
with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers based on your system's capability
    futures = [executor.submit(download_image, row) for _, row in df.iterrows()]
    for future in as_completed(futures):
        print(future.result())

print("All images processed.")
