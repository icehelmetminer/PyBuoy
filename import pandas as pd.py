import pandas as pd
import os
from tqdm import tqdm

def remove_duplicates_from_csv(csv_file):
    """
    Remove duplicate images based on entries in a CSV file.
    Assumes the CSV has columns 'image1' and 'image2'.
    """
    # Read the CSV file containing duplicates
    df = pd.read_csv(csv_file)

    # Iterate through each pair of duplicates
    for _, row in tqdm(df.iterrows(), desc="Processing duplicates", total=len(df)):
        # Remove the second image in each duplicate pair
        if os.path.exists(row['image2']):
            os.remove(row['image2'])
            print(f"Removed duplicate image: {row['image2']}")

if __name__ == "__main__":
    # Path to the CSV file with duplicates
    csv_file = 'duplicate_images.csv'
    print('Removing duplicate images')
    remove_duplicates_from_csv(csv_file)
