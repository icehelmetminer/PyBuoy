import os
import shutil
from datetime import datetime, timedelta

def clean_old_images(directory, days_old=7):
    cutoff_time = datetime.now() - timedelta(days=days_old)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_time < cutoff_time:
                os.remove(file_path)
                print(f"Removed old file: {file_path}")

def safe_execute(function):
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {function.__name__}: {e}")
            return None
    return wrapper

# Usage Example
@safe_execute
def scrape_buoycam(buoycam_id):
    # scraping logic
    pass

# Example Usage
clean_old_images('images/buoys', days_old=7)
