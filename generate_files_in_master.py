import os

# Define the folder structure
folders = [
    'master_folder',
    'master_folder/config',
    'master_folder/utils',
    'master_folder/logs',
    'master_folder/images',
    'master_folder/images/buoys',
    'master_folder/images/save_images',
    'master_folder/panels'
]

files = [
    'master_folder/config.py',
    'master_folder/scraper.py',
    'master_folder/image_processor.py',
    'master_folder/logger.py',
    'master_folder/scheduled_tasks.py',
    'master_folder/run_schedule.py',
    'master_folder/main.py',
    'master_folder/utils/duplicate_removal.py',
    'master_folder/utils/white_image_removal.py'
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create blank files
for file in files:
    with open(file, 'w') as f:
        pass

print("Folders and files created successfully.")
