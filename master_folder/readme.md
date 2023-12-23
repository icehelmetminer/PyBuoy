# PySeas

```md
PySeas/
├── config.py                     # Configuration settings (e.g., global variables)
├── scraper.py                    # Contains the `BuoyCamScraper` class for scraping images
├── image_processor.py            # Contains the `ImageProcessor` class for processing images
├── logger.py                     # Setup for logging
├── scheduled_tasks.py            # Functions for scheduled tasks (e.g., `scheduled_task`)
├── run_schedule.py               # Contains the `run_schedule` function to execute scheduled tasks
├── main.py                       # Main script integrating all components
├── utils/
│   ├── duplicate_removal.py      # Functions for removing duplicates
│   ├── white_image_removal.py    # Functions for removing white images
│   └── other utilities           # Additional utility functions
├── logs/                         # Directory for log files
├── images/                       # Directory to store scraped images
│   ├── buoys/                    # Subdirectory for buoy images
│   └── save_images/              # Subdirectory for saved images
└── panels/                       # Directory for processed panels
```
main.py includes the main execution logic, threading setup, and initialization of other modules.
scheduled_tasks.py includes the scheduled_task function that contains the scraping and processing logic.
run_schedule.py includes the run_schedule function that runs the schedule continuously in a separate thread.
scraper.py, image_processor.py, duplicate_removal.py, and white_image_removal.py contain their respective classes and functions as previously described.
config.py and logger.py handle configuration settings and logging setup.

Horizon Detection: https://github.com/sallamander/horizon-detection/blob/master/README.md
