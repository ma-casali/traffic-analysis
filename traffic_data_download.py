import os

# Optional: set Kaggle credentials manually (if not using ~/.kaggle/kaggle.json)
os.environ["KAGGLE_USERNAME"] = "your_username"
os.environ["KAGGLE_KEY"] = "your_api_key"

# Create a data directory
os.makedirs("data", exist_ok=True)

# Download the dataset and unzip
os.system("kaggle datasets download -d nigelwilliams/ngsim-vehicle-trajectory-data-us-101 -p data")
os.system("unzip data/ngsim-vehicle-trajectory-data-us-101.zip -d data/ngsim")
