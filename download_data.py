import os
import zipfile
import urllib.request

def download_and_extract(url, output_zip, extract_dir):
    if not os.path.exists(output_zip):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, output_zip)

    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Done.")

if __name__ == "__main__":
    DATA_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    ZIP_PATH = "data/kagglecatsanddogs.zip"
    EXTRACT_DIR = "data"

    os.makedirs("data", exist_ok=True)
    download_and_extract(DATA_URL, ZIP_PATH, EXTRACT_DIR)
