import kaggle as kg
import zipfile
import os

DATA_PATH = "./data"

os.makedirs(DATA_PATH, exist_ok=True)

kg.api.authenticate()
kg.api.competition_download_files(
    competition="jigsaw-toxic-comment-classification-challenge",
    path=DATA_PATH,
    unzip=True,
)

for file in os.listdir(DATA_PATH):
    zipfile.ZipFile(f"{DATA_PATH}/{file}").extractall(DATA_PATH)
    os.remove(f"{DATA_PATH}/{file}")
