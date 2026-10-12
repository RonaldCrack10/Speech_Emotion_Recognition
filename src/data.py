## The objective here is to create a common dataset from the Ravedess and Tess Datasets.

'''
RAVEDESS Encoding
03-01-06-01-02-01-12.wav heißt modality-channel-emotion-intensity-statement-repetition-actor. Emotion ist das 3. Feld, Actor das letzte.


TESS Encoding
TESS hat Namen wie OAF_back_angry.wav: Sprecherin, Wort, Emotion. ps steht für pleasant surprise.

'''

import pandas as pd
from pathlib import Path


ravdess_path = r"C:\\Users\\rashi\Desktop\\Studium\\Fünftes Semester\\Advances In AI\\Datasets\\Ravedess dataset"
tess_path = r"C:\\Users\\rashi\\Desktop\\Studium\\Fünftes Semester\\Advances In AI\\Datasets\\TESS Toronto emotional speech set data"

ravdess_map = {

    "01" : "neutral",
    "02" : None,
    "03" : "happy",
    "04" : "sad",
    "05" : "angry",
    "06" : "fear",
    "07" : "disgust",
    "08" : "surprise"

}

tess_map = {

    "angry": "angry",
    "disgust": "disgust", 
    "fear": "fear", 
    "happy": "happy",
    "neutral": "neutral", 
    "sad": "sad", 
    "ps": "surprise",
}


labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
label_to_id = {l : i for i, l in enumerate(labels)}


def build_metadata(ravdess_dir:str , tess_dir: str) -> pd.DataFrame: # Get the metadata from audio files

    rows = []
    for p in Path(ravdess_dir).rglob("*.wav"):
        parts = p.stem.split('-')
        if len(parts) != 7:
            continue
        emo = ravdess_map.get(parts[2]) # emotion
        if emo is None:
            continue
        rows.append(dict(path = str(p), 
                        label = label_to_id[emo],
                         speaker = f"rav_{int(parts[6]):02d}", 
                         corpus = "ravdess"))


    for p in Path(tess_dir).rglob("*.wav"):
        parts = p.stem.split("_")
        emo = tess_map.get(parts[-1].lower())
        if emo is None:
            continue
        rows.append(dict(path= str(p),
                         label = label_to_id[emo],
                         speaker = f"tess_{parts[0].lower()}",
                         corpus = "tess"))

    return pd.DataFrame(rows)


def split(df: pd.DataFrame,
          val_actors: set = {15, 16, 17, 18, 19},
          test_actors: set = {20, 21, 22, 23, 24}): # split the dataset in such a way that no speaker is in 2 or 3 different sets

    val_id = [f"rav_{a}" for a in val_actors]
    test_id = [f"rav_{b}" for b in test_actors]

    val = df[df['speaker'].isin(val_id)]
    test = df[df['speaker'].isin(test_id)]

    train = df[~df["speaker"].isin(set(val_id) | set(test_id))] # |  Union between the 2 sets... ~ inverse

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == '__main__':
    df = build_metadata(ravdess_dir = ravdess_path, tess_dir = tess_path)
    train, val, test = split(df= df)
    print(
        "Train size: ", train.shape,
        "Val size: ", val.shape,
        "Test size: ", test.shape
    )
