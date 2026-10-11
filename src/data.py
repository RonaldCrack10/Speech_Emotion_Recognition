## The objective here is to create a common dataset from the Ravedess and Tess Datasets.

'''
RAVEDESS Encoding
03-01-06-01-02-01-12.wav heißt modality-channel-emotion-intensity-statement-repetition-actor. Emotion ist das 3. Feld, Actor das letzte.


TESS Encoding
TESS hat Namen wie OAF_back_angry.wav: Sprecherin, Wort, Emotion. ps steht für pleasant surprise.

'''

import pandas as pd
from pathlib import Path

ravdess_map = {

    "1" : "neutral",
    "2" : "None",
    "3" : "happy",
    "4" : "sad",
    "5" : "angry",
    "6" : "fear",
    "7" : "disgust",
    "8" : "surprise"

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
        rows.append(dict(path = str(p), label = label_to_id[emo],
                         speaker = f"rav_{int(parts[6]):02d})", corpus = "ravdess"))


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