import pandas as pd
from transformers import pipeline
from nltk import sent_tokenize
import nltk
import torch
from glob import glob


nltk.download("punkt")

# Load the zero-shot classification model from Hugging Face
model_name="valhalla/distilbart-mnli-12-3"
device =0 if torch.cuda.is_available() else -1


# Function to load the classification model
def load_model(device):
    theme_classifer = pipeline(
        "zero-shot-classification",
        model = model_name,
        device=0 if device == 0 else -1
    )
    return theme_classifer

theme_classifier = load_model(device)


# Test the classifier with an example sentence
theme_list = ["friendship", "hope", "sacrifice", "battle", "self development", "betrayal", "love", "dialogue"]

theme_classifier(
    "I gave her a left hook then an uppercut then a kick",
    theme_list,
)


# Function to load and process subtitle files
def load_subtitles(dataset_path):
    subtitles_path = glob(dataset_path + "/*.ass")
    
    subtitles = []
    episode_num = []
    
    for path in subtitles_path:
        
        with open(path, "r") as file:
            lines = file.readlines()
            lines = lines[27:]
            lines = [",".join(line.split(",")[9:]) for line in lines]
            lines = [line.replace("\\N", "") for line in lines]
            
        script = " ".join(lines)
        
        episode = int(path.split("-")[-1].split(".")[0].strip())
        
        subtitles.append(script)
        episode_num.append(episode)
        
    df =pd.DataFrame.from_dict({"episode": episode_num, "script": subtitles})
    
    return df


 # Define and load the path to the subtitle dataset
dataset_path = "../data/Subtitles"
df = load_subtitles(dataset_path)
