import pandas as pd
import numpy as np
from transformers import pipeline
import torch
from glob import glob
import spacy

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


#Divide subsentences the first script
script_per_episode = " ".join(df.iloc[:3]["script"])

nlp = spacy.load("en_core_web_sm")




def get_themes(script_per_episode):
    doc = nlp(script_per_episode)
    script_sentences = [sent.text for sent in doc.sents]
    
    sentence_batch_size = 20
    script_batches = []
    
    for index in range(0,len(script_sentences), sentence_batch_size):
        sent = " ".join(script_sentences[index:index + sentence_batch_size])
        script_batches.append(sent)
        
    
    #Run Model
    theme_output = theme_classifier(
        script_batches,
        theme_list,
        multi_label = True
    )
    
    
    #Wrangle Output
    themes = {}
    
    for output in theme_output:
        for label,score in zip(output["labels"], output["scores"]):
            if label not in themes:
                themes[label] = []
            themes[label].append(score)
    
    
    
    #Take means of scores
    themes = {key: np.mean(np.array(value)) for key, value in themes.items()}
    
    return themes

get_themes(script_per_episode)