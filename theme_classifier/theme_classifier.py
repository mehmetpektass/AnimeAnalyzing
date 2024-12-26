from transformers import pipeline
from nltk import sent_tokenize
import nltk
import torch
from glob import glob

nltk.download("punkt")


#Load Model
model_name="valhalla/distilbart-mnli-12-3"
device =0 if torch.cuda.is_available() else -1


def load_model(device):
    theme_classifer = pipeline(
        "zero-shot-classification",
        model = model_name,
        device=0 if device == 0 else -1
    )
    return theme_classifer

theme_classifier = load_model(device)

theme_list = ["friendship", "hope", "sacrifice", "battle", "self development", "betrayal", "love", "dialogue"]

theme_classifier(
    "I gave her a left hook then an uppercut then a kick",
    theme_list,
)