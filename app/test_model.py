# ============================================================
# test_model.py
# Local verification of the trained RoBERTa model
# ============================================================

# Import PyTorch.
import torch

# Import the Hugging Face tokenizer and model.
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ------------------------------------------------------------
# MODEL PATH
# ------------------------------------------------------------

MODEL_PATH = r".\model"


# ------------------------------------------------------------
# LOAD TOKENIZER
# ------------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Tokenizer loaded successfully.")


# ------------------------------------------------------------
# LOAD TRAINED ROBERTA MODEL
# ------------------------------------------------------------

print("Loading RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

print("RoBERTa model loaded successfully.")


# ------------------------------------------------------------
# SET MODEL TO EVALUATION MODE
# ------------------------------------------------------------

model.eval()


# ------------------------------------------------------------
# TEST TEXT
# ------------------------------------------------------------

text = "What a beautiful sunset today!"


print()
print("Test text:")
print(text)


# ------------------------------------------------------------
# TOKENIZE TEXT
# ------------------------------------------------------------

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128
)


# ------------------------------------------------------------
# RUN MODEL
# ------------------------------------------------------------

print()
print("Running prediction...")

with torch.no_grad():

    outputs = model(
        **inputs
    )


# ------------------------------------------------------------
# CONVERT LOGITS TO PROBABILITIES
# ------------------------------------------------------------

probabilities = torch.softmax(
    outputs.logits,
    dim=-1
)[0]


# ------------------------------------------------------------
# GET PREDICTED CLASS
# ------------------------------------------------------------

predicted_id = int(
    torch.argmax(
        probabilities
    )
)


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print()
print("==============================================")
print("       LOCAL MODEL VERIFICATION")
print("==============================================")

print(
    "Prediction ID:",
    predicted_id
)

print(
    "Probabilities:",
    probabilities.tolist()
)

print(
    "Model device:",
    next(model.parameters()).device
)

print()
print("MODEL VERIFICATION SUCCESSFUL")