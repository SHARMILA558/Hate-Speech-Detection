# ============================================================
# moderation.py
# RoBERTa-based AI Content Moderation Engine
# ============================================================

# Import PyTorch for model inference.
import torch

# Import Hugging Face tokenizer and classification model.
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Import utility functions and model path.
from utils import (
    MODEL_DIR,
    load_model_config
)


# ============================================================
# LOAD DEPLOYMENT CONFIGURATION
# ============================================================

# Read the configuration generated during model training.
CONFIG = load_model_config()


# ============================================================
# SELECT COMPUTING DEVICE
# ============================================================

# Use CUDA when a compatible GPU is available.
# Otherwise use the CPU.
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

# Load the tokenizer saved together with the trained model.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)


# ============================================================
# LOAD TRAINED ROBERTA MODEL
# ============================================================

# Load the fine-tuned RoBERTa classification model.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

# Move the model to GPU or CPU.
model.to(
    DEVICE
)


# ============================================================
# EVALUATION MODE
# ============================================================

# Disable training behaviour such as dropout.
model.eval()


# ============================================================
# LABEL MAPPING
# ============================================================

# Convert JSON string keys back to integers.
ID2LABEL = {
    int(key): value
    for key, value in CONFIG["id2label"].items()
}


# ============================================================
# MAXIMUM TOKEN LENGTH
# ============================================================

# Use the same maximum sequence length used during training.
MAX_LENGTH = int(
    CONFIG.get(
        "max_length",
        128
    )
)


# ============================================================
# MODERATION THRESHOLDS
# ============================================================

# Probability required for automatic hate-speech removal.
HATE_REMOVE_THRESHOLD = float(
    CONFIG.get(
        "hate_remove_threshold",
        0.90
    )
)


# Probability above which human review is recommended.
REVIEW_THRESHOLD = float(
    CONFIG.get(
        "review_threshold",
        0.50
    )
)


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_text(text):
    """
    Run RoBERTa classification on one piece of text.

    Returns:
        label
        confidence
        class probabilities
    """

    # Make sure the input is a string.
    if not isinstance(text, str):

        raise TypeError(
            "Text must be a string."
        )


    # Remove unnecessary whitespace.
    text = text.strip()


    # Prevent empty submissions.
    if not text:

        raise ValueError(
            "Text cannot be empty."
        )


    # --------------------------------------------------------
    # TOKENIZE INPUT
    # --------------------------------------------------------

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt"
    )


    # --------------------------------------------------------
    # MOVE INPUT TO SAME DEVICE AS MODEL
    # --------------------------------------------------------

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }


    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **encoded
        )


    # --------------------------------------------------------
    # CONVERT LOGITS TO PROBABILITIES
    # --------------------------------------------------------

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]


    # --------------------------------------------------------
    # FIND PREDICTED CLASS
    # --------------------------------------------------------

    predicted_id = int(
        torch.argmax(
            probabilities
        ).item()
    )


    # --------------------------------------------------------
    # GET CONFIDENCE
    # --------------------------------------------------------

    confidence = float(
        probabilities[
            predicted_id
        ].item()
    )


    # --------------------------------------------------------
    # CREATE READABLE PROBABILITY DICTIONARY
    # --------------------------------------------------------

    probability_dict = {}

    for index in range(
        len(ID2LABEL)
    ):

        label = ID2LABEL[
            index
        ]

        probability_dict[
            label
        ] = float(
            probabilities[
                index
            ].item()
        )


    # --------------------------------------------------------
    # RETURN MODEL RESULT
    # --------------------------------------------------------

    return {

        "label_id":
            predicted_id,

        "label":
            ID2LABEL[
                predicted_id
            ],

        "confidence":
            confidence,

        "probabilities":
            probability_dict
    }


# ============================================================
# MODERATION AGENT
# ============================================================

def moderate_text(text):
    """
    Convert RoBERTa prediction into a moderation decision.

    Possible decisions:

        ALLOW
        REVIEW
        REMOVE
    """

    # Get the model prediction.
    prediction = predict_text(
        text
    )


    # Extract probability values.
    probabilities = prediction[
        "probabilities"
    ]


    # Get hate-speech probability.
    hate_probability = probabilities[
        "Hate Speech"
    ]


    # Get offensive-language probability.
    offensive_probability = probabilities[
        "Offensive"
    ]


    # Get safe-content probability.
    neither_probability = probabilities[
        "Neither"
    ]


    # ========================================================
    # DECISION 1 — HIGH CONFIDENCE HATE SPEECH
    # ========================================================

    if hate_probability >= HATE_REMOVE_THRESHOLD:

        decision = "REMOVE"

        reason = (
            "High-confidence hate speech detected."
        )


    # ========================================================
    # DECISION 2 — POSSIBLE HATE SPEECH
    # ========================================================

    elif hate_probability >= REVIEW_THRESHOLD:

        decision = "REVIEW"

        reason = (
            "Potential hate speech detected. "
            "Human moderation review is recommended."
        )


    # ========================================================
    # DECISION 3 — OFFENSIVE CONTENT
    # ========================================================

    elif offensive_probability >= REVIEW_THRESHOLD:

        decision = "REVIEW"

        reason = (
            "Offensive language detected. "
            "Human moderation review is recommended."
        )


    # ========================================================
    # DECISION 4 — SAFE CONTENT
    # ========================================================

    else:

        decision = "ALLOW"

        reason = (
            "No significant harmful-content "
            "signal was detected."
        )


    # ========================================================
    # RETURN COMPLETE MODERATION RESULT
    # ========================================================

    return {

        "text":
            text,

        "classification":
            prediction[
                "label"
            ],

        "confidence":
            prediction[
                "confidence"
            ],

        "probabilities":
            probabilities,

        "decision":
            decision,

        "reason":
            reason
    }