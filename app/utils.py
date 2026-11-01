# ============================================================
# utils.py
# Utility functions for the HateSpeechAI moderation agent
# ============================================================

# Import JSON for reading the model configuration file.
import json

# Import Path for safe cross-platform file handling.
from pathlib import Path


# ============================================================
# PROJECT DIRECTORY
# ============================================================

# app/ is the directory containing this file.
APP_DIR = Path(__file__).resolve().parent


# The project root is one directory above app/.
PROJECT_DIR = APP_DIR.parent


# ============================================================
# MODEL DIRECTORY
# ============================================================

# This is where the trained RoBERTa model is stored.
MODEL_DIR = PROJECT_DIR / "model"


# ============================================================
# DEPLOYMENT CONFIGURATION
# ============================================================

# Configuration file generated/saved with the trained model.
CONFIG_FILE = MODEL_DIR / "deployment_config.json"


# ============================================================
# CHECK MODEL DIRECTORY
# ============================================================

def check_model_directory():
    """
    Verify that the trained model directory exists.
    """

    # Check whether the model directory exists.
    if not MODEL_DIR.exists():

        raise FileNotFoundError(
            f"Model directory was not found: {MODEL_DIR}"
        )


# ============================================================
# LOAD MODEL CONFIGURATION
# ============================================================

def load_model_config():
    """
    Load deployment configuration.

    If deployment_config.json does not exist,
    safe default configuration is returned.
    """

    # Make sure the model directory exists.
    check_model_directory()


    # If the configuration file exists,
    # read it from disk.
    if CONFIG_FILE.exists():

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            config = json.load(
                file
            )

        return config


    # --------------------------------------------------------
    # DEFAULT CONFIGURATION
    # --------------------------------------------------------

    # Your trained model uses:
    #
    # 0 = Hate Speech
    # 1 = Offensive
    # 2 = Neither
    #
    # These defaults allow the application to work even
    # if deployment_config.json was not included in the
    # downloaded model archive.

    return {

        "id2label": {

            "0": "Hate Speech",

            "1": "Offensive",

            "2": "Neither"
        },

        "label2id": {

            "Hate Speech": 0,

            "Offensive": 1,

            "Neither": 2
        },

        "max_length": 128,

        "hate_remove_threshold": 0.90,

        "review_threshold": 0.50
    }