import os
from dotenv import load_dotenv
load_dotenv()

FIREWORKS_API_KEY = os.getenv("MY_KEY")
MODEL_ID = "gemini-2.0-flash" 
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1/chat/completions",
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"
GOOGLE_URL_DETECT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"
MODEL_VL = "google/gemma-4-26b-a4b-it:free"

REPO_SETTINGS = {
    "default": {
        "owner": "grishakalinin2014-alt",
        "repo": "UAV-Agent",
        "branch": "main",
        "folder": "images/blank"
    },
    "fire": {
        "owner": "grishakalinin2014-alt",
        "repo": "UAV-Agent",
        "branch": "main",
        "folder": "images/fire_buildings"
    },
    "hogweed_thickets": {
        "owner": "grishakalinin2014-alt",
        "repo": "UAV-Agent",
        "branch": "main",
        "folder": "images/hogweed_thickets"
    }
}