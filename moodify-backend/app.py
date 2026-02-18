from flask import Flask, request, jsonify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import spacy
from transformers import pipeline
# AI Context Classifier (Zero-shot classification)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Candidate music contexts (NOT hardcoded logic, just labels for AI to choose from)
CANDIDATE_CONTEXTS = [
    "relaxing music",
    "focus music",
    "workout music",
    "sad emotional music",
    "happy upbeat music",
    "party music",
    "lofi study music",
    "chill music"
]


# Load environment variables
load_dotenv()

app = Flask(__name__)

# Spotify Authentication (Client Credentials Flow - No login required)
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

nlp=spacy.load("en_core_web_sm")

def detect_context(user_input):
    # Step 1: Use AI to classify user intent/context
    result = classifier(user_input, CANDIDATE_CONTEXTS)

    # Top AI-predicted context
    ai_context = result["labels"][0]

    # Step 2: Preserve original query (VERY IMPORTANT)
    # This keeps artist names like Drake intact
    enhanced_query = f"{user_input} {ai_context}"

    return enhanced_query


# Health check route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Moodify Backend is Running 🎵",
        "status": "success"
    })

# Main recommendation API
@app.route("/recommend", methods=["POST"])
def recommend_music():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide input text"}), 400

    user_text = data["text"]

    # Detect context (lightweight logic, no heavy AI)
    context_query = detect_context(user_text)

    # Search tracks from Spotify
    results = sp.search(
    q=context_query,
    type="track",
    market="IN",
    limit=10
    )

    tracks_list = []

    for track in results["tracks"]["items"]:
        tracks_list.append({
            "song_name": track["name"],
            "artist": track["artists"][0]["name"],
            "album": track["album"]["name"],
            "spotify_url": track["external_urls"]["spotify"],
            "preview_url": track["preview_url"]
        })

    return jsonify({
        "input_text": user_text,
        "detected_context": context_query,
        "total_recommendations": len(tracks_list),
        "recommended_tracks": tracks_list
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
