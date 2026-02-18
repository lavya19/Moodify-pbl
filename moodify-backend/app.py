from flask import Flask, request, jsonify #web framework for API
import spotipy #python client for Spotify API
from spotipy.oauth2 import SpotifyClientCredentials
import os #loads api keys securely from .env
import json #for analyzing AI's JSON responses
import re
from dotenv import load_dotenv 
from groq import Groq 

load_dotenv()

app = Flask(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))


# ── BPM ranges mapped to energy level ──
# This is what we use to FILTER tracks after getting features
BPM_RANGES = {
    "low":    (50,  85),   # calm, sad, slow ballads
    "medium": (85,  115),  # chill, neutral, moderate
    "high":   (115, 200),  # energetic, party, gym
}

#converts AI's raw text response into JSON
def parse_json_safe(text):
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.replace("```", "").strip()
    return json.loads(text)

#this function uses groq AI to analyze the user's text input 
# and extract key information about their music preferences.
# It returns this information in a structured JSON format for further processing.
def parse_request(user_text):
    """Single AI call — extracts artist, mood, context, energy, genre."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You analyze music requests. Always respond with valid JSON only. No markdown, no explanation."
                },
                {
                    "role": "user",
                    "content": f"""
Analyze this music request: "{user_text}"

Extract:
- artist: the music artist name if mentioned, otherwise null
- mood: the emotional mood (sad, happy, chill, energetic, romantic, angry, melancholic, calm, etc.)
- context: the activity or setting (late night drive, gym, study, heartbreak, party, etc.)
- energy: MUST be one of exactly: "low", "medium", or "high"
  * low = calm, sad, slow, relaxing, sleeping, studying
  * medium = chill, neutral, cruising, romantic
  * high = energetic, party, gym, workout, dancing
- genre_hint: a genre that fits (bollywood, pop, hip-hop, lofi, classical, etc.)

Return ONLY this JSON:
{{
  "artist": "name or null",
  "mood": "mood word",
  "context": "context phrase",
  "energy": "low/medium/high",
  "genre_hint": "genre"
}}
"""
                }
            ],
            temperature=0 #ensures consistent output, no randomness
        )
        raw = response.choices[0].message.content
        print("AI parse:", raw)
        result = parse_json_safe(raw)

        # Clean up null artist
        if not result.get("artist") or str(result["artist"]).lower() in ("null", "none", ""):
            result["artist"] = None

        # Ensure energy is always valid
        if result.get("energy") not in ("low", "medium", "high"):
            result["energy"] = "medium"

        return result

    except Exception as e:
        print("AI parse failed:", e)
        return {"artist": None, "mood": "general", "context": "general", "energy": "medium", "genre_hint": "pop"}


# This is the core function where we ask the AI to estimate the BPM, energy, and valence for each track,
# and then use those estimates to filter the tracks based on the user's requirements.
def estimate_features_for_tracks(tracks, mood, energy):
    """
    AI estimates BPM, energy, valence per track.
    Crucially, we tell the AI what energy level we want
    so it gives us accurate estimates, not generic ones.
    """
    try:
        bpm_min, bpm_max = BPM_RANGES[energy]
        track_list = [
            {"title": t["name"], "artist": t["artists"][0]["name"]}
            for t in tracks[:15]
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a music expert. Return only valid JSON arrays, no markdown, no explanation."
                },
                {
                    "role": "user",
                    "content": f"""
For each track below, estimate its actual audio characteristics.
Be accurate based on your real knowledge of each song.

Mood: {mood}
Energy level: {energy} (this means BPM should typically be between {bpm_min} and {bpm_max})

Estimate for each track:
- bpm: integer, the actual tempo of this specific song
- energy: float 0.0-1.0 (0=very calm, 1=very intense)
- valence: float 0.0-1.0 (0=sad/negative, 1=happy/positive)

Tracks:
{json.dumps(track_list)}

Return a JSON array in the exact same order, nothing else:
[{{"bpm": 75, "energy": 0.3, "valence": 0.2}}, ...]
"""
                }
            ],
            temperature=0.1
        )
        raw = response.choices[0].message.content
        print("AI features:", raw[:150])
        return parse_json_safe(raw)

    except Exception as e:
        print("Feature estimation failed:", e)
        bpm_mid = (BPM_RANGES[energy][0] + BPM_RANGES[energy][1]) // 2
        defaults = {
            "low":    {"bpm": bpm_mid, "energy": 0.25, "valence": 0.3},
            "medium": {"bpm": bpm_mid, "energy": 0.55, "valence": 0.5},
            "high":   {"bpm": bpm_mid, "energy": 0.85, "valence": 0.7},
        }
        return [defaults[energy]] * len(tracks)

#
def filter_by_bpm(playlist, energy):
    """
    Filter out tracks whose BPM doesn't match the requested energy level.
    we use the three predefined BPM ranges for low, medium, and high energy.
    If filtering removes everything, relax the range by ±15 BPM and try again.
    If still nothing, return all sorted by BPM closeness to target.
    """
    bpm_min, bpm_max = BPM_RANGES[energy]

    # Strict filter first
    filtered = [t for t in playlist if isinstance(t["bpm"], (int, float)) and bpm_min <= t["bpm"] <= bpm_max]

    if filtered:
        print(f"BPM filter ({bpm_min}-{bpm_max}): {len(playlist)} → {len(filtered)} tracks")
        return filtered

    # Relaxed filter (±15 BPM buffer)
    relaxed = [t for t in playlist if isinstance(t["bpm"], (int, float)) and (bpm_min - 15) <= t["bpm"] <= (bpm_max + 15)]

    if relaxed:
        print(f"BPM filter relaxed: {len(playlist)} → {len(relaxed)} tracks")
        return relaxed

    # Last resort: sort by how close BPM is to the middle of the target range
    target_mid = (bpm_min + bpm_max) / 2
    print("BPM filter: no match found, sorting by BPM proximity")
    return sorted(playlist, key=lambda t: abs(t.get("bpm", 999) - target_mid))


# Basic health check endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Moodify AI Backend Running 🎧"})


# This is the main endpoint that handles music recommendation requests.
# Also the part where we combine all the steps: parsing user input, searching Spotify, estimating features, and filtering tracks.
# This is where we send our api response back to the local url using hoppscotch.io
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True)
        user_text = data.get("text", "").strip()
        if not user_text:
            return jsonify({"error": "No input provided"}), 400

        print(f"\n🎵 Request: '{user_text}'")

        # Step 1: Parse intent
        parsed = parse_request(user_text)
        artist  = parsed["artist"]
        mood    = parsed["mood"]
        context = parsed["context"]
        energy  = parsed["energy"]
        genre   = parsed["genre_hint"]

        print(f"Artist: {artist} | Mood: {mood} | Energy: {energy} | BPM target: {BPM_RANGES[energy]}")

        # Step 2: Build Spotify query
        # We run 2 searches and combine — one normal, one with "rare" offset
        # to get beyond just the top hits
        all_tracks = []
        seen_ids = set()

        if artist:
            queries = [f"artist:{artist}", f"artist:{artist}"]
            offsets = [0, 20]  # second search starts at offset 20 = deeper cuts
        else:
            base_query = f"{mood} {genre} {context}".strip()
            queries = [base_query, f"{genre} {mood} underground", f"{genre} {context} indie"]
            offsets = [0, 0, 0]

        for query, offset in zip(queries, offsets):
            try:
                results = sp.search(q=query, type="track", limit=50, market="IN", offset=offset)
                for t in results["tracks"]["items"]:
                    if t["id"] not in seen_ids:
                        seen_ids.add(t["id"])
                        all_tracks.append(t)
            except Exception as e:
                print(f"Search failed for '{query}':", e)

        tracks = all_tracks

        if not tracks:
            return jsonify({"input": user_text, "parsed_intent": parsed, "playlist": [], "message": "No tracks found."})

        # Step 3: Filter by artist if detected
        if artist:
            filtered = [
                t for t in tracks
                if artist.lower() in t["artists"][0]["name"].lower()
                or t["artists"][0]["name"].lower() in artist.lower()
            ]
            tracks = filtered if filtered else tracks

        tracks = tracks[:15]

        # Step 4: Get AI-estimated features (BPM, energy, valence)
        features_list = estimate_features_for_tracks(tracks, mood, energy)

        # Step 5: Build raw playlist with features attached
        raw_playlist = []
        for track, features in zip(tracks, features_list):
            raw_playlist.append({
                "song_name": track["name"],
                "artist": track["artists"][0]["name"],
                "album": track["album"]["name"],
                "spotify_url": track["external_urls"]["spotify"],
                "preview_url": track["preview_url"],
                "bpm": features.get("bpm", 0),
                "energy": features.get("energy", 0),
                "valence": features.get("valence", 0)
            })

        # Step 6: Filter by BPM range matching the mood/energy
        playlist = filter_by_bpm(raw_playlist, energy)

        bpm_min, bpm_max = BPM_RANGES[energy]

        return jsonify({
            "input": user_text,
            "detected_artist": artist,
            "mood": mood,
            "context": context,
            "energy_level": energy,
            "genre_hint": genre,
            "bpm_target_range": f"{bpm_min}-{bpm_max}",
            "total_tracks": len(playlist),
            "playlist": playlist
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
