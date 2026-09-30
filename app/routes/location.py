import os
import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "AIzaSyDac-XVwE7B5drHxpeFq-qQZp6vmXWf3MI")

@router.post("/api/v1/location/fallback")
def get_location_from_google():
    """Fallback location lookups via Wi-Fi/Cell signals when browser GPS is unavailable"""
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}"
    
    # Optional parameters: Wi-Fi access points or cell towers from client device
    payload = {
        "considerIp": True
    }
    
    response = requests.post(url, json=payload)
    data = response.json()
    
    if "location" in data:
        return {
            "latitude": data["location"]["lat"],
            "longitude": data["location"]["lng"],
            "accuracy": data["accuracy"]  # Accuracy radius in meters
        }
    
    raise HTTPException(status_code=400, detail="Could not determine location")
