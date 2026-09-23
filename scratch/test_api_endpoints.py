import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_appside_face_recognize_route_exists():
    print("\n--- Testing Appside Face Recognize Route ---")
    # Empty payload should return HTTP 422 Unprocessable Entity (due to missing image parameter)
    response = client.post("/api/v1/appside/face/recognize")
    print(f"Status Code without payload: {response.status_code}")
    assert response.status_code == 422, f"Expected 422 for missing image, got {response.status_code}"
    
    # Check that error response explicitly references field 'image'
    detail = response.json().get("detail", [])
    fields = [d.get("loc", [])[-1] for d in detail if isinstance(d, dict)]
    print(f"Validated missing parameter fields: {fields}")
    assert "image" in fields, "Field 'image' must be the required parameter name"
    print("✅ Endpoint /api/v1/appside/face/recognize verified with required 'image' parameter!")

if __name__ == "__main__":
    test_appside_face_recognize_route_exists()
    print("\n🎉 ALL API ENDPOINT CHECKS PASSED!")
