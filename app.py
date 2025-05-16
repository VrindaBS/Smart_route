from flask import Flask, jsonify, render_template, request
import requests
import json
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)

# Configuration
TOMTOM_API_KEY = "ZyE651ie3JVxkgHqKQmWjD4eyC6GtTSP"
CAMERA_JSON_FEED = "http://data.livetraffic.com/cameras/traffic-cam.json"
MODEL_PATH = "static/final_mobilenetv2_model.h5"

# Step 1: Geocoding with TomTom Search API
def geocode_address(address):
    # Add a timestamp to prevent caching
    timestamp = datetime.now().timestamp()
    url = f"https://api.tomtom.com/search/2/geocode/{address}.json?key={TOMTOM_API_KEY}&countrySet=AU&_={timestamp}"
    print(f"Geocoding URL: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print(f"Geocoding response for {address}: {json.dumps(data, indent=2)}")
        if data["results"]:
            position = data["results"][0]["position"]
            print(f"Geocoded {address} to: ({position['lat']}, {position['lon']})")
            return position["lat"], position["lon"]
    raise Exception(f"Geocoding failed for {address}")

# Step 2: Routing with TomTom Routing API (fetch up to 3 routes)
def get_routes(source_coords, dest_coords, max_routes=3):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{source_coords[0]},{source_coords[1]}:{dest_coords[0]},{dest_coords[1]}/json?key={TOMTOM_API_KEY}&traffic=true&maxAlternatives={max_routes-1}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        routes = []
        for i, route in enumerate(data["routes"][:max_routes]):
            points = [(point["latitude"], point["longitude"]) for point in route["legs"][0]["points"]]
            travel_time = route["summary"]["travelTimeInSeconds"] / 60  # in minutes
            routes.append({"id": i+1, "points": points, "travel_time": travel_time})
        return routes
    raise Exception("Routing failed")

# Step 3: Fetch Camera Data from Public JSON Feed
def get_nsw_traffic_cameras():
    try:
        response = requests.get(CAMERA_JSON_FEED, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "features" in data and isinstance(data["features"], list):
            print("Camera JSON structure sample:", json.dumps(data["features"][:2], indent=2))
            return data["features"]
        raise Exception("No 'features' list found in JSON. Check structure.")
    except Exception as e:
        raise Exception(f"Failed to fetch camera data: {str(e)}")

def find_nearest_cameras(route_points, cameras, max_distance_km=0.5, fallback_distance_km=2.0):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def find_closest_camera_to_point(point_lat, point_lon, cameras, max_distance):
        closest_camera = None
        min_distance = float('inf')
        for camera in cameras:
            if not isinstance(camera, dict):
                continue
            try:
                coordinates = camera.get("geometry", {}).get("coordinates", [0, 0])
                cam_lon = coordinates[0] if len(coordinates) >= 1 else 0
                cam_lat = coordinates[1] if len(coordinates) >= 2 else 0
                properties = camera.get("properties", {})
                cam_url = properties.get("href")
                cam_description = properties.get("title", "Unknown")

                if not cam_url or cam_lat == 0 or cam_lon == 0:
                    continue

                distance = haversine(point_lat, point_lon, float(cam_lat), float(cam_lon))
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    closest_camera = {
                        "url": cam_url,
                        "description": cam_description,
                        "latitude": float(cam_lat),
                        "longitude": float(cam_lon)
                    }
            except Exception as e:
                print(f"Error processing camera {camera}: {str(e)}")
                continue
        return closest_camera, min_distance

    if not isinstance(cameras, list):
        print("Error: Camera data is not a list. Actual type:", type(cameras))
        return []

    if not route_points or len(route_points) < 10:
        print(f"Route has too few points ({len(route_points)}). Selecting closest camera.")
        point_lat, point_lon = route_points[0] if route_points else (-33.8568, 151.2153)  # Fallback to Sydney Opera House
        closest_camera, min_distance = find_closest_camera_to_point(point_lat, point_lon, cameras, float('inf'))
        if closest_camera:
            print(f"Selected closest camera: {closest_camera['description']} at {min_distance:.2f} km")
            return [closest_camera]
        return []

    # Step 1: Determine indices for every 10th point, aiming for up to 3 cameras
    total_points = len(route_points)
    step = max(1, total_points // 10)  # Ensure step is at least 1
    selected_indices = []
    num_cameras = min(3, total_points // step)  # Aim for up to 3 cameras
    if num_cameras == 1:
        selected_indices = [total_points // 2]  # Middle point
    elif num_cameras == 2:
        selected_indices = [step * 2, step * 8]  # Roughly at 20% and 80%
    else:
        selected_indices = [step * 2, step * 5, step * 8]  # Roughly at 20%, 50%, 80%

    print(f"Total points: {total_points}, Step: {step}, Selected indices: {selected_indices}")

    # Step 2: Find the closest camera to each selected point
    camera_images = []
    used_cameras = set()  # To avoid selecting the same camera twice
    for idx in selected_indices:
        if idx >= total_points:
            idx = total_points - 1
        point_lat, point_lon = route_points[idx]
        progress = (idx / (total_points - 1)) * 100 if total_points > 1 else 0
        print(f"Looking for camera near point {idx} (Progress: {progress:.2f}%): [{point_lat}, {point_lon}]")

        # Try strict distance first
        camera, distance = find_closest_camera_to_point(point_lat, point_lon, cameras, max_distance_km)
        if camera and camera["url"] not in used_cameras:
            camera_images.append(camera)
            used_cameras.add(camera["url"])
            print(f"Selected camera: {camera['description']} at {distance:.2f} km (Progress: {progress:.2f}%)")
        else:
            # Fallback to larger distance
            print(f"No camera found within {max_distance_km} km at point {idx}. Trying {fallback_distance_km} km.")
            camera, distance = find_closest_camera_to_point(point_lat, point_lon, cameras, fallback_distance_km)
            if camera and camera["url"] not in used_cameras:
                camera_images.append(camera)
                used_cameras.add(camera["url"])
                print(f"Fallback - Selected camera: {camera['description']} at {distance:.2f} km (Progress: {progress:.2f}%)")

    # Step 3: If fewer than 3 cameras, fill with the closest cameras overall
    if len(camera_images) < 3:
        print(f"Found only {len(camera_images)} cameras. Filling with closest cameras overall.")
        all_candidates = []
        for idx, (point_lat, point_lon) in enumerate(route_points):
            camera, distance = find_closest_camera_to_point(point_lat, point_lon, cameras, float('inf'))
            if camera and camera["url"] not in used_cameras:
                progress = (idx / (total_points - 1)) * 100 if total_points > 1 else 0
                all_candidates.append({
                    "camera": camera,
                    "distance": distance,
                    "progress": progress
                })

        all_candidates.sort(key=lambda x: x["distance"])
        for candidate in all_candidates:
            if len(camera_images) >= 3:
                break
            camera = candidate["camera"]
            if camera["url"] not in used_cameras:
                camera_images.append(camera)
                used_cameras.add(camera["url"])
                print(f"Added closest camera: {camera['description']} at {candidate['distance']:.2f} km "
                      f"(Progress: {candidate['progress']:.2f}%)")

    return camera_images[:3]

# Step 4: Load Pre-trained MobileNetV2 Model and Predict Congestion
def load_mobilenetv2_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Loaded MobileNetV2 model from {MODEL_PATH}")
        return model
    except Exception as e:
        raise Exception(f"Failed to load MobileNetV2 model: {str(e)}")

def preprocess_image(image_url):
    session = requests.Session()
    headers_list = [
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/jpeg,image/png,*/*',
            'Referer': 'https://www.livetraffic.com/'
        },
        {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
            'Accept': 'image/jpeg,image/png,*/*',
            'Referer': 'https://www.livetraffic.com/'
        }
    ]
    try:
        session.get('https://www.livetraffic.com/', timeout=10)
    except Exception as e:
        print(f"Failed to initialize session: {str(e)}")

    for attempt in range(3):
        headers = headers_list[attempt % len(headers_list)]
        try:
            print(f"Downloading image from {image_url} (Attempt {attempt+1}) with headers: {headers}")
            response = session.get(image_url, timeout=15, headers=headers)
            print(f"HTTP Status: {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")
            if response.status_code != 200:
                print(f"Failed to download image: HTTP {response.status_code}")
                continue
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type.lower():
                print(f"Invalid content type: {content_type}")
                print(f"Response content preview: {response.text[:200]}")
                continue
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                print("Failed to decode image")
                continue
            img = cv2.resize(img, (64, 64))  # Adjusted to 64x64 as per MobileNetV2 training
            img = img / 255.0  # Normalize
            print("Image preprocessed successfully")
            return np.expand_dims(img, axis=0)
        except Exception as e:
            print(f"Error processing image {image_url}: {str(e)}")
    return None

def predict_congestion(model, camera_images):
    predictions = []
    for camera in camera_images:
        print(f"Predicting congestion for camera: {camera['description']}")
        img = preprocess_image(camera["url"])
        if img is not None:
            try:
                pred = model.predict(img)
                print(f"Raw model prediction: {pred}")
                congestion_level = np.argmax(pred, axis=1)[0]
                congestion_level = max(1, min(4, congestion_level + 1))  # Map 0-3 to 1-4, clip to 1-4
                levels = {1: "Low", 2: "Low", 3: "Medium", 4: "High"}
                congestion = levels.get(congestion_level, "Medium")
                predictions.append({
                    "url": camera["url"],
                    "description": camera["description"],
                    "latitude": camera["latitude"],
                    "longitude": camera["longitude"],
                    "congestion": congestion
                })
                print(f"Prediction successful: {congestion}")
            except Exception as e:
                print(f"Model prediction failed for {camera['description']}: {str(e)}")
                predictions.append({
                    "url": camera["url"],
                    "description": camera["description"],
                    "latitude": camera["latitude"],
                    "longitude": camera["longitude"],
                    "congestion": "Medium"  # Fallback
                })
        else:
            print(f"Skipping prediction for {camera['description']} due to image processing failure")
            predictions.append({
                "url": camera["url"],
                "description": camera["description"],
                "latitude": camera["latitude"],
                "longitude": camera["longitude"],
                "congestion": "Medium"  # Fallback
            })
    return predictions

# Step 5: Analyze Congestion (camera-based only)
def analyze_traffic(camera_predictions):
    if not camera_predictions:
        print("Warning: No cameras found. Cannot estimate congestion.")
        return "Medium"  # Fallback

    congestion_score = 0
    valid_predictions = 0
    for pred in camera_predictions:
        if pred["congestion"] == "High":
            congestion_score += 3
            valid_predictions += 1
        elif pred["congestion"] == "Medium":
            congestion_score += 2
            valid_predictions += 1
        elif pred["congestion"] == "Low":
            congestion_score += 1
            valid_predictions += 1
        else:
            print(f"Invalid congestion level for camera {pred['description']}: {pred['congestion']}")

    if valid_predictions == 0:
        print("Warning: No valid predictions. Using fallback congestion.")
        return "Medium"  # Fallback

    avg_score = congestion_score / valid_predictions
    if avg_score >= 3:
        return "High"
    elif avg_score >= 2:
        return "Medium"
    else:
        return "Low"

# API Endpoint to Get Route Data
@app.route('/api/routes')
def get_route_data():
    try:
        # Get source and destination from query parameters
        source_address = request.args.get('source')
        dest_address = request.args.get('destination')

        # Log the received query parameters
        print(f"Received query parameters - Source: {source_address}, Destination: {dest_address}")

        if not source_address or not dest_address:
            return jsonify({"error": "Source and destination addresses are required."}), 400

        # Geocoding
        print("Geocoding addresses...")
        source_coords = geocode_address(source_address)
        dest_coords = geocode_address(dest_address)
        print(f"Source: {source_coords}, Destination: {dest_coords}")

        # Routing
        print("Calculating routes...")
        routes = get_routes(source_coords, dest_coords)
        print(f"Found {len(routes)} routes")

        # Fetch Camera Data
        print("Fetching traffic camera data...")
        cameras = get_nsw_traffic_cameras()

        # Load MobileNetV2 Model
        print("Loading MobileNetV2 model...")
        model = load_mobilenetv2_model()

        # Process Routes
        route_results = []
        for route in routes:
            route_id = route["id"]
            route_points = route["points"]
            travel_time = route["travel_time"]
            print(f"\nProcessing Route {route_id} (Travel Time: {travel_time:.2f} minutes)")

            # Find Cameras
            camera_images = find_nearest_cameras(route_points, cameras)
            print(f"Found {len(camera_images)} cameras for Route {route_id}")
            for i, cam in enumerate(camera_images, 1):
                print(f"Camera {i}: {cam['description']} ({cam['url']})")

            # Predict Congestion
            camera_predictions = predict_congestion(model, camera_images)
            for i, pred in enumerate(camera_predictions, 1):
                print(f"Camera {i} ({pred['description']}): Congestion Level - {pred['congestion']}")

            # Analyze Traffic
            overall_congestion = analyze_traffic(camera_predictions)
            print(f"Overall Congestion Level for Route {route_id}: {overall_congestion}")

            route_results.append({
                "id": route_id,
                "points": route_points,
                "travel_time": travel_time,
                "cameras": camera_predictions,
                "congestion": overall_congestion,
                "source": {"latitude": source_coords[0], "longitude": source_coords[1], "name": source_address},
                "destination": {"latitude": dest_coords[0], "longitude": dest_coords[1], "name": dest_address}
            })

        return jsonify(route_results)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Serve HTML Page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4001, debug=True)