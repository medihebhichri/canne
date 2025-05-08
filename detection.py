import cv2
from ultralytics import YOLO
import time
import urllib.request
import numpy as np
from flask import Flask, request, jsonify
import threading
import requests

# Initialize Flask app for receiving triggers from ESP32
app = Flask(__name__)

# Global variables
processing_active = False
obstacle_direction = "none"
detection_start_time = 0
detection_duration = 10  # seconds
esp32_ip = None  # Will be set when receiving trigger

# Load YOLO model
print("Loading YOLO model...")
model = YOLO('best.pt')
model.conf = 0.5  # Lower the confidence threshold to 0.a5

# ESP32 camera URL
camera_url = "http://192.168.137.31/capture"


# Function to get a frame from the ESP32-CAM
def get_esp32_frame():
    try:
        img_resp = urllib.request.urlopen(camera_url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        return True, frame
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return False, None


# Function to send detection to ESP32
def send_detection_to_esp32(object_name):
    global esp32_ip

    if not esp32_ip:
        print("ESP32 IP address not available")
        return False

    # Construct the URL
    url = f"http://{esp32_ip}/detection?object={object_name}"

    try:
        print(f"Sending detection '{object_name}' to ESP32 at {url}")
        response = requests.get(url, timeout=3)
        print(f"Response status: {response.status_code}, text: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending detection: {e}")
        return False


# Route to receive triggers from ESP32
@app.route('/trigger', methods=['GET'])
def trigger_detection():
    global processing_active, obstacle_direction, detection_start_time, esp32_ip

    # Get parameters
    direction = request.args.get('direction', 'unknown')
    esp32_ip = request.args.get('esp32_ip', None)

    print(f"Received trigger from ESP32 with direction: {direction}, IP: {esp32_ip}")

    # If detection is already active, just update the direction
    if processing_active:
        obstacle_direction = direction
        return jsonify({"status": "Detection already running", "direction": direction})

    # Start new detection session
    processing_active = True
    obstacle_direction = direction
    detection_start_time = time.time()

    # Start detection in a separate thread
    threading.Thread(target=start_detection_session).start()

    return jsonify({
        "status": "Detection started",
        "direction": direction,
        "duration": detection_duration,
        "esp32_ip": esp32_ip
    })


# Function to run the detection session
def start_detection_session():
    global processing_active, obstacle_direction

    print(f"Starting detection session for obstacle on: {obstacle_direction}")
    if esp32_ip:
        print(f"ESP32 IP address: {esp32_ip}")
    else:
        print("WARNING: ESP32 IP address not provided!")

    # Person detection state
    person_detected = False

    # Create window for display
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

    # Run detection for specified duration
    start_time = time.time()

    while processing_active and (time.time() - start_time < detection_duration):
        # Get a frame
        ret, frame = get_esp32_frame()

        if not ret or frame is None:
            print("Failed to get frame, retrying...")
            time.sleep(0.5)
            continue

        # Run YOLO detection
        results = model(frame)

        # Check for person detection
        if len(results) > 0:
            # Extract class names from results
            class_names = []

            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        class_name = result.names[cls_id]

                        print(f"Detected: {class_name} with confidence {conf:.2f}")
                        class_names.append(class_name)

                        # Only notify once for each class type
                        if class_name == 'person' and not person_detected and conf >= 0.5:
                            print("PERSON DETECTED! Sending to ESP32")
                            if send_detection_to_esp32('person'):
                                print("Successfully sent person detection to ESP32")
                                person_detected = True

        # Display the results
        annotated_frame = results[0].plot()

        # Add info text
        remaining_time = max(0, detection_duration - (time.time() - start_time))
        cv2.putText(annotated_frame, f"Time: {remaining_time:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show if person detected
        if person_detected:
            cv2.putText(annotated_frame, "PERSON DETECTED", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Detection', annotated_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    processing_active = False

    print("Detection session completed")
    if person_detected:
        print("RESULTS: Person was detected during this session")
    else:
        print("RESULTS: No person was detected during this session")


# Main function
def main():
    # Start the Flask server
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False)).start()

    print("Server started at http://0.0.0.0:5000/")
    print("Waiting for obstacle detection triggers from ESP32...")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated")


if __name__ == "__main__":
    main()
