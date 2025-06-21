import serial
import threading
import time
import keyboard  # Admin access required for non-Windows
import requests
import socketio
from collections import deque
import json

# === CONFIG ===
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust to your ESP32 port
BAUD_RATE = 115200
FLASK_SERVER_URL = 'http://localhost:5000'
LABEL_KEYS = {
    'w': 'walking',
    's': 'standing',
    'f': 'freezing'
}
BUFFER_DURATION = 5  # seconds of rolling data

# === Global Variables ===
data_buffer = deque()  # holds (timestamp, ax, ay, az, gx, gy, gz)
label_events = deque()  # holds (timestamp, label)
sio = socketio.Client()
serial_connected = False
server_connected = False
current_session_active = False

# === SocketIO Events ===
@sio.event
def connect():
    global server_connected
    server_connected = True
    print("[SocketIO] Connected to Flask server")

@sio.event
def disconnect():
    global server_connected
    server_connected = False
    print("[SocketIO] Disconnected from Flask server")

@sio.event
def status(data):
    print(f"[SocketIO] Server status: {data['message']}")

# === Thread: Serial Reading ===
def read_serial():
    global serial_connected
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        serial_connected = True
        print(f"[Serial] Connected to {SERIAL_PORT}")
        
        while True:
            if not serial_connected:
                break
                
            try:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    # Assume format: ax,ay,az,gx,gy,gz
                    values = line.split(',')
                    if len(values) == 6:
                        ax, ay, az, gx, gy, gz = map(float, values)
                        timestamp = time.time()
                        
                        # Store in buffer
                        data_buffer.append((timestamp, ax, ay, az, gx, gy, gz))
                        
                        # Trim old data
                        while data_buffer and timestamp - data_buffer[0][0] > BUFFER_DURATION:
                            data_buffer.popleft()
                        
                        # Send to Flask server if connected and session is active
                        if server_connected and current_session_active:
                            send_imu_data_to_server(ax, ay, az, gx, gy, gz, timestamp)
                            
            except UnicodeDecodeError:
                continue
            except ValueError as e:
                print(f"[Serial] Error parsing line: {line} | {e}")
            except Exception as e:
                print(f"[Serial] Unexpected error: {e}")
                
    except serial.SerialException as e:
        print(f"[Serial] Failed to connect to {SERIAL_PORT}: {e}")
        serial_connected = False
    except Exception as e:
        print(f"[Serial] Unexpected error: {e}")
        serial_connected = False

def send_imu_data_to_server(ax, ay, az, gx, gy, gz, timestamp):
    """Send IMU data to Flask server via SocketIO"""
    try:
        imu_data = {
            'timestamp': timestamp,
            'acc_x': round(ax, 3),
            'acc_y': round(ay, 3),
            'acc_z': round(az, 3),
            'gyro_x': round(gx, 3),
            'gyro_y': round(gy, 3),
            'gyro_z': round(gz, 3)
        }
        
        # The Flask server will handle storing this data
        # We just emit it like the simulated data
        sio.emit('real_imu_data', imu_data)
        
    except Exception as e:
        print(f"[SocketIO] Error sending IMU data: {e}")

# === Thread: Key Labeling ===
def key_listener():
    print("[Key] Press W (walk), S (stand), F (freeze)")
    print("[Key] Press Q to quit, R to start/stop recording")
    
    while True:
        try:
            # Check for label keys
            for key, label in LABEL_KEYS.items():
                if keyboard.is_pressed(key):
                    timestamp = time.time()
                    label_events.append((timestamp, label))
                    print(f"[Key] {label.upper()} at {timestamp:.2f}")
                    
                    # Send annotation to server
                    if server_connected:
                        send_annotation_to_server(label)
                    
                    time.sleep(0.2)  # Prevent repeat labels
            
            # Check for control keys
            if keyboard.is_pressed('r'):
                toggle_recording()
                time.sleep(0.5)  # Prevent repeat
                
            if keyboard.is_pressed('q'):
                print("[Key] Quitting...")
                cleanup_and_exit()
                break
                
            time.sleep(0.05)  # Small delay to prevent high CPU usage
            
        except Exception as e:
            print(f"[Key] Error in key listener: {e}")
            time.sleep(0.1)

def send_annotation_to_server(label):
    """Send state annotation to Flask server"""
    try:
        response = requests.post(
            f'{FLASK_SERVER_URL}/annotate_state',
            json={'state': label},
            timeout=5
        )
        if response.status_code == 200:
            print(f"[Server] State '{label}' annotated successfully")
        else:
            print(f"[Server] Failed to annotate state: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[Server] Error sending annotation: {e}")

def toggle_recording():
    """Start or stop recording session"""
    global current_session_active
    
    try:
        if not current_session_active:
            # Start recording
            response = requests.post(f'{FLASK_SERVER_URL}/start_session', timeout=5)
            if response.status_code == 200:
                data = response.json()
                current_session_active = True
                print(f"[Server] Recording started - Session: {data.get('session_id', 'Unknown')}")
            else:
                print(f"[Server] Failed to start recording: {response.status_code}")
        else:
            # Stop recording
            response = requests.post(f'{FLASK_SERVER_URL}/stop_session', timeout=5)
            if response.status_code == 200:
                current_session_active = False
                print("[Server] Recording stopped")
            else:
                print(f"[Server] Failed to stop recording: {response.status_code}")
                
    except requests.exceptions.RequestException as e:
        print(f"[Server] Error toggling recording: {e}")

def connect_to_server():
    """Connect to Flask SocketIO server"""
    global server_connected
    
    try:
        sio.connect(FLASK_SERVER_URL)
        server_connected = True
        return True
    except Exception as e:
        print(f"[SocketIO] Failed to connect to server: {e}")
        server_connected = False
        return False

def cleanup_and_exit():
    """Clean up connections and exit"""
    global serial_connected, server_connected
    
    print("[Cleanup] Shutting down...")
    
    # Stop recording if active
    if current_session_active:
        try:
            requests.post(f'{FLASK_SERVER_URL}/stop_session', timeout=2)
        except:
            pass
    
    # Disconnect from server
    if server_connected:
        sio.disconnect()
    
    # Stop serial connection
    serial_connected = False
    
    print("[Cleanup] Shutdown complete")
    exit(0)

def check_server_status():
    """Check if Flask server is running"""
    try:
        response = requests.get(f'{FLASK_SERVER_URL}/', timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main function to start all threads and connections"""
    print("=" * 50)
    print("ESP32 IMU Connector for FOG Detection System")
    print("=" * 50)
    
    # Check if Flask server is running
    print("[Server] Checking Flask server connection...")
    if not check_server_status():
        print(f"[Server] ERROR: Flask server not running at {FLASK_SERVER_URL}")
        print("[Server] Please start the Flask server first: python app.py")
        return
    
    print("[Server] Flask server is running")
    
    # Connect to SocketIO server
    print("[SocketIO] Connecting to server...")
    if not connect_to_server():
        print("[SocketIO] ERROR: Could not connect to SocketIO server")
        return
    
    # Start serial reading thread
    print(f"[Serial] Attempting to connect to {SERIAL_PORT}...")
    serial_thread = threading.Thread(target=read_serial, daemon=True)
    serial_thread.start()
    
    # Wait a moment to see if serial connection is successful
    time.sleep(2)
    if not serial_connected:
        print(f"[Serial] WARNING: Could not connect to {SERIAL_PORT}")
        print("[Serial] Make sure your ESP32 is connected and the port is correct")
        print("[Serial] You can still use the web interface for manual annotation")
    
    # Start key listener thread
    print("[Key] Starting keyboard listener...")
    key_thread = threading.Thread(target=key_listener, daemon=True)
    key_thread.start()
    
    print("\n" + "=" * 50)
    print("SYSTEM READY")
    print("=" * 50)
    print("Controls:")
    print("  W - Label as Walking")
    print("  S - Label as Standing") 
    print("  F - Label as Freezing")
    print("  R - Start/Stop Recording")
    print("  Q - Quit")
    print("")
    print(f"Serial Status: {'Connected' if serial_connected else 'Disconnected'}")
    print(f"Server Status: {'Connected' if server_connected else 'Disconnected'}")
    print(f"Web Interface: {FLASK_SERVER_URL}")
    print("=" * 50)
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
            
            # Check connections periodically
            if not server_connected:
                print("[SocketIO] Connection lost, attempting to reconnect...")
                connect_to_server()
                
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
        cleanup_and_exit()

if __name__ == "__main__":
    main() 