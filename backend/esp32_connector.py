import serial
import threading
import time
import requests
import socketio
from collections import deque
import json

# === CONFIG ===
SERIAL_PORT = '/dev/cu.usbserial-0001'  # Adjust to your ESP32 port
BAUD_RATE = 115200
FLASK_SERVER_URL = 'http://localhost:8080'
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
    print(f"[Serial] Attempting to open serial port: {SERIAL_PORT} at {BAUD_RATE} baud")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        serial_connected = True
        print(f"[Serial] Connected to {SERIAL_PORT}")
        
        while True:
            if not serial_connected:
                print("[Serial] serial_connected flag is False, breaking loop.")
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
                        
                        # Send to Flask server if connected
                        if server_connected:
                            send_imu_data_to_server(ax, ay, az, gx, gy, gz, timestamp)
                
            except UnicodeDecodeError as e:
                print(f"[Serial] UnicodeDecodeError: {e}")
                continue
            except ValueError as e:
                print(f"[Serial] Error parsing line: {line} | {e}")
            except Exception as e:
                print(f"[Serial] Unexpected error in read loop: {e}")
                
    except serial.SerialException as e:
        print(f"[Serial] Failed to connect to {SERIAL_PORT}: {e}")
        serial_connected = False
    except Exception as e:
        print(f"[Serial] Unexpected error during serial connection: {e}")
        serial_connected = False

def send_imu_data_to_server(ax, ay, az, gx, gy, gz, timestamp):
    """Send RAW IMU data to Flask server via SocketIO (no state, just sensor values)"""
    try:
        from datetime import datetime
        imu_data = {
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'acc_x': round(ax, 3),
            'acc_y': round(ay, 3),
            'acc_z': round(az, 3),
            'gyro_x': round(gx, 3),
            'gyro_y': round(gy, 3),
            'gyro_z': round(gz, 3)
        }
        sio.emit('real_imu_data', imu_data)
        print(f"[Data] Sent: ax={ax:.2f}, ay={ay:.2f}, az={az:.2f}, gx={gx:.2f}, gy={gy:.2f}, gz={gz:.2f}")
    except Exception as e:
        print(f"[SocketIO] Error sending IMU data: {e}")

def connect_to_server():
    """Connect to Flask SocketIO server"""
    global server_connected
    print(f"[SocketIO] Attempting to connect to {FLASK_SERVER_URL} ...")
    try:
        sio.connect(FLASK_SERVER_URL)
        server_connected = True
        print("[SocketIO] Successfully connected to server.")
        return True
    except Exception as e:
        print(f"[SocketIO] Failed to connect to server: {e}")
        server_connected = False
        return False

def cleanup_and_exit():
    global serial_connected, server_connected
    print("[Cleanup] Shutting down...")
    if current_session_active:
        try:
            requests.post(f'{FLASK_SERVER_URL}/stop_session', timeout=2)
        except:
            pass
    if server_connected:
        sio.disconnect()
    serial_connected = False
    print("[Cleanup] Shutdown complete")
    exit(0)

def check_server_status():
    print(f"[Server] Checking Flask server at {FLASK_SERVER_URL}/ ...")
    try:
        response = requests.get(f'{FLASK_SERVER_URL}/', timeout=5)
        print(f"[Server] Server responded with status code: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"[Server] Exception while checking server status: {e}")
        return False

def main():
    print("=" * 50)
    print("ESP32 IMU Connector for FOG Detection System (NO KEYBOARD)")
    print("=" * 50)
    print("[Server] Checking Flask server connection...")
    if not check_server_status():
        print(f"[Server] ERROR: Flask server not running at {FLASK_SERVER_URL}")
        print("[Server] Please start the Flask server first: python app.py")
        return
    print("[Server] Flask server is running")
    print("[SocketIO] Connecting to server...")
    if not connect_to_server():
        print("[SocketIO] ERROR: Could not connect to SocketIO server")
        return
    print(f"[Serial] Attempting to connect to {SERIAL_PORT}...")
    serial_thread = threading.Thread(target=read_serial, daemon=True)
    serial_thread.start()
    time.sleep(2)
    if not serial_connected:
        print(f"[Serial] WARNING: Could not connect to {SERIAL_PORT}")
        print("[Serial] Make sure your ESP32 is connected and the port is correct")
        print("[Serial] You can still use the web interface for manual annotation")
    print("\n" + "=" * 50)
    print("SYSTEM READY (NO KEYBOARD)")
    print("=" * 50)
    print(f"Serial Status: {'Connected' if serial_connected else 'Disconnected'}")
    print(f"Server Status: {'Connected' if server_connected else 'Disconnected'}")
    print(f"Web Interface: {FLASK_SERVER_URL}")
    print("=" * 50)
    try:
        while True:
            time.sleep(1)
            if not server_connected:
                print("[SocketIO] Connection lost, attempting to reconnect...")
                connect_to_server()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
        cleanup_and_exit()

if __name__ == "__main__":
    main()