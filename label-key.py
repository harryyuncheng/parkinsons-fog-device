import serial
import threading
import time
import keyboard  # Admin access required for non-Windows
import csv
from collections import deque

# === CONFIG ===
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust to your ESP32 port
BAUD_RATE = 115200
LABEL_KEYS = {
    'w': 'walking',
    's': 'standing',
    'f': 'freezing'
}
BUFFER_DURATION = 5  # seconds of rolling data
LOG_FILE = 'imu_log.csv'

# === Data Structures ===
data_buffer = deque()  # holds (timestamp, ax, ay, az, gx, gy, gz)
label_intervals = deque()  # holds (start_ts, end_ts, label)

# === Thread: Serial Reading ===
def read_serial():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("[Serial] Connected.")
    while True:
        line = ser.readline().decode('utf-8').strip()
        try:
            # Assume format: ax,ay,az,gx,gy,gz
            ax, ay, az, gx, gy, gz = map(float, line.split(','))
            timestamp = time.time()
            data_buffer.append((timestamp, ax, ay, az, gx, gy, gz))

            # Trim old data
            while data_buffer and timestamp - data_buffer[0][0] > BUFFER_DURATION:
                data_buffer.popleft()

        except Exception as e:
            print(f"[Serial] Error parsing line: {line} | {e}")

# === Thread: Key Labeling ===
def key_listener():
    print("[Key] Press W (walk), S (stand), F (freeze)")
    key_states = {key: False for key in LABEL_KEYS}
    key_start_times = {key: None for key in LABEL_KEYS}

    while True:
        for key, label in LABEL_KEYS.items():
            if keyboard.is_pressed(key):
                if not key_states[key]:
                    # Key just pressed
                    key_states[key] = True
                    key_start_times[key] = time.time()
                    print(f"[Key] {label.upper()} START at {key_start_times[key]:.2f}")
            else:
                if key_states[key]:
                    # Key just released
                    key_states[key] = False
                    start_ts = key_start_times[key]
                    end_ts = time.time()
                    label_intervals.append((start_ts, end_ts, label))
                    print(f"[Key] {label.upper()} END at {end_ts:.2f}")
        time.sleep(0.01)

# === Logger: Match labels to data ===
def logger():
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label'])

        while True:
            time.sleep(0.5)
            while data_buffer:
                datapoint = data_buffer.popleft()
                ts, *imu_data = datapoint

                label = None
                for start_ts, end_ts, event_label in list(label_intervals):
                    # Label if data point is in the interval, but not in the last 0.2s before release
                    if start_ts <= ts < end_ts - 0.2:
                        label = event_label
                        break

                writer.writerow([ts] + imu_data + [label])
                f.flush()

# === Run Threads ===
if __name__ == "__main__":
    threading.Thread(target=read_serial, daemon=True).start()
    threading.Thread(target=key_listener, daemon=True).start()
    logger()  # run on main thread
