from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sqlite3
import json
import numpy as np
from datetime import datetime
import threading
import time
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fog_detection_secret_key'
CORS(app)  # Enable CORS for Next.js frontend
socketio = SocketIO(app, cors_allowed_origins="*")

# Database setup
def init_db():
    conn = sqlite3.connect('fog_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS imu_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  acc_x REAL, acc_y REAL, acc_z REAL,
                  gyro_x REAL, gyro_y REAL, gyro_z REAL,
                  label TEXT,
                  session_id TEXT)''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Global variables for data streaming
streaming = False
current_session_id = None
use_real_data = False  # Flag to use real ESP32 data instead of simulated

class IMUDataGenerator:
    """Simulates real-time IMU data for demonstration"""
    def __init__(self):
        self.time_step = 0
        self.current_state = 'standing'  # 'walking', 'standing', 'freezing'
        
    def generate_sample(self):
        """Generate a single IMU sample"""
        if self.current_state == 'freezing':
            # Simulate FOG: reduced movement, more irregular patterns
            acc_x = np.random.normal(0, 0.5) + 0.1 * np.sin(self.time_step * 0.1)
            acc_y = np.random.normal(0, 0.5) + 0.1 * np.cos(self.time_step * 0.1)
            acc_z = np.random.normal(9.8, 0.5)
            gyro_x = np.random.normal(0, 0.2)
            gyro_y = np.random.normal(0, 0.2)
            gyro_z = np.random.normal(0, 0.2)
        elif self.current_state == 'walking':
            # Simulate walking: higher acceleration and gyro activity
            acc_x = np.random.normal(0, 1.5) + 2 * np.sin(self.time_step * 0.3)
            acc_y = np.random.normal(0, 1.5) + 1.5 * np.cos(self.time_step * 0.3)
            acc_z = np.random.normal(9.8, 1.0) + 0.5 * np.sin(self.time_step * 0.2)
            gyro_x = np.random.normal(0, 0.8) + 0.5 * np.sin(self.time_step * 0.25)
            gyro_y = np.random.normal(0, 0.8) + 0.5 * np.cos(self.time_step * 0.25)
            gyro_z = np.random.normal(0, 0.8)
        else:  # standing
            # Simulate standing: low activity, mostly gravity
            acc_x = np.random.normal(0, 0.3)
            acc_y = np.random.normal(0, 0.3)
            acc_z = np.random.normal(9.8, 0.3)
            gyro_x = np.random.normal(0, 0.1)
            gyro_y = np.random.normal(0, 0.1)
            gyro_z = np.random.normal(0, 0.1)
        
        self.time_step += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'acc_x': round(acc_x, 3),
            'acc_y': round(acc_y, 3),
            'acc_z': round(acc_z, 3),
            'gyro_x': round(gyro_x, 3),
            'gyro_y': round(gyro_y, 3),
            'gyro_z': round(gyro_z, 3)
        }

imu_generator = IMUDataGenerator()

def store_imu_data(data, label='standing'):
    """Store IMU data in database"""
    conn = sqlite3.connect('fog_data.db')
    c = conn.cursor()
    c.execute('''INSERT INTO imu_data 
                 (timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, label, session_id)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (data['timestamp'], data['acc_x'], data['acc_y'], data['acc_z'],
               data['gyro_x'], data['gyro_y'], data['gyro_z'], label, current_session_id))
    conn.commit()
    conn.close()

def data_streaming_thread():
    """Background thread for streaming IMU data (only used for simulated data)"""
    global streaming, use_real_data
    while streaming:
        try:
            # Only generate simulated data if not using real ESP32 data
            if not use_real_data:
                # Generate IMU sample
                imu_sample = imu_generator.generate_sample()
                
                # Store in database with current state
                store_imu_data(imu_sample, label=imu_generator.current_state)
                
                # Add current state to sample for frontend
                imu_sample['current_state'] = imu_generator.current_state
                
                # Emit to frontend
                socketio.emit('imu_data', imu_sample)
                
                # Wait for next sample (50Hz = 20ms)
                time.sleep(0.02)
            else:
                # When using real data, just sleep and let the ESP32 connector handle data
                time.sleep(0.1)
        except Exception as e:
            print(f"Error in data streaming: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return jsonify({'message': 'FOG Detection Backend API', 'status': 'running'})

@app.route('/start_session', methods=['POST'])
def start_session():
    global streaming, current_session_id
    if not streaming:
        current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        streaming = True
        thread = threading.Thread(target=data_streaming_thread)
        thread.daemon = True
        thread.start()
        return jsonify({'status': 'started', 'session_id': current_session_id})
    return jsonify({'status': 'already_running', 'session_id': current_session_id})

@app.route('/stop_session', methods=['POST'])
def stop_session():
    global streaming
    streaming = False
    return jsonify({'status': 'stopped'})

@app.route('/annotate_state', methods=['POST'])
def annotate_state():
    """Annotate current time window with activity state"""
    data = request.json
    state = data.get('state', 'standing')  # 'walking', 'standing', 'freezing'
    
    # Validate state
    if state not in ['walking', 'standing', 'freezing']:
        return jsonify({'error': 'Invalid state. Must be walking, standing, or freezing'}), 400
    
    # Update the last few records in database 
    conn = sqlite3.connect('fog_data.db')
    c = conn.cursor()
    
    # Mark last 2 seconds of data (100 samples at 50Hz) with the new state
    c.execute('''UPDATE imu_data 
                 SET label = ? 
                 WHERE session_id = ? 
                 AND id IN (SELECT id FROM imu_data 
                           WHERE session_id = ? 
                           ORDER BY id DESC LIMIT 100)''',
              (state, current_session_id, current_session_id))
    
    conn.commit()
    conn.close()
    
    # Update simulation
    imu_generator.current_state = state
    
    socketio.emit('state_annotation', {'state': state, 'timestamp': datetime.now().isoformat()})
    
    return jsonify({'status': 'annotated', 'state': state})

@app.route('/get_session_data/<session_id>')
def get_session_data(session_id):
    """Get all data for a specific session"""
    conn = sqlite3.connect('fog_data.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM imu_data WHERE session_id = ? ORDER BY timestamp''', (session_id,))
    data = c.fetchall()
    conn.close()
    
    # Convert to list of dictionaries
    columns = ['id', 'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label', 'session_id']
    result = [dict(zip(columns, row)) for row in data]
    
    return jsonify(result)

@app.route('/get_sessions')
def get_sessions():
    """Get list of all recording sessions"""
    conn = sqlite3.connect('fog_data.db')
    c = conn.cursor()
    c.execute('''SELECT session_id, COUNT(*) as sample_count, 
                        MIN(timestamp) as start_time, 
                        MAX(timestamp) as end_time,
                        SUM(CASE WHEN label = 'walking' THEN 1 ELSE 0 END) as walking_count,
                        SUM(CASE WHEN label = 'standing' THEN 1 ELSE 0 END) as standing_count,
                        SUM(CASE WHEN label = 'freezing' THEN 1 ELSE 0 END) as freezing_count
                 FROM imu_data 
                 GROUP BY session_id 
                 ORDER BY start_time DESC''')
    sessions = c.fetchall()
    conn.close()
    
    columns = ['session_id', 'sample_count', 'start_time', 'end_time', 'walking_count', 'standing_count', 'freezing_count']
    result = [dict(zip(columns, row)) for row in sessions]
    
    return jsonify(result)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to FOG Detection System'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('real_imu_data')
def handle_real_imu_data(data):
    """Handle real IMU data from ESP32 connector"""
    global current_session_id, use_real_data
    
    if current_session_id and streaming:
        try:
            # Store in database with current state
            # The ESP32 connector doesn't know the current state, so we use the generator's state
            store_imu_data(data, label=imu_generator.current_state)
            
            # Add current state to sample for frontend
            data['current_state'] = imu_generator.current_state
            
            # Forward to all connected clients (including the web frontend)
            emit('imu_data', data, broadcast=True)
            
            # Mark that we're using real data
            use_real_data = True
            
        except Exception as e:
            print(f"Error handling real IMU data: {e}")

@socketio.on('esp32_status')
def handle_esp32_status(data):
    """Handle ESP32 connector status updates"""
    print(f"ESP32 Connector: {data}")
    emit('esp32_status', data, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)