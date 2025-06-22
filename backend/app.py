from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime
import threading
import time
import csv
import os
from fog_predictor import initialize_predictor, get_predictor

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

# Create data directory for CSV exports
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created data directory: {DATA_DIR}")

# Global variables for data streaming
streaming = False
current_session_id = None
current_state = 'standing'  # Track current state for labeling

# Initialize FOG predictor
print("🤖 Initializing FOG predictor...")
predictor_initialized = initialize_predictor()
if predictor_initialized:
    print("✅ FOG predictor ready for real-time monitoring!")
else:
    print("❌ FOG predictor failed to initialize")

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

@app.route('/')
def index():
    return jsonify({'message': 'FOG Detection Backend API', 'status': 'running'})

@app.route('/start_session', methods=['POST'])
def start_session():
    global streaming, current_session_id
    if not streaming:
        current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        streaming = True
        print(f"Started session: {current_session_id}")
        return jsonify({'status': 'started', 'session_id': current_session_id})
    return jsonify({'status': 'already_running', 'session_id': current_session_id})

@app.route('/stop_session', methods=['POST'])
def stop_session():
    global streaming, current_session_id
    if streaming and current_session_id:
        streaming = False
        session_to_return = current_session_id
        print(f"Session stopped: {current_session_id}")
        return jsonify({'status': 'stopped', 'session_id': session_to_return})
    else:
        return jsonify({'status': 'no_active_session'})

@app.route('/annotate_state', methods=['POST'])
def annotate_state():
    """Annotate current time window with activity state"""
    global current_state
    
    data = request.json
    state = data.get('state', 'standing')  # 'walking', 'standing', 'freezing'
    
    # Validate state
    if state not in ['walking', 'standing', 'freezing']:
        return jsonify({'error': 'Invalid state. Must be walking, standing, or freezing'}), 400
    
    # Update current state
    current_state = state
    
    # Update the last few records in database 
    conn = sqlite3.connect('fog_data.db')
    c = conn.cursor()
    
    # Mark last 2 seconds of data with the new state
    c.execute('''UPDATE imu_data 
                 SET label = ? 
                 WHERE session_id = ? 
                 AND id IN (SELECT id FROM imu_data 
                           WHERE session_id = ? 
                           ORDER BY id DESC LIMIT 100)''',
              (state, current_session_id, current_session_id))
    
    conn.commit()
    conn.close()
    
    print(f"State updated to: {state}")
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

@app.route('/save_session_csv/<session_id>', methods=['POST'])
def save_session_csv(session_id):
    """Save session data as CSV file in the backend data directory"""
    try:
        # Get session data from database
        conn = sqlite3.connect('fog_data.db')
        c = conn.cursor()
        c.execute('''SELECT timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, label 
                     FROM imu_data WHERE session_id = ? ORDER BY timestamp''', (session_id,))
        data = c.fetchall()
        conn.close()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No data found for session'}), 404
        
        # Create CSV filename
        csv_filename = f"session_{session_id}.csv"
        csv_filepath = os.path.join(DATA_DIR, csv_filename)
        
        # Write CSV file
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label'])
            # Write data
            writer.writerows(data)
        
        print(f"📁 Saved CSV: {csv_filepath} ({len(data)} samples)")
        
        return jsonify({
            'status': 'success', 
            'message': f'Session data saved as {csv_filename}',
            'filepath': csv_filepath,
            'samples': len(data)
        })
        
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict', methods=['GET'])
def get_prediction():
    """Get current FOG prediction"""
    predictor = get_predictor()
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        prediction = predictor.predict()
        buffer_status = predictor.get_buffer_status()
        
        return jsonify({
            'prediction': prediction,
            'buffer_status': buffer_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset_predictor', methods=['POST'])
def reset_predictor():
    """Reset the prediction buffer"""
    predictor = get_predictor()
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        predictor.reset_buffer()
        return jsonify({'status': 'success', 'message': 'Predictor buffer reset'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('🔌 Client connected to Flask backend WebSocket')
    emit('status', {'message': 'Connected to FOG Detection System'})

@socketio.on('disconnect')
def handle_disconnect():
    print('🔌 Client disconnected from Flask backend WebSocket')

@socketio.on('real_imu_data')
def handle_real_imu_data(data):
    """Handle raw IMU data from ESP32 connector (6 sensor values only)"""
    global current_session_id, streaming, current_state
    
    # Add data to predictor for real-time monitoring (always, regardless of recording)
    predictor = get_predictor()
    if predictor:
        try:
            predictor.add_data_point(data)
            
            # Get real-time prediction
            prediction_result = predictor.predict()
            
            # Add prediction to data being sent to frontend
            frontend_data = data.copy()
            frontend_data['current_state'] = current_state
            frontend_data['ai_prediction'] = prediction_result
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            frontend_data = data.copy()
            frontend_data['current_state'] = current_state
            frontend_data['ai_prediction'] = None
    else:
        frontend_data = data.copy()
        frontend_data['current_state'] = current_state
        frontend_data['ai_prediction'] = None
    
    # Store in database only if recording session is active
    if current_session_id and streaming:
        try:
            # Store RAW IMU data in database with current user-annotated state
            store_imu_data(data, label=current_state)
            print(f"📤 Stored & forwarded: acc_x={data.get('acc_x')}, state={current_state}")
            
        except Exception as e:
            print(f"❌ Error handling real IMU data: {e}")
    # Always log received data but don't store if no session
    elif not streaming:
        print(f"⚠️ ESP32 data received but no recording session active")
    
    # Forward to all connected clients (including the web frontend) with prediction
    emit('imu_data', frontend_data, broadcast=True)

@socketio.on('esp32_status')
def handle_esp32_status(data):
    """Handle ESP32 connector status updates"""
    print(f"ESP32 Connector: {data}")
    emit('esp32_status', data, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)