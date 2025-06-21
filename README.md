# FOG Detection System

A comprehensive real-time system for collecting and annotating IMU data for Freezing of Gait (FOG) detection in Parkinson's patients.

## System Architecture

```
ESP32 → Serial → Backend (ESP32 Connector) → Flask API → Next.js Frontend
                       ↓                        ↓
                 Terminal Keys              Real-time WebSocket
                       ↓                        ↓
                 HTTP API ← ← ← ← ← ← ← ← Database (SQLite)
```

## Components

### 🔧 Backend (`/backend/`)

- **Flask API** (`app.py`) - REST endpoints and WebSocket server
- **ESP32 Connector** (`esp32_connector.py`) - Serial bridge for real ESP32 data
- **SQLite Database** - Stores all IMU data with 3-way classification labels

### 🖥️ Frontend (`/frontend/`)

- **Next.js App** - Modern React-based web interface
- **Real-time Charts** - Live visualization of IMU data (accelerometer & gyroscope)
- **State Annotation** - 3-way classification: Walking / Standing / Freezing

## Quick Start

### 1. Backend Setup

```bash
# Install backend dependencies
cd backend
pip install -r backend_requirements.txt

# Start Flask server
python app.py
```

### 2. Frontend Setup

```bash
# Install frontend dependencies
cd frontend
npm install

# Start Next.js development server
npm run dev
```

### 3. Access the System

- **Web Interface**: http://localhost:3000
- **Backend API**: http://localhost:6000

## Usage Modes

### 🧪 Simulated Data (Testing)

1. Start both backend and frontend
2. Open web interface
3. Click "Start Recording"
4. Use keyboard shortcuts to annotate states:
   - `W` - Walking
   - `S` - Standing
   - `F` - Freezing

### 🔌 Real ESP32 Data

1. Connect ESP32 via USB
2. Update serial port in `backend/esp32_connector.py`
3. Start backend and frontend
4. Start ESP32 connector:
   ```bash
   cd backend
   python esp32_connector.py
   ```
5. Real data will appear automatically in web interface

## ESP32 Data Format

Your ESP32 should send CSV data over serial:

```
ax,ay,az,gx,gy,gz
1.23,-0.45,9.67,12.34,-5.67,8.90
```

Where:

- `ax, ay, az` = Accelerometer (m/s²)
- `gx, gy, gz` = Gyroscope (°/s)

## Features

### 📊 Real-time Visualization

- Live accelerometer and gyroscope charts with X,Y,Z legends
- Color-coded state indicators
- Sample counters for each state

### 🏷️ Data Annotation

- **3-way classification**: Walking vs Standing vs Freezing
- **Keyboard shortcuts**: W/S/F keys
- **Real-time feedback** with immediate visual updates

### 💾 Data Management

- **Session recording** with unique IDs
- **SQLite storage** with timestamps and labels
- **CSV export** for machine learning model training
- **Session history** with statistics

### 🔄 Multiple Input Sources

- **Simulated data** for testing and development
- **Real ESP32 data** via serial connection
- **Manual annotation** via web interface or ESP32 connector

## API Endpoints

- `GET /` - Health check
- `POST /start_session` - Start recording session
- `POST /stop_session` - Stop recording session
- `POST /annotate_state` - Annotate current state (`{'state': 'walking'|'standing'|'freezing'}`)
- `GET /get_sessions` - Get all recording sessions
- `GET /get_session_data/<session_id>` - Get data for specific session

## WebSocket Events

- `imu_data` - Real-time IMU data stream
- `state_annotation` - State annotation updates
- `esp32_status` - ESP32 connector status

## Development

### File Structure

```
parkinsons/
├── backend/
│   ├── app.py                    # Flask API server
│   ├── esp32_connector.py        # ESP32 serial bridge
│   ├── backend_requirements.txt  # Python dependencies
│   └── esp32_requirements.txt    # ESP32 connector dependencies
├── frontend/
│   ├── app/
│   │   └── page.tsx             # Main Next.js page
│   ├── lib/
│   │   └── api.ts               # API service layer
│   └── package.json             # Node.js dependencies
└── README.md                    # This file
```

### Environment Variables

Create `.env.local` in `/frontend/`:

```
NEXT_PUBLIC_BACKEND_URL=http://localhost:5000
```

### Database Schema

```sql
CREATE TABLE imu_data (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    acc_x REAL, acc_y REAL, acc_z REAL,
    gyro_x REAL, gyro_y REAL, gyro_z REAL,
    label TEXT,  -- 'walking', 'standing', 'freezing'
    session_id TEXT
);
```

## Troubleshooting

### Backend Issues

- **Port 6000 in use**: Change port in `app.py`
- **Database errors**: Delete `fog_data.db` to reset
- **CORS errors**: Ensure Flask-CORS is installed

### ESP32 Issues

- **Serial connection failed**: Check port in `esp32_connector.py`
- **Data format errors**: Ensure ESP32 sends correct CSV format
- **Permission denied**: Run with admin/sudo privileges

### Frontend Issues

- **API connection failed**: Check backend is running on port 5000
- **WebSocket disconnects**: Check firewall settings
- **Build errors**: Run `npm install` to update dependencies

## Next Steps

1. **Model Integration**: Add trained FOG detection models for real-time prediction
2. **Data Analytics**: Implement statistical analysis and reporting
3. **Cloud Storage**: Add cloud backup and sync capabilities
4. **Mobile App**: Create companion mobile application
5. **Multi-patient**: Add patient management and multi-session support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both simulated and real data
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
