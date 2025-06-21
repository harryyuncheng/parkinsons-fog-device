import sqlite3
import csv
import sys
from datetime import datetime

DB_FILE = 'fog_data.db'
CSV_FILE = f'imu_data_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

def export_to_csv(db_file=DB_FILE, csv_file=CSV_FILE):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''SELECT timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, label, session_id FROM imu_data ORDER BY timestamp''')
    rows = c.fetchall()
    conn.close()

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label', 'session_id'])
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {csv_file}")

if __name__ == '__main__':
    db_file = sys.argv[1] if len(sys.argv) > 1 else DB_FILE
    csv_file = sys.argv[2] if len(sys.argv) > 2 else CSV_FILE
    export_to_csv(db_file, csv_file)
