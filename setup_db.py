import sqlite3

def initialize_database():
    conn = sqlite3.connect('hospital.db')
    c = conn.cursor()

    # 1. Base Authentication Table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT,
                    role TEXT)''')

    # 2. Complete Clinician Registry
    c.execute('''CREATE TABLE IF NOT EXISTS doctors (
                    user_id INTEGER,
                    name TEXT,
                    specialty TEXT,
                    qualifications TEXT,
                    experience_years INTEGER,
                    rating REAL,
                    status TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id))''')

    # 3. Master Appointment Desk
    c.execute('''CREATE TABLE IF NOT EXISTS appointments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_username TEXT,
                    doctor_name TEXT,
                    date TEXT, 
                    time_slot TEXT, 
                    status TEXT)''')

    # 4. Deep Diagnostic Medical Records
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, 
                    patient_id TEXT, 
                    name TEXT, 
                    age INTEGER, 
                    sex TEXT, 
                    spo2 INTEGER, 
                    temp REAL, 
                    diagnosis TEXT, 
                    confidence REAL, 
                    ai_notes TEXT)''')

    # --- BASE VALUE INJECTIONS ---
    c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES ('admin', '1234', 'admin')")
    c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES ('PT-7265', '1234', 'patient')")

    # The 8 Specified Pulmonologists
    doctors_data = [
        ("dr.krithi", "Dr. Krithi U Kotian", "Chief Pulmonologist", "MBBS, MD (Pulmonary Med), FCCP", 15, 4.9, "On Call"),
        ("dr.devika", "Dr. Devika P Shetty", "Senior Radiologist", "MBBS, MD (Radiodiagnosis)", 12, 4.8, "On Duty"),
        ("dr.shaktha", "Dr. Shaktha", "Internal Medicine Specialist", "MBBS, MD (Internal Medicine)", 10, 4.7, "On Duty"),
        ("dr.shrejal", "Dr. Shrejal B Shetty", "Cardiothoracic Surgeon", "MBBS, MS, MCh", 14, 4.9, "On Call"),
        ("dr.sanjana", "Dr. Sanjana Tej", "Respiratory Therapist", "BSc, MSc (Respiratory Care)", 8, 4.6, "Off Duty"),
        ("dr.lawrence", "Dr. Lawrence D'souza", "Infectious Disease Specialist", "MBBS, MD, FNB", 18, 5.0, "On Duty"),
        ("dr.pavithra", "Dr. Pavithra Acharya", "Pulmonologist", "MBBS, DNB (Respiratory Diseases)", 9, 4.7, "On Leave"),
        ("dr.garette", "Dr. Garette Graham", "Critical Care Intensivist", "MD, EDIC", 11, 4.8, "On Duty")
    ]

    for doc in doctors_data:
        c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, '1234', 'doctor')", (doc[0],))
        c.execute("SELECT id FROM users WHERE username=?", (doc[0],))
        user_id = c.fetchone()[0]
        c.execute("INSERT INTO doctors VALUES (?, ?, ?, ?, ?, ?, ?)", (user_id, doc[1], doc[2], doc[3], doc[4], doc[5], doc[6]))

    conn.commit()
    conn.close()
    print("✅ System SQLite Database Built & Initialized!")

if __name__ == '__main__':
    initialize_database()