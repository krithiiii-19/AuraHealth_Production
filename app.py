import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, send_file
import sqlite3
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import json

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

app = Flask(__name__)
app.secret_key = "AURA_HEALTH_ENTERPRISE_KEY"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("Warming up DenseNet121 Neural Engine...")
base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
model = Model(inputs=base_model.input, outputs=Dense(1, activation='sigmoid')(Dropout(0.5)(x)))
grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[base_model.output, model.output])

try:
    model.load_weights('best_densenet_model.keras')
    model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
    grad_model(np.zeros((1, 224, 224, 3), dtype=np.float32))
    print("AI Engine & Thermal Mapper Online and Ready.")
except Exception as e:
    print(f"Warning: Could not load AI weights. Ensure 'best_densenet_model.keras' is in the root folder. Error: {e}")

def get_gradcam_heatmap(img_array):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(tf.maximum(heatmap, 0))
    max_val = tf.math.reduce_max(heatmap)
    return heatmap.numpy() if max_val == 0 else (heatmap / max_val).numpy()

def add_watermark(p):
    p.set_font("Helvetica", "B", 60)
    p.set_text_color(220, 220, 220)
    p.rotate(45, 100, 150)
    p.text(30, 190, "AURA HEALTH")
    p.rotate(0)
    p.set_text_color(0, 0, 0)

def get_db_connection():
    conn = sqlite3.connect('hospital.db')
    conn.row_factory = sqlite3.Row
    try: conn.execute('ALTER TABLE patients ADD COLUMN doctor_name TEXT DEFAULT "General Admission"')
    except: pass
    try: conn.execute('ALTER TABLE appointments ADD COLUMN xray_path TEXT DEFAULT ""')
    except: pass
    try: conn.execute('ALTER TABLE appointments ADD COLUMN scan_status TEXT DEFAULT "PENDING"')
    except: pass
    try: conn.execute('ALTER TABLE patients ADD COLUMN orig_img TEXT')
    except: pass
    try: conn.execute('ALTER TABLE patients ADD COLUMN heat_img TEXT')
    except: pass
    try: conn.execute('ALTER TABLE patients ADD COLUMN pdf_link TEXT')
    except: pass
    
    conn.execute('CREATE TABLE IF NOT EXISTS doctor_leaves (doctor_name TEXT, leave_date TEXT)')
    try: conn.execute('ALTER TABLE doctor_leaves ADD COLUMN shift TEXT DEFAULT "Full Day"')
    except: pass
    conn.commit()
    return conn

@app.route('/')
def landing_page(): 
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (request.form['username'], request.form['password'])).fetchone()
        conn.close()
        if user:
            session.update({'logged_in': True, 'username': user['username'], 'role': user['role']})
            if user['role'] == 'admin': return redirect(url_for('admin_dashboard'))
            if user['role'] == 'doctor': return redirect(url_for('doctor_dashboard'))
            return redirect(url_for('patient_dashboard'))
        flash("Invalid credentials.", "error")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        if conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone():
            flash("Username/MRN already taken. Please choose another.", "error")
        else:
            conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'patient')", (username, password))
            conn.commit()
            flash("Patient Registration Successful. Please log in.", "success")
            conn.close()
            return redirect(url_for('login'))
        conn.close()
    return render_template('register.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin_dashboard():
    if session.get('role') != 'admin': return redirect(url_for('login'))
    conn = get_db_connection()
    if request.method == 'POST':
        if 'delete_doc' in request.form:
            doc_id = request.form['doc_id']
            conn.execute('DELETE FROM doctors WHERE user_id = ?', (doc_id,))
            conn.execute('DELETE FROM users WHERE id = ?', (doc_id,))
            flash("Physician credentials revoked successfully.", "success")
        elif 'add_doc' in request.form:
            username, name, spec = request.form['username'], request.form['name'], request.form['specialty']
            if conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone():
                flash("Doctor Username already exists.", "error")
            else:
                conn.execute("INSERT INTO users (username, password, role) VALUES (?, '1234', 'doctor')", (username,))
                user_id = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()['id']
                conn.execute("INSERT INTO doctors (user_id, name, specialty, qualifications, experience_years, rating, status) VALUES (?, ?, ?, 'MD/MBBS', 5, 5.0, 'On Duty')", (user_id, name, spec))
                flash(f"Dr. {name} successfully added.", "success")
        elif 'change_status' in request.form:
            doc_id = request.form['doc_id']
            new_status = request.form['new_status']
            conn.execute('UPDATE doctors SET status = ? WHERE user_id = ?', (new_status, doc_id))
            flash(f"Physician network status updated to '{new_status}'.", "success")
        elif 'assign_patient' in request.form:
            pat_mrn = request.form['patient_mrn']
            doc_name = request.form['doctor_name']
            apt_date = request.form['apt_date']
            apt_time = request.form['apt_time']
            pat = conn.execute('SELECT * FROM users WHERE username=? AND role="patient"', (pat_mrn,)).fetchone()
            if not pat:
                flash(f"Error: Patient MRN '{pat_mrn}' is not registered in the network.", "error")
            else:
                conn.execute('INSERT INTO appointments (patient_username, doctor_name, date, time_slot, status, scan_status, xray_path) VALUES (?,?,?,?,?,?,?)',
                             (pat_mrn, doc_name, apt_date, apt_time, 'CONFIRMED', 'PENDING', ""))
                flash(f"Success: Patient {pat_mrn} officially routed to {doc_name}.", "success")
        conn.commit()

    doctors = conn.execute('SELECT * FROM doctors').fetchall()
    patients_count = len(conn.execute('SELECT DISTINCT patient_id FROM patients').fetchall())
    all_leaves = conn.execute('SELECT * FROM doctor_leaves ORDER BY leave_date DESC').fetchall()
    today_str = datetime.now().strftime('%Y-%m-%d')
    conn.close()
    return render_template('admin_dash.html', doctors=doctors, patient_count=patients_count, leaves=all_leaves, today=today_str)

@app.route('/patient', methods=['GET', 'POST'])
def patient_dashboard():
    if session.get('role') != 'patient': return redirect(url_for('login'))
    conn = get_db_connection()
    
    if request.method == 'POST' and 'book_doc' in request.form:
        doc_name, date, slot = request.form['book_doc'], request.form['date'], request.form['slot']
        file = request.files.get('patient_xray')
        xray_path = ""
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            xray_path = f"uploads/{session['username']}_{filename}"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], f"{session['username']}_{filename}"))
        
        if conn.execute('SELECT * FROM appointments WHERE doctor_name=? AND date=? AND time_slot=?', (doc_name, date, slot)).fetchone():
            flash("Error: Slot already taken.", "error")
        else:
            conn.execute('INSERT INTO appointments (patient_username, doctor_name, date, time_slot, status, xray_path, scan_status) VALUES (?,?,?,?,?,?,?)',
                         (session['username'], doc_name, date, slot, 'CONFIRMED', xray_path, 'PENDING'))
            conn.commit()
            flash("Appointment locked and X-Ray securely uploaded to hospital servers.", "success")
    
    doctors = conn.execute('SELECT * FROM doctors ORDER BY rating DESC').fetchall()
    records = conn.execute('SELECT * FROM patients WHERE patient_id = ? ORDER BY timestamp DESC', (session['username'],)).fetchall()
    
    # FIX: Ensure Patient Dashboard also filters out old, past appointments
    today_str = datetime.now().strftime('%Y-%m-%d')
    apts = conn.execute('SELECT * FROM appointments WHERE patient_username = ? AND date >= ? ORDER BY date ASC, time_slot ASC', (session['username'], today_str)).fetchall()
    
    all_apts = conn.execute('SELECT doctor_name, date, time_slot FROM appointments').fetchall()
    booked_data = [{"doctor": r["doctor_name"], "date": r["date"], "slot": r["time_slot"]} for r in all_apts]
    
    all_leaves = conn.execute('SELECT doctor_name, leave_date, shift FROM doctor_leaves').fetchall()
    leave_data = [{"doctor": r["doctor_name"], "date": r["leave_date"], "shift": r["shift"]} for r in all_leaves]
    
    conn.close()
    return render_template('patient_dash.html', doctors=doctors, records=records, appointments=apts, booked_data=booked_data, leave_data=leave_data)

@app.route('/doctor', methods=['GET', 'POST'])
def doctor_dashboard():
    if session.get('role') != 'doctor': return redirect(url_for('login'))
    conn = get_db_connection()
    doc_info = conn.execute('SELECT * FROM doctors JOIN users ON doctors.user_id = users.id WHERE users.username = ?', (session['username'],)).fetchone()
    result_data = None
    
    if request.method == 'POST':
        if 'accept_apt' in request.form:
            conn.execute('UPDATE appointments SET status="CONFIRMED" WHERE id=?', (request.form['apt_id'],))
            conn.commit()
            flash("Appointment Confirmed.", "success")
            
        elif 'mark_leave' in request.form:
            shift_type = request.form.get('leave_shift', 'Full Day')
            conn.execute("INSERT INTO doctor_leaves (doctor_name, leave_date, shift) VALUES (?, ?, ?)", (doc_info['name'], request.form['leave_date'], shift_type))
            conn.commit()
            flash(f"System updated: You are marked as Unavailable on {request.form['leave_date']} ({shift_type}).", "success")
            
        elif 'delete_record' in request.form:
            record_ts = request.form.get('record_timestamp')
            del_pat_id = request.form.get('del_patient_id')
            conn.execute("DELETE FROM patients WHERE timestamp = ? AND patient_id = ? AND doctor_name = ?", (record_ts, del_pat_id, doc_info['name']))
            conn.commit()
            flash(f"EMR System: Patient record for {del_pat_id} securely purged.", "success")
            
        elif 'schedule_followup' in request.form:
            patient_mrn = request.form.get('patient_mrn')
            f_date = request.form.get('followup_date')
            f_time = request.form.get('followup_time')
            if conn.execute("SELECT * FROM users WHERE username = ? AND role = 'patient'", (patient_mrn,)).fetchone():
                conn.execute("INSERT INTO appointments (patient_username, doctor_name, date, time_slot, status) VALUES (?, ?, ?, ?, 'CONFIRMED')",
                             (patient_mrn, doc_info['name'], f_date, f_time))
                conn.commit()
                flash(f"Follow-up successfully locked for {patient_mrn}.", "success")
            else:
                flash("Error: Patient MRN not found.", "error")
                
        elif 'generate_bill' in request.form:
            pat_id = request.form.get('bill_pat_id')
            pat_record = conn.execute("SELECT * FROM patients WHERE patient_id = ? AND doctor_name = ? ORDER BY timestamp DESC", (pat_id, doc_info['name'])).fetchone()
            
            if not pat_record:
                flash("Error: No clinical record found to generate a bill.", "error")
            else:
                pdf = FPDF()
                pdf.add_page()
                
                pdf.set_fill_color(15, 23, 42)
                pdf.rect(15, 10, 14, 14, 'F')
                pdf.set_fill_color(100, 255, 218)
                pdf.rect(21, 12.5, 2, 9, 'F')
                pdf.rect(17.5, 16, 9, 2, 'F')
                
                pdf.set_xy(34, 11)
                pdf.set_font("Helvetica", style="B", size=24)
                pdf.set_text_color(15, 23, 42)
                pdf.cell(0, 8, "AURA HEALTH MEDICAL CENTER", align="L", ln=1)
                
                pdf.set_xy(34, 20)
                pdf.set_font("Helvetica", style="I", size=11)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 5, "Official Medical Invoice & Billing Statement", align="L", ln=1)
                
                pdf.set_text_color(0, 0, 0)
                pdf.line(15, 30, 195, 30)
                pdf.set_y(38)
                
                pdf.set_font("Helvetica", style="B", size=12)
                pdf.cell(100, 6, f"Patient Name: {pat_record['name']}", ln=0)
                pdf.cell(90, 6, f"Invoice Date: {datetime.now().strftime('%Y-%m-%d')}", ln=1)
                
                pdf.set_font("Helvetica", size=10)
                pdf.cell(100, 6, f"Patient MRN: {pat_record['patient_id']}", ln=0)
                pdf.cell(90, 6, f"Attending Physician: Dr. {doc_info['name']}", ln=1)
                
                pdf.ln(12)
                
                pdf.set_font("Helvetica", style="B", size=11)
                pdf.set_fill_color(240, 240, 240)
                pdf.cell(110, 10, "  Description of Services", border=1, fill=True)
                pdf.cell(30, 10, "Qty", border=1, align="C", fill=True)
                pdf.cell(45, 10, "Amount (INR)", border=1, align="R", fill=True)
                pdf.ln(10)
                
                pdf.set_font("Helvetica", size=10)
                items = [
                    ("Specialist Consultation", "1", "Rs 1,500.00"),
                    ("Pulmonary Radiograph (X-Ray)", "1", "Rs 850.00"),
                    ("AI Neural Diagnostic Matrix", "1", "Rs 1,200.00"),
                    ("Facility & Equipment Fee", "1", "Rs 450.00")
                ]
                for item in items:
                    pdf.cell(110, 10, f"  {item[0]}", border=1)
                    pdf.cell(30, 10, item[1], border=1, align="C")
                    pdf.cell(45, 10, f"{item[2]}  ", border=1, align="R")
                    pdf.ln(10)
                    
                pdf.set_font("Helvetica", style="B", size=12)
                pdf.cell(140, 12, "TOTAL AMOUNT DUE", border=1, align="R")
                pdf.cell(45, 12, "Rs 4,000.00  ", border=1, align="R")
                pdf.ln(25)
                
                pdf.set_font("Helvetica", style="B", size=10)
                pdf.cell(0, 5, "_______________________", align="R", ln=1)
                pdf.cell(0, 5, "Authorized Signature     ", align="R", ln=1)
                
                pdf.ln(20)
                pdf.set_font("Helvetica", style="I", size=9)
                pdf.set_text_color(150, 150, 150)
                pdf.cell(0, 5, "Payment is due within 30 days. Thank you for trusting Aura Health.", align="C")
                
                bill_filename = f"Invoice_{pat_id}_{datetime.now().strftime('%H%M%S')}.pdf"
                bill_path = os.path.join(app.config['UPLOAD_FOLDER'], bill_filename)
                
                with open(bill_path, 'wb') as f:
                    f.write(bytes(pdf.output()))
                    
                return send_file(bill_path, as_attachment=True, download_name=f"AuraHealth_Invoice_{pat_id}.pdf")

        elif 'transmit_rx' in request.form:
            pat_mrn = request.form.get('rx_patient_mrn')
            medication = request.form.get('rx_medication')
            instructions = request.form.get('rx_instructions')
            
            pat = conn.execute("SELECT * FROM users WHERE username = ? AND role = 'patient'", (pat_mrn,)).fetchone()
            if not pat:
                flash(f"Transmission Failed: Patient MRN '{pat_mrn}' not registered.", "error")
            else:
                pdf = FPDF()
                pdf.add_page()
                
                pdf.set_fill_color(15, 23, 42)
                pdf.rect(15, 10, 14, 14, 'F')
                pdf.set_fill_color(100, 255, 218)
                pdf.rect(21, 12.5, 2, 9, 'F')
                pdf.rect(17.5, 16, 9, 2, 'F')
                
                pdf.set_xy(34, 11)
                pdf.set_font("Helvetica", style="B", size=24)
                pdf.set_text_color(15, 23, 42)
                pdf.cell(0, 8, "AURA HEALTH MEDICAL CENTER", align="L", ln=1)
                
                pdf.set_xy(34, 20)
                pdf.set_font("Helvetica", style="I", size=11)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 5, "Official Electronic Prescription (eRx)", align="L", ln=1)
                
                pdf.set_text_color(0, 0, 0)
                pdf.line(15, 30, 195, 30)
                pdf.set_y(38)

                pdf.set_font("Helvetica", style="B", size=12)
                pdf.cell(100, 6, f"Patient MRN: {pat_mrn}", ln=0)
                pdf.cell(90, 6, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=1)
                pdf.cell(100, 6, f"Prescribing Physician: Dr. {doc_info['name']}", ln=1)
                
                pdf.ln(15)
                
                pdf.set_font("Helvetica", style="B", size=36)
                pdf.cell(20, 15, "Rx", ln=1)
                
                pdf.set_font("Helvetica", style="B", size=14)
                pdf.cell(0, 10, f"Medication: {medication}", ln=1)
                
                pdf.set_font("Helvetica", size=12)
                pdf.multi_cell(0, 8, f"Sig (Instructions): {instructions}")
                
                pdf.ln(30)
                
                pdf.set_font("Helvetica", style="B", size=10)
                pdf.cell(0, 5, "_______________________", align="R", ln=1)
                pdf.cell(0, 5, f"Dr. {doc_info['name']}        ", align="R", ln=1)
                pdf.cell(0, 5, "DEA Number: AH-7299384   ", align="R", ln=1)
                
                rx_filename = f"Prescription_{pat_mrn}_{datetime.now().strftime('%H%M%S')}.pdf"
                rx_path = os.path.join(app.config['UPLOAD_FOLDER'], rx_filename)
                
                with open(rx_path, 'wb') as f:
                    f.write(bytes(pdf.output()))
                    
                return send_file(rx_path, as_attachment=True, download_name=f"AuraHealth_Rx_{pat_mrn}.pdf")

        elif 'patient_id' in request.form:
            patient_id = request.form['patient_id']
            patient_name = request.form.get('patient_name', 'System Auto-Fetch')
            patient_age = int(request.form.get('age', 0) or 0)
            patient_gender = request.form.get('gender', 'U')
            spo2 = int(request.form.get('spo2', 96) or 96)
            temp = float(request.form.get('temp', 37.0) or 37.0)
            
            file = request.files.get('xray_image')
            original_path = ""
            orig_relative_path = ""
            
            if file and file.filename != '':
                orig_relative_path = f"uploads/orig_{secure_filename(file.filename)}"
                original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"orig_{secure_filename(file.filename)}")
                file.save(original_path)
            else:
                apt = conn.execute('SELECT xray_path FROM appointments WHERE patient_username=? AND doctor_name=? AND xray_path != "" ORDER BY id DESC LIMIT 1', (patient_id, doc_info['name'])).fetchone()
                if apt and apt['xray_path']:
                    orig_relative_path = apt['xray_path']
                    original_path = os.path.join('static', orig_relative_path)
            
            if original_path and os.path.exists(original_path):
                img_resized = Image.open(original_path).convert('RGB').resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
                pred = model.predict(img_array, verbose=0)[0][0]
                confidence = float(max(pred, 1 - pred) * 100)
                
                etiology_text = ""
                if pred > 0.35:
                    diag = "PNEUMONIA DETECTED"
                    impression = f"Extensive Radiographic Analysis confirms multi-focal alveolar opacities consistent with acute pulmonary consolidation. Confidence: {confidence:.2f}%."
                    etiology_text = "ETIOLOGY & PATHOGENESIS (HOW & WHY IT FORMS):\nPneumonia develops when pathogenic microorganisms evade upper respiratory defenses and colonize the lower respiratory tract, triggering a robust inflammatory response. This causes alveolar sacs to fill with purulent exudate, severely impairing gas exchange."
                    recommendations = "- Initiate broad-spectrum empiric antimicrobial therapy immediately.\n- Continuous vitals monitoring required; administer O2 if SpO2 falls <92%.\n- Schedule follow-up radiograph in 48-72 hours to assess resolution."
                    home_prescription = "1. Medication: Complete the full course of prescribed antibiotics, even if symptoms improve.\n2. Hydration: Consume at least 2.5 to 3 liters of water daily to thin mucus secretions.\n3. Activity: Strict bed rest. Elevate head while sleeping to ease breathing."
                else:
                    diag = "NORMAL (HEALTHY)"
                    impression = f"Radiographic Analysis reveals clear lung fields bilaterally. Confidence: {confidence:.2f}%."
                    recommendations = "- Routine outpatient clinical follow-up as necessary.\n- No acute pharmacological or radiological intervention required at this time.\n- Monitor for any newly developing respiratory symptoms."
                    home_prescription = "1. Maintenance: Continue standard daily vitamin intake (Vitamin C, Zinc).\n2. Activity: Resume normal cardiovascular and daily activities."
                
                heatmap = get_gradcam_heatmap(img_array)
                heatmap_color = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (224, 224))), cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(np.array(img_resized), 0.6, heatmap_color, 0.4, 0)
                heat_filename = f"heat_pat_{patient_id}_{datetime.now().strftime('%H%M%S')}.jpg"
                heat_path = os.path.join(app.config['UPLOAD_FOLDER'], heat_filename)
                heat_relative_path = f"uploads/{heat_filename}"
                cv2.imwrite(heat_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
                
                pdf_filename = f"Report_{patient_id}_{datetime.now().strftime('%H%M%S')}.pdf"
                pdf_relative_path = f"uploads/{pdf_filename}"
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                conn.execute("INSERT INTO patients (timestamp, patient_id, name, age, sex, spo2, temp, diagnosis, confidence, ai_notes, doctor_name, orig_img, heat_img, pdf_link) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (timestamp, patient_id, patient_name, patient_age, patient_gender, spo2, temp, diag, round(confidence, 2), impression, doc_info['name'], orig_relative_path, heat_relative_path, pdf_relative_path))
                
                conn.execute("UPDATE appointments SET scan_status='DIAGNOSED BY DOCTOR' WHERE patient_username=? AND doctor_name=? AND xray_path != ''", (patient_id, doc_info['name']))
                conn.commit()
                
                pdf = FPDF()
                pdf.add_page()
                add_watermark(pdf)
                
                pdf.set_fill_color(15, 23, 42)
                pdf.rect(15, 10, 14, 14, 'F')
                pdf.set_fill_color(100, 255, 218)
                pdf.rect(21, 12.5, 2, 9, 'F')
                pdf.rect(17.5, 16, 9, 2, 'F')
                
                pdf.set_xy(34, 11)
                pdf.set_font("Helvetica", style="B", size=24)
                pdf.set_text_color(15, 23, 42)
                pdf.cell(0, 8, "AURA HEALTH MEDICAL CENTER", align="L", ln=1)
                
                pdf.set_xy(34, 20)
                pdf.set_font("Helvetica", style="I", size=11)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 5, "Department of Pulmonary Radiology - Automated EMR File", align="L", ln=1)
                
                pdf.set_text_color(0, 0, 0)
                pdf.line(15, 30, 195, 30)
                pdf.set_y(36)
                
                pdf.set_font("Helvetica", style="B", size=12)
                pdf.cell(0, 8, "PART I: PATIENT DEMOGRAPHICS & VITALS", ln=1)
                pdf.set_font("Helvetica", size=10)
                pdf.cell(90, 5, f"Patient MRN: {patient_id}", ln=0)
                pdf.cell(90, 5, f"Scan Date: {timestamp}", ln=1)
                pdf.cell(90, 5, f"Admitting SpO2: {spo2}%", ln=0)
                pdf.cell(90, 5, f"Admitting Temp: {temp}C", ln=1)
                pdf.ln(5)
                
                pdf.set_font("Helvetica", style="B", size=12)
                pdf.cell(0, 8, "RADIOLOGICAL FINDINGS & AI DIAGNOSIS", ln=1)
                pdf.cell(0, 6, f"PRIMARY DIAGNOSIS: {diag} (AI Confidence: {confidence:.2f}%)", ln=1)
                pdf.set_font("Helvetica", size=10)
                pdf.multi_cell(0, 5, f"CLINICAL IMPRESSION:\n{impression}")
                
                if etiology_text:
                    pdf.ln(2)
                    pdf.set_font("Helvetica", style="I", size=10)
                    pdf.multi_cell(0, 5, etiology_text)
                    
                pdf.set_y(150)
                pdf.image(original_path, x=25, y=pdf.get_y(), w=70)
                pdf.image(heat_path, x=115, y=pdf.get_y(), w=70)
                
                pdf.add_page()
                add_watermark(pdf)
                
                pdf.set_font("Helvetica", style="B", size=14)
                pdf.cell(0, 10, "PART II: PHARMACOLOGICAL & OUTPATIENT CARE PLAN", ln=1)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(8)
                
                pdf.set_font("Helvetica", style="B", size=11)
                pdf.cell(0, 8, "INPATIENT PROTOCOL RECOMMENDATIONS:", ln=1)
                pdf.set_font("Helvetica", size=10)
                pdf.multi_cell(0, 5, recommendations)
                pdf.ln(8)
                
                pdf.set_font("Helvetica", style="B", size=11)
                pdf.cell(0, 8, "OUTPATIENT / AT-HOME PRESCRIPTION PLAN:", ln=1)
                pdf.set_font("Helvetica", size=10)
                pdf.multi_cell(0, 5, home_prescription)
                pdf.ln(10)
                
                pdf.set_y(-40)
                pdf.line(120, pdf.get_y(), 190, pdf.get_y())
                pdf.set_xy(120, pdf.get_y() + 2)
                pdf.set_font("Helvetica", style="B", size=10)
                pdf.cell(70, 5, f"Attending: {doc_info['name']}", align="C", ln=1)
                
                with open(os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename), 'wb') as f:
                    f.write(bytes(pdf.output()))
                    
                result_data = {
                    "diagnosis": diag, "confidence": confidence, "impression": impression,
                    "recommendations": recommendations, "home_prescription": home_prescription,
                    "name": patient_name, "age": patient_age, "gender": patient_gender,
                    "orig_img": orig_relative_path, "heat_img": heat_relative_path, "pdf_link": pdf_relative_path
                }
            else:
                flash("No X-ray found. Please upload one or wait for patient upload.", "error")
    
    patients = conn.execute('SELECT * FROM patients WHERE doctor_name = ? ORDER BY timestamp DESC', (doc_info['name'],)).fetchall()
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    tomorrow_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    today_apts = conn.execute('SELECT * FROM appointments WHERE doctor_name = ? AND date = ? ORDER BY time_slot ASC', (doc_info['name'], today_str)).fetchall()
    tomorrow_apts = conn.execute('SELECT * FROM appointments WHERE doctor_name = ? AND date = ? ORDER BY time_slot ASC', (doc_info['name'], tomorrow_str)).fetchall()
    
    # FIX: Added 'AND date >= ?' so it strictly filters for current and upcoming days only
    apts = conn.execute('SELECT * FROM appointments WHERE doctor_name = ? AND date >= ? ORDER BY date ASC, time_slot ASC', (doc_info['name'], today_str)).fetchall()
    leaves = conn.execute('SELECT * FROM doctor_leaves WHERE doctor_name = ? ORDER BY leave_date DESC', (doc_info['name'],)).fetchall()
    
    conn.close()
    return render_template('doctor_dash.html', doctor=doc_info, result=result_data, patients=patients, appointments=apts, leaves=leaves, today_appointments=today_apts, tomorrow_appointments=tomorrow_apts)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing_page'))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)