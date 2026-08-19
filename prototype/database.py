"""
SmartCare Hospital -- No-Show Prediction Prototype
Task 08: AI Prototype Development (v3)

DATABASE (schema + connection helper + bulk migration)
----------------------------------------------------------
Tables:
  - staff        : login accounts for hospital staff
  - clients      : registered patients, keyed by NIC
  - doctors      : 4 doctors per department (28 total), each with 2 weekly
                    working days and one fixed time slot
  - appointments : booking records -- prediction, doctor, booking number,
                    payment, and status/lifecycle
  - admissions    : past hospital admission records (separate history)

Run this file directly once to create smartcare.db, seed the 28 doctors,
and bulk-import all 1000 rows of smartcare_ai_dataset_1000.csv as real
clients + their historical appointment (+ admission) records:
    python database.py
"""

import sqlite3
import random
from pathlib import Path
from datetime import date, timedelta
import pandas as pd

DB_PATH = Path("smartcare.db")
RAW_DATASET_PATH = Path("smartcare_ai_dataset_1000.csv")

DEPARTMENTS = ["Cardiology", "General Medicine", "Laboratory Services", "Neurology",
               "Orthopedics", "Pediatrics", "Radiology"]

# 28 doctors (4 per department), Sinhala names, each with a 2-day weekly
# schedule and one fixed time slot -- deterministic, not randomly generated,
# so the demo is reproducible.
DOCTOR_NAMES = [
    "Dr. Nimal Perera", "Dr. Kamal Silva", "Dr. Sunil Fernando", "Dr. Priyantha Jayasuriya",
    "Dr. Chandrika Wickramasinghe", "Dr. Anura Bandara", "Dr. Nalini Rajapaksha", "Dr. Ruwan Gunawardena",
    "Dr. Malini Dissanayake", "Dr. Ajith Karunaratne", "Dr. Chamari Wijesinghe", "Dr. Saman Kularatne",
    "Dr. Nadeeka Abeywardena", "Dr. Ranjith Herath", "Dr. Dilani Amarasinghe", "Dr. Susantha Ratnayake",
    "Dr. Kumari Senanayake", "Dr. Gayan Mendis", "Dr. Iresha Gamage", "Dr. Prasanna Weerasinghe",
    "Dr. Chathurika Peiris", "Dr. Lasantha Rathnayake", "Dr. Nirosha Wanigasekara", "Dr. Tharindu Ekanayake",
    "Dr. Sandya Liyanage", "Dr. Mahesh Wijekoon", "Dr. Vindya Samarasinghe", "Dr. Rohan Dias",
]

DAY_PAIRS = [("Monday", "Thursday"), ("Tuesday", "Friday"), ("Wednesday", "Saturday"), ("Monday", "Saturday")]
TIME_SLOTS = ["8:00 AM - 11:00 AM", "11:00 AM - 2:00 PM", "2:00 PM - 5:00 PM", "4:00 PM - 7:00 PM"]

WEEKDAY_NAME_TO_NUM = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                        "Friday": 4, "Saturday": 5, "Sunday": 6}

# Name pools used only to generate plausible client names for the bulk-imported
# dataset, which has no real names -- deterministic per patient_id, not random.
FIRST_NAMES_M = ["Nuwan", "Kasun", "Dinesh", "Chamara", "Isuru", "Sampath", "Roshan", "Janaka",
                  "Chathura", "Buddhika", "Prasad", "Asanka", "Manoj", "Thilina", "Sanjaya"]
FIRST_NAMES_F = ["Nadeesha", "Chathurika", "Dilrukshi", "Piumi", "Anusha", "Kavindi", "Hasini",
                  "Ishara", "Nilmini", "Sachini", "Yamuna", "Tharushi", "Madhavi", "Chalani", "Rukshan"]
LAST_NAMES = ["Perera", "Silva", "Fernando", "Jayasuriya", "Wickramasinghe", "Bandara", "Rajapaksha",
              "Gunawardena", "Dissanayake", "Karunaratne", "Wijesinghe", "Kularatne", "Abeywardena",
              "Herath", "Amarasinghe", "Ratnayake", "Senanayake", "Mendis", "Gamage", "Weerasinghe"]


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS staff (
            staff_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS clients (
            nic TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            contact_number TEXT,
            blood_group TEXT,
            address TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            day1 TEXT NOT NULL,
            day2 TEXT NOT NULL,
            time_slot TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_nic TEXT NOT NULL,
            doctor_id INTEGER,
            department TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            booking_date TEXT NOT NULL,
            appointment_date TEXT NOT NULL,
            booking_number INTEGER,
            status TEXT NOT NULL DEFAULT 'Pending',
            risk_label TEXT,
            risk_probability REAL,
            payment_type TEXT,
            payment_amount REAL,
            payment_confirmed INTEGER DEFAULT 0,
            entered_by_staff_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (client_nic) REFERENCES clients (nic),
            FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id),
            FOREIGN KEY (entered_by_staff_id) REFERENCES staff (staff_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS admissions (
            admission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_nic TEXT NOT NULL,
            admission_date TEXT NOT NULL,
            department TEXT,
            notes TEXT,
            FOREIGN KEY (client_nic) REFERENCES clients (nic)
        )
    """)

    conn.commit()
    conn.close()


def seed_doctors() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM doctors")
    if cur.fetchone()["n"] > 0:
        conn.close()
        return

    idx = 0
    for dept in DEPARTMENTS:
        for slot_in_dept in range(4):
            name = DOCTOR_NAMES[idx]
            day1, day2 = DAY_PAIRS[idx % 4]
            time_slot = TIME_SLOTS[(idx + 1) % 4]
            cur.execute(
                "INSERT INTO doctors (name, department, day1, day2, time_slot) VALUES (?, ?, ?, ?, ?)",
                (name, dept, day1, day2, time_slot),
            )
            idx += 1
    conn.commit()
    conn.close()
    print(f"Seeded {idx} doctors (4 per department x 7 departments).")


def _synthetic_nic(patient_id: str, seed_rng: random.Random) -> str:
    # 12-digit synthetic NIC-style number, deterministic per patient_id
    return "".join(str(seed_rng.randint(0, 9)) for _ in range(12)) + "V"


def _synthetic_name(patient_id: str, gender: str, seed_rng: random.Random) -> str:
    first = seed_rng.choice(FIRST_NAMES_M if gender == "Male" else FIRST_NAMES_F)
    last = seed_rng.choice(LAST_NAMES)
    return f"{first} {last}"


_STATUS_MAP = {
    "Completed": "Confirmed",   # attended -> confirmed/attended in our operational vocabulary
    "No-Show": "No-Show",
    "Cancelled": "Cancelled",
    "Scheduled": "Pending",     # still pending; auto-no-show logic will catch past-dated ones
}


def import_bulk_dataset() -> None:
    """
    One-time bulk import: every row of the raw 1000-row coursework dataset
    becomes a real client + their one historical appointment (+ synthetic
    admission records, matching their previous_admissions count). Safe to
    call every run -- skipped automatically if clients already exist.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM clients")
    if cur.fetchone()["n"] > 0:
        conn.close()
        print("Clients already populated -- skipping bulk import.")
        return

    if not RAW_DATASET_PATH.exists():
        conn.close()
        print(f"WARNING: {RAW_DATASET_PATH} not found -- bulk import skipped.")
        return

    df = pd.read_csv(RAW_DATASET_PATH)

    # Pre-load doctors grouped by department for round-robin assignment
    doctors_by_dept = {}
    for row in cur.execute("SELECT * FROM doctors").fetchall():
        doctors_by_dept.setdefault(row["department"], []).append(dict(row))

    dept_counters = {d: 0 for d in DEPARTMENTS}
    booking_number_counters = {}  # (doctor_id, appointment_date) -> running count

    clients_rows, appointments_rows, admissions_rows = [], [], []

    for _, r in df.iterrows():
        pid = r["patient_id"]
        rng = random.Random(pid)  # deterministic per patient, reproducible across runs

        nic = _synthetic_nic(pid, rng)
        name = _synthetic_name(pid, r["gender"], rng)
        clients_rows.append((nic, name, int(r["age"]), r["gender"], None, r["blood_group"], None))

        dept = r["department"] if r["department"] in DEPARTMENTS else DEPARTMENTS[0]
        dept_doctors = doctors_by_dept.get(dept, [])
        doctor = dept_doctors[dept_counters[dept] % len(dept_doctors)] if dept_doctors else None
        dept_counters[dept] += 1

        appt_date = pd.to_datetime(r["appointment_date"]).date()
        booking_date = appt_date - timedelta(days=int(r["waiting_days"]))
        status = _STATUS_MAP.get(r["appointment_status"], "Pending")

        doctor_id = doctor["doctor_id"] if doctor else None
        key = (doctor_id, str(appt_date))
        booking_number_counters[key] = booking_number_counters.get(key, 0) + 1
        booking_number = booking_number_counters[key]

        payment_type = r["payment_method"] if pd.notna(r.get("payment_method")) else None
        payment_amount = float(r["total_bill_lkr"]) if pd.notna(r.get("total_bill_lkr")) else None
        payment_confirmed = 1 if r.get("payment_status") == "Paid" else 0

        appointments_rows.append((
            nic, doctor_id, dept, r["diagnosis"], str(booking_date), str(appt_date), booking_number,
            status, None, None, payment_type, payment_amount, payment_confirmed, None,
        ))

        prev_admissions = int(r["previous_admissions"]) if pd.notna(r["previous_admissions"]) else 0
        for k in range(prev_admissions):
            adm_date = appt_date - timedelta(days=90 * (k + 1))
            admissions_rows.append((nic, str(adm_date), dept, "Historical admission (bulk import)"))

    cur.executemany(
        "INSERT OR IGNORE INTO clients (nic, full_name, age, gender, contact_number, blood_group, address) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)", clients_rows,
    )
    cur.executemany(
        "INSERT INTO appointments (client_nic, doctor_id, department, diagnosis, booking_date, "
        "appointment_date, booking_number, status, risk_label, risk_probability, payment_type, "
        "payment_amount, payment_confirmed, entered_by_staff_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", appointments_rows,
    )
    cur.executemany(
        "INSERT INTO admissions (client_nic, admission_date, department, notes) VALUES (?, ?, ?, ?)",
        admissions_rows,
    )

    conn.commit()
    conn.close()
    print(f"Bulk-imported {len(clients_rows)} clients, {len(appointments_rows)} appointments, "
          f"{len(admissions_rows)} admission records from {RAW_DATASET_PATH.name}.")


if __name__ == "__main__":
    initialize_database()
    seed_doctors()
    import_bulk_dataset()
    print("Database ready:", DB_PATH.resolve())
