# 🛡️ AI Smart Hostel Security & Attendance System

An AI-powered real-time surveillance and attendance platform using face recognition, live camera monitoring, automated attendance logging, and Telegram alerts to secure hostel and campus environments.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🎥 **Live Face Recognition** | Detects and recognises students in real time via InsightFace embeddings |
| 📊 **Smart Attendance Tracking** | SQLite-backed daily logs with real-time analytics dashboard |
| 📷 **Unknown Visitor Detection** | Captures and saves snapshots of unrecognised faces |
| 🤖 **Telegram Security Alerts** | Instant alerts for unknown visitors, loitering, and security threats |
| 🔐 **Threat Detection Engine** | 3-mode threat detector — repeated unknowns, loitering, rapid failures |
| ⚡ **Centroid Face Tracker** | Tracks faces across frames; recognition cached 2 s → 15–25 FPS on CPU |

---

## 🧠 System Architecture

```
Camera Feed
      │
      ▼
Face Detection (InsightFace)
      │
      ▼
CentroidTracker
      │
      ▼
Face Recognition (best cosine match)
      │
 ┌────┴────────────────┐
 │                     │
 ▼                     ▼
Attendance DB       Unknown Capture
 │                     │
 ▼                     ▼
Dashboard          Telegram Alerts
                       │
                       ▼
               SecurityService
          (REPEATED_UNKNOWN | LOITERING | RAPID_FAIL)
```

---

## 🗂 Project Structure

```
AI Smart Hostel Attendance System/
│
├── config/
│   ├── __init__.py
│   └── settings.py          ← single source of truth for all parameters
│
├── core/
│   ├── face_engine.py       ← InsightFace ArcFace detection
│   ├── recognition.py       ← embeddings I/O, best-match, registration
│   ├── attendance.py        ← SQLite attendance queries
│   ├── anti_spoof.py        ← blink-based liveness detection
│   └── tracker.py           ← centroid face tracker with 2 s recognition cache
│
├── services/
│   ├── telegram_service.py  ← Telegram Bot API (sendMessage / sendPhoto)
│   ├── greeting_service.py  ← offline TTS voice greeting (pyttsx3)
│   ├── security_service.py  ← threat detection engine + SQLite event log
│   └── telegram_bot_listener.py ← interactive bot (/start /updates /attendance)
│
├── runtime/
│   └── camera_runtime.py    ← main live loop (tracker + all services wired)
│
├── dashboard/
│   ├── app.py               ← production 7-page Streamlit panel
│   └── dashboard.py         ← extended 9-page variant
│
├── data/
│   ├── attendance.db        ← SQLite (attendance + security_events tables)
│   └── embeddings/          ← per-student .pkl files (20 embeddings each)
│
└── captures/                ← timestamped face JPEGs
```

---

## ⚙️ Installation

```bash
# 1. Clone
git clone https://github.com/yourusername/ai-smart-hostel-security.git
cd ai-smart-hostel-security

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt
```

---

## 📡 Telegram Bot Setup

### 1 — Create a bot
Message `@BotFather` on Telegram and run `/newbot`. Copy the **BOT TOKEN**.

### 2 — Get your Chat ID
```
https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
```
Look for `"chat": {"id": <number>}`. That number is your `CHAT_ID`.

### 3 — Add credentials to settings
Edit `config/settings.py`:
```python
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID"
```

---

## ▶️ Running the System

### 1 — Live camera recognition
```bash
source venv/bin/activate
python runtime/camera_runtime.py
```
Starts the webcam, runs face detection + recognition, logs attendance, sends Telegram alerts for unknown visitors.

### 2 — Telegram bot listener (optional)
```bash
python services/telegram_bot_listener.py
```

Available commands:

| Command | Response |
|---|---|
| `/start` | Welcome message |
| `/updates` | System summary (attendance + alerts) |
| `/attendance` | Today's present students |
| `/alerts` | Recent security events |

### 3 — Dashboard
```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501)

---

## 📊 Database Schema

**`attendance` table** — `data/attendance.db`

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Auto-increment PK |
| name | TEXT | Student name (lowercase) |
| date | TEXT | YYYY-MM-DD |
| time | TEXT | HH:MM:SS |

**`security_events` table** — same DB

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Auto-increment PK |
| timestamp | TEXT | ISO-8601 datetime |
| event_type | TEXT | `ENTRY` \| `REPEATED_UNKNOWN` \| `LOITERING` \| `RAPID_FAIL` |
| person | TEXT | Recognised name or `Unknown` |
| confidence | REAL | Cosine similarity score |
| camera_id | INTEGER | Camera index |

---

## 🔒 Threat Detection

| Threat | Trigger | Alert |
|---|---|---|
| **Repeated Unknown** | Same unknown face ≥ 3× in 60 s | Telegram + DB log |
| **Loitering** | Face in frame > 30 s | Telegram + DB log |
| **Rapid Failures** | ≥ 5 recognition failures in 10 s | Telegram + DB log |

---

## 🧾 Unknown Visitor Evidence

Snapshots saved to `captures/`:
```
captures/unknown_2026-03-06_05-12-15.jpg
captures/naman_2026-03-06_08-45-02.jpg
```
Unknown captures are automatically attached to Telegram alerts.

---

## 📈 Future Improvements

- Multi-camera monitoring
- Face mask / anti-spoofing (blink detection module already included)
- Cloud deployment (AWS / GCP)
- Admin authentication for dashboard
- Mobile app integration
- Face age & gender estimation overlay

---

## 🧑‍💻 Technologies

`Python` · `OpenCV` · `InsightFace` · `Streamlit` · `SQLite` · `Telegram Bot API` · `NumPy` · `pyttsx3` · `MediaPipe`

---

## 🏫 Use Cases

- Hostel entry security
- Campus attendance systems
- Office / lab access control
- Smart building monitoring

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 👨‍💻 Author

**Naman** — AI / Computer Vision Developer  
⭐ Star this project on GitHub if it helped you!
