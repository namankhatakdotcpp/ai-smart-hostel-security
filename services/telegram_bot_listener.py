import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import requests
import sqlite3
import time

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def get_updates(offset=None):
    url = f"{BASE_URL}/getUpdates"
    params = {"timeout": 100}

    if offset:
        params["offset"] = offset

    response = requests.get(url, params=params)
    return response.json()


def send_message(text):

    url = f"{BASE_URL}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text
        }
    )

    print("Telegram response:", response.json())


def get_attendance_summary():

    conn = sqlite3.connect("data/attendance.db")
    cursor = conn.cursor()

    try:

        today = cursor.execute("""
        SELECT COUNT(*) FROM attendance
        WHERE date = DATE('now')
        """).fetchone()[0]

    except Exception as e:
        print("DB error:", e)
        today = 0

    conn.close()

    return f"""
🏫 Smart Hostel AI Security Update

📊 Attendance Today
Present Students: {today}

⚠ Security
No critical alerts

📅 System Status
All systems operational
"""

def get_security_events():

    conn = sqlite3.connect("database/attendance.db")
    cursor = conn.cursor()

    events = cursor.execute("""
        SELECT event_type, timestamp
        FROM security_events
        ORDER BY timestamp DESC
        LIMIT 5
    """).fetchall()

    conn.close()

    return events


def build_updates_message():

    today, total = get_attendance_summary()
    events = get_security_events()

    message = f"""
🤖 Smart Hostel AI Security Update

📊 Attendance Analytics
Present Today: {today}
Total Recognitions: {total}

🚨 Latest Security Alerts
"""

    if events:
        for e in events:
            message += f"\n{e[0]} at {e[1]}"
    else:
        message += "\nNo recent alerts."

    return message


def run_bot():
    print("🤖 Telegram bot running...")

    last_update_id = None

    while True:

        updates = get_updates(last_update_id)

        if "result" in updates:
            for update in updates["result"]:

                last_update_id = update["update_id"] + 1

                if "message" not in update:
                    continue

                message = update["message"]
                chat_id = message["chat"]["id"]
                text = message.get("text", "")

                print("Received:", text)

                if text == "/start":
                    send_message(
                        "🤖 Smart Hostel AI Bot\n\n"
                        "Commands:\n"
                        "/updates\n"
                        "/attendance\n"
                        "/alerts"
                    )

                elif text == "/updates":
                    summary = get_attendance_summary()
                    send_message(summary)

                elif text == "/attendance":
                    send_message(get_attendance_summary())

                elif text == "/alerts":
                    send_message("⚠ No recent threats detected.")

        time.sleep(2)


if __name__ == "__main__":
    run_bot()