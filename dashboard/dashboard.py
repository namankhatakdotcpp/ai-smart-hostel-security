"""
dashboard/dashboard.py — Smart Hostel AI Security Panel (v2)
=============================================================
8-page production Streamlit dashboard.

Run:
    streamlit run dashboard/dashboard.py
"""

import sys
import os
import sqlite3
import glob
import time
import json
from datetime import datetime, date, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup (must happen before any project imports) ───────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Project imports ────────────────────────────────────────────────────────────
from core.recognition          import (
    load_all_embeddings, find_best_match, save_capture,
    register_student, list_all_students, delete_student,
    EMBEDDINGS_DIR,
)
from core.attendance           import (
    mark_attendance, get_today_records, get_all_records,
    delete_student_records, DB_PATH,
)
from services.security_service import load_threat_log
from config.settings           import CAPTURES_DIR, CAMERA_ID

SECURITY_EVENT_LOG = os.path.join(ROOT, "data", "security_events.json")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Hostel AI Security Panel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"]  { background:#0d1117; }
  [data-testid="stHeader"]            { background:#0d1117; border-bottom:1px solid #21262d; }
  [data-testid="stSidebar"]           { background:#161b22; border-right:1px solid #30363d; }
  section[data-testid="stSidebar"] *  { color:#c9d1d9 !important; }
  h1,h2,h3,h4                         { color:#f0f6fc !important; }
  p, li, span, label, div            { color:#c9d1d9; }
  hr                                  { border-color:#30363d !important; }

  .kpi { background:#161b22; border:1px solid #30363d; border-radius:10px;
         padding:1rem 1.3rem; margin-bottom:.4rem; }
  .kpi .val { font-size:2rem; font-weight:700; margin:0; }
  .kpi .lbl { color:#8b949e; font-size:.8rem; margin:0; }

  .alert-box { background:#2d1b1b; border:1px solid #da3633; border-radius:8px;
               padding:.8rem 1rem; margin:.4rem 0; }
  .ok-box    { background:#1b2d1b; border:1px solid #238636; border-radius:8px;
               padding:.8rem 1rem; margin:.4rem 0; }
  .info-box  { background:#1b1f2d; border:1px solid #1f6feb; border-radius:8px;
               padding:.8rem 1rem; margin:.4rem 0; }

  [data-testid="stDataFrame"] > div { background:#161b22 !important; }
  .stImage > div > div > p          { color:#8b949e !important; font-size:.75rem; }
  .stProgress > div > div > div     { background:#1f6feb !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=5)
def load_full_df() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["id", "name", "date", "time"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM attendance ORDER BY date DESC, time DESC", conn
        )
    if not df.empty:
        df["name"] = df["name"].str.title()
    return df


@st.cache_data(ttl=5)
def load_security_events() -> list[dict]:
    if not os.path.exists(SECURITY_EVENT_LOG):
        return []
    try:
        with open(SECURITY_EVENT_LOG) as f:
            return json.load(f)
    except Exception:
        return []


def today_df(full: pd.DataFrame) -> pd.DataFrame:
    today = date.today().isoformat()
    return full[full["date"] == today] if not full.empty else full


def week_df(full: pd.DataFrame) -> pd.DataFrame:
    week_ago = (date.today() - timedelta(days=6)).isoformat()
    return full[full["date"] >= week_ago] if not full.empty else full


def get_captures(prefix: str = "") -> list[str]:
    return sorted(glob.glob(os.path.join(CAPTURES_DIR, f"{prefix}*.jpg")), reverse=True)


def known_captures() -> list[str]:
    return [f for f in get_captures() if not os.path.basename(f).startswith("unknown_")]


def load_rgb(path: str):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


# ── UI Components ──────────────────────────────────────────────────────────────

def kpi(col, label: str, value, color: str = "#58a6ff"):
    with col:
        st.markdown(
            f'<div class="kpi"><p class="lbl">{label}</p>'
            f'<p class="val" style="color:{color}">{value}</p></div>',
            unsafe_allow_html=True,
        )


def image_grid(files: list[str], cols: int = 3, max_show: int = 12):
    files = files[:max_show]
    columns = st.columns(cols)
    for i, fp in enumerate(files):
        img = load_rgb(fp)
        if img is None:
            continue
        parts = os.path.basename(fp).replace(".jpg", "").split("_")
        name  = parts[0].title()
        ts    = " ".join(parts[1:])
        with columns[i % cols]:
            st.image(img, caption=f"{name} · {ts}", use_container_width=True)
    if len(files) == max_show:
        st.caption(f"Showing {max_show} most recent.")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Smart Hostel AI")
    st.markdown("**Security Panel v2**")
    st.markdown("---")
    page = st.radio("", [
        "🏠  Overview",
        "📷  Live Recognition",
        "📝  Register Student",
        "👥  Recognized Students",
        "🚨  Unknown Visitors",
        "🔐  Security Events",
        "📊  Analytics",
        "⚙️  Admin Panel",
        "📋  Daily Logs",
    ], label_visibility="collapsed")
    st.markdown("---")
    auto_ref = st.checkbox("Auto-refresh (5s)", value=False)
    if auto_ref:
        time.sleep(5)
        st.rerun()
    st.markdown(
        f"<small>Updated: {datetime.now().strftime('%H:%M:%S')}</small>",
        unsafe_allow_html=True,
    )

# Pre-load data used by multiple pages
full    = load_full_df()
t_df    = today_df(full)
threats = load_threat_log()
events  = load_security_events()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("🛡️ Smart Hostel AI Security Panel")
    st.caption(f"Today — {date.today().strftime('%A, %d %B %Y')}")
    st.markdown("---")

    students    = list_all_students()
    unk_caps    = get_captures("unknown_")
    kn_caps     = known_captures()
    dwell_alerts = [e for e in events if e.get("event_type") == "DWELL_ALERT"]

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "👥 Registered Students",    len(students))
    kpi(c2, "✅ Present Today",           t_df["name"].nunique() if not t_df.empty else 0, "#3fb950")
    kpi(c3, "🚨 Unknown Detections",      len(unk_caps), "#da3633")
    kpi(c4, "⚠️  Security Alerts",        len(threats), "#f0883e")
    kpi(c5, "⏱️  Dwell Alerts",            len(dwell_alerts), "#a371f7")

    st.markdown("---")
    col_a, col_b = st.columns([1.6, 1])

    with col_a:
        st.markdown("### 🕐 Today's Activity")
        if t_df.empty:
            st.info("No activity yet today.")
        else:
            disp = t_df[["name", "time"]].head(10).rename(
                columns={"name": "Student", "time": "Time"}
            )
            st.dataframe(disp, use_container_width=True, hide_index=True)

        # Recent security events (last 5)
        if events:
            st.markdown("### 🔐 Recent Security Events")
            for ev in reversed(events[-5:]):
                etype = ev.get("event_type", "—")
                color = "alert-box" if etype in ("UNKNOWN", "DWELL_ALERT") else "ok-box"
                label = {"ENTRY": "✅ Entry", "UNKNOWN": "⚠️ Unknown",
                         "DWELL_ALERT": "⏱️ Dwell Alert"}.get(etype, etype)
                st.markdown(
                    f'<div class="{color}">{label} — <b>{ev.get("person","—")}</b>'
                    f' | conf {ev.get("confidence",0):.2f}'
                    f' | {ev.get("timestamp","")[:19].replace("T"," ")}</div>',
                    unsafe_allow_html=True,
                )

    with col_b:
        st.markdown("### 📸 Latest Capture")
        if kn_caps:
            img = load_rgb(kn_caps[0])
            if img is not None:
                name_part = os.path.basename(kn_caps[0]).split("_")[0].title()
                st.image(img, caption=f"{name_part} — latest", use_container_width=True)
        else:
            st.info("No captures yet.")

        if threats:
            st.markdown("### ⚠️ Last Security Alert")
            last = threats[-1]
            st.markdown(
                f'<div class="alert-box">🚨 <b>{last["count"]} unknowns</b> in '
                f'{last["window_s"]//60} min — {last["time_str"]}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════
elif "Live Recognition" in page:
    st.title("📷 Live Recognition")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("Click the button to grab a live frame with face recognition overlay.")
        if st.button("📸 Capture & Recognise Now"):
            from core.face_engine import FaceEngine

            with st.spinner("Initialising model and capturing frame…"):
                engine = FaceEngine()
                db     = load_all_embeddings()
                cap    = cv2.VideoCapture(CAMERA_ID)
                ret, frame = cap.read()
                cap.release()

            if ret and frame is not None:
                faces = engine.process_frame(frame)
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    if db:
                        name, score, _ = find_best_match(face.embedding, db)
                    else:
                        name, score = "No DB", 0.0
                    color = (0, 220, 0) if name != "Unknown" else (0, 0, 220)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name}  {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         caption=f"{len(faces)} face(s) — {datetime.now().strftime('%H:%M:%S')}",
                         use_container_width=True)
            else:
                st.error("Could not capture frame from camera.")
        else:
            st.info("Click the button above to grab a live frame.")

    with col2:
        st.markdown("#### Run Full Live System")
        st.code("source venv/bin/activate\npython runtime/camera_runtime.py", language="bash")
        st.markdown("---")
        st.markdown("#### Tracker Info")
        st.markdown(
            '<div class="info-box">v2 runtime uses <b>CentroidTracker</b><br>'
            'Recognition fires <b>once per new face</b>, cached 2 seconds.<br>'
            'Target: <b>15–25 FPS</b> on CPU.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Registered")
        for s in list_all_students():
            st.markdown(f"• **{s['name'].title()}** — {s['embedding_count']} samples")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REGISTER STUDENT
# ═══════════════════════════════════════════════════════════════════════════════
elif "Register Student" in page:
    st.title("📝 Register New Student")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Camera Registration (20 samples)")
        with st.form("reg_form"):
            r_name = st.text_input("Student Name", placeholder="e.g. Naman Sharma")
            r_room = st.text_input("Room Number",  placeholder="e.g. A-203")
            submit = st.form_submit_button("🎥 Start Face Capture")

        if submit:
            if not r_name.strip():
                st.error("Please enter a student name.")
            else:
                st.info(f"Opening webcam for '{r_name}'… An OpenCV window will appear.")
                with st.spinner("Follow the on-screen guide — 20 samples needed…"):
                    ok = register_student(r_name.strip(), r_room.strip())
                if ok:
                    st.success(f"✅ '{r_name}' registered with 20 samples!")
                    st.cache_data.clear()
                else:
                    st.error("Registration failed or aborted.")

    with col2:
        st.markdown("### Currently Registered")
        students = list_all_students()
        if students:
            df_s = pd.DataFrame(students)
            df_s.columns = ["Name", "Room", "Embeddings"]
            df_s["Name"] = df_s["Name"].str.title()
            st.dataframe(df_s, use_container_width=True, hide_index=True)
        else:
            st.info("No students registered yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOGNIZED STUDENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Recognized Students" in page:
    st.title("👥 Recognized Students")
    st.markdown("---")
    kn = known_captures()
    if not kn:
        st.info("No student captures yet. Run `python runtime/camera_runtime.py`.")
    else:
        st.markdown(f"**{len(kn)} capture(s)**")
        image_grid(kn, cols=3, max_show=15)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: UNKNOWN VISITORS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Unknown Visitors" in page:
    st.title("🚨 Unknown Visitors")
    st.markdown("---")

    unk = get_captures("unknown_")
    c1, c2, c3 = st.columns(3)
    c1.metric("Unknown Captures",     len(unk))
    c2.metric("Security Alerts",      len(threats))
    dwell_count = len([e for e in events if e.get("event_type") == "DWELL_ALERT"])
    c3.metric("Dwell Alerts (>30s)",  dwell_count)

    if threats:
        st.markdown("### ⚠️ Rolling-Window Security Alerts")
        for t in reversed(threats[-5:]):
            st.markdown(
                f'<div class="alert-box">🚨 <b>{t["count"]} unknowns</b> in '
                f'{t["window_s"]//60} min — {t["time_str"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    if not unk:
        st.success("No unknown visitors captured. ✅")
    else:
        image_grid(unk, cols=3, max_show=12)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SECURITY EVENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Security Events" in page:
    st.title("🔐 Security Event Log")
    st.caption("Structured log from runtime/camera_runtime.py — all face events.")
    st.markdown("---")

    if not events:
        st.info("No security events yet. Run `python runtime/camera_runtime.py` to start logging.")
    else:
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            types = ["All"] + sorted({e.get("event_type","") for e in events})
            sel_type = st.selectbox("Event Type", types)
        with col2:
            persons = ["All"] + sorted({e.get("person","") for e in events})
            sel_person = st.selectbox("Person", persons)

        filtered = [e for e in events
                    if (sel_type == "All" or e.get("event_type") == sel_type)
                    and (sel_person == "All" or e.get("person") == sel_person)]

        st.markdown(f"**{len(filtered)} event(s)**")

        if filtered:
            df_ev = pd.DataFrame(filtered)
            df_ev["timestamp"] = df_ev["timestamp"].str[:19].str.replace("T", " ")
            df_ev = df_ev[["timestamp", "event_type", "person", "confidence", "camera_id"]]
            df_ev.columns = ["Timestamp", "Event", "Person", "Confidence", "Camera"]
            st.dataframe(df_ev, use_container_width=True, hide_index=True)

            csv = df_ev.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV", csv, "security_events.csv", "text/csv")

        # Event type breakdown chart
        if len(events) >= 3:
            st.markdown("---")
            st.markdown("#### Event Type Breakdown")
            type_counts = pd.Series(
                [e.get("event_type") for e in events]
            ).value_counts().reset_index()
            type_counts.columns = ["Event Type", "Count"]
            st.bar_chart(type_counts.set_index("Event Type"))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.title("📊 Attendance Analytics")
    st.markdown("---")

    if full.empty:
        st.warning("No attendance data yet.")
        st.stop()

    w_df = week_df(full)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📅 Daily Entries — Last 7 Days")
        daily = (
            w_df.groupby("date").size().reset_index(name="Entries").sort_values("date")
        )
        daily["date"] = pd.to_datetime(daily["date"])
        st.bar_chart(daily.set_index("date")["Entries"])

    with col2:
        st.markdown("#### 👤 Total Attendance per Student")
        per_s = (
            full.groupby("name").size().reset_index(name="Days")
            .sort_values("Days", ascending=False).set_index("name")
        )
        st.bar_chart(per_s["Days"])

    st.markdown("---")
    col3, col4 = st.columns([1, 2])

    with col3:
        st.markdown(f"#### ✅ Today ({date.today().isoformat()})")
        st.metric("Students Present",   t_df["name"].nunique() if not t_df.empty else 0)
        st.metric("Total Entries",      len(t_df))
        st.metric("Unknown Captures",   len(get_captures("unknown_")))
        st.metric("Security Events",    len(events))

    with col4:
        st.markdown("#### 📈 Weekly Attendance Rate")
        total_days = w_df["date"].nunique()
        if total_days > 0:
            rate = w_df.groupby("name").size().reset_index(name="Days Present")
            rate["Attendance %"] = (rate["Days Present"] / total_days * 100).round(1)
            rate = rate.sort_values("Attendance %", ascending=False)
            rate.columns = ["Student", "Days Present", "Attendance %"]
            st.dataframe(rate, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ADMIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
elif "Admin" in page:
    st.title("⚙️ Admin Panel")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["👤 Student List", "➕ Add Student", "🗑️ Remove Student"])

    with tab1:
        students = list_all_students()
        if students:
            df_s = pd.DataFrame(students)
            df_s.columns = ["Name", "Room", "Embeddings"]
            df_s["Name"] = df_s["Name"].str.title()
            st.dataframe(df_s, use_container_width=True, hide_index=True)
            st.metric("Total Registered", len(students))
        else:
            st.info("No students registered yet.")

    with tab2:
        st.markdown("Opens the webcam and captures 20 face samples.")
        with st.form("admin_add"):
            a_name = st.text_input("Full Name")
            a_room = st.text_input("Room Number")
            a_go   = st.form_submit_button("🎥 Start Capture")
        if a_go:
            if not a_name.strip():
                st.error("Name is required.")
            else:
                with st.spinner(f"Capturing 20 samples for '{a_name}'…"):
                    ok = register_student(a_name.strip(), a_room.strip())
                if ok:
                    st.success(f"✅ '{a_name}' registered.")
                    st.cache_data.clear()
                else:
                    st.error("Registration failed.")

    with tab3:
        students = list_all_students()
        if not students:
            st.info("No students to remove.")
        else:
            to_del = st.selectbox("Select student", [s["name"].title() for s in students])
            st.warning(f"Deletes **{to_del}'s** embedding file and all attendance records.")
            if st.button("🗑️ Confirm Delete"):
                try:
                    delete_student(to_del)
                    delete_student_records(to_del)
                    st.success(f"✅ '{to_del}' removed.")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DAILY LOGS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Daily Logs" in page:
    st.title("📋 Attendance Logs")
    st.markdown("---")

    if full.empty:
        st.warning("No records yet.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        sel_name = st.selectbox("Student",
                                ["All"] + sorted(full["name"].unique().tolist()))
    with col2:
        sel_date = st.selectbox("Date",
                                ["All"] + sorted(full["date"].unique().tolist(), reverse=True))

    filt = full.copy()
    if sel_name != "All":
        filt = filt[filt["name"] == sel_name]
    if sel_date != "All":
        filt = filt[filt["date"] == sel_date]

    st.markdown(f"**{len(filt)} record(s)**")
    st.dataframe(
        filt[["name", "date", "time"]].rename(
            columns={"name": "Student", "date": "Date", "time": "Time"}
        ),
        use_container_width=True, hide_index=True,
    )
    st.markdown("---")
    st.download_button(
        "⬇️ Download CSV",
        filt[["name", "date", "time"]].to_csv(index=False).encode(),
        f"attendance_{date.today()}.csv", "text/csv",
    )