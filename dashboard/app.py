"""
dashboard/app.py — Smart Hostel AI Security Panel (production)
==============================================================
7-page real-time Streamlit security monitoring dashboard.

Run:
    streamlit run dashboard/app.py

Pages:
    1. Overview         — KPI cards + live activity
    2. Live Camera      — OpenCV snapshot with face-box overlay
    3. Recognized       — Attendance table + captures gallery
    4. Unknown Visitors — Unknown captures + alert log
    5. Analytics        — Charts (daily / weekly / student breakdown)
    6. Daily Logs       — Security events table (SQLite)
    7. Admin Panel      — Register / delete students + embedding viewer
"""

import sys
import os
import glob
import sqlite3
import time
from datetime import datetime, date, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── Resolve project root & import project modules ──────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.settings           import CAPTURES_DIR, CAMERA_ID, DB_PATH, SIMILARITY_THRESHOLD
from core.recognition          import (
    load_all_embeddings, find_best_match, register_student,
    list_all_students, delete_student,
)
from core.attendance           import delete_student_records
from services.security_service import get_recent_events

# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Smart Hostel AI Security Panel",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Dark GitHub-style theme ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ─────────────────────────────────────── */
[data-testid="stAppViewContainer"]{ background:#0d1117; }
[data-testid="stHeader"]          { background:#0d1117; border-bottom:1px solid #21262d; }
[data-testid="stSidebar"]         { background:#161b22; border-right:1px solid #21262d; }
section[data-testid="stSidebar"] *{ color:#c9d1d9 !important; }

h1,h2,h3,h4  { color:#f0f6fc !important; }
p,li,label   { color:#c9d1d9; }
hr           { border-color:#30363d !important; }

/* ── KPI cards ────────────────────────────────── */
.kpi{
  background:#161b22; border:1px solid #30363d;
  border-radius:12px; padding:1.1rem 1.4rem;
  margin-bottom:.5rem;
}
.kpi .val{ font-size:2.2rem; font-weight:700; margin:0; line-height:1; }
.kpi .lbl{ color:#8b949e; font-size:.78rem; margin:.3rem 0 0; }

/* ── Alert boxes ──────────────────────────────── */
.box-red  { background:#2d1b1b; border:1px solid #da3633;
            border-radius:8px; padding:.75rem 1rem; margin:.4rem 0; }
.box-green{ background:#1b2d1b; border:1px solid #238636;
            border-radius:8px; padding:.75rem 1rem; margin:.4rem 0; }
.box-blue { background:#1b1f2d; border:1px solid #1f6feb;
            border-radius:8px; padding:.75rem 1rem; margin:.4rem 0; }

/* ── Tables & images ──────────────────────────── */
[data-testid="stDataFrame"]>div { background:#161b22 !important; }
.stImage>div>div>p              { color:#8b949e !important; font-size:.73rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS  (cached for 5 s so auto-refresh is fast)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=5)
def _attendance_df() -> pd.DataFrame:
    """Load full attendance table from SQLite."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["id", "name", "date", "time"])
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM attendance ORDER BY date DESC, time DESC", conn)
    if not df.empty:
        df["name"] = df["name"].str.title()
    return df


@st.cache_data(ttl=5)
def _security_events_df(limit: int = 200) -> pd.DataFrame:
    """Load recent rows from SQLite security_events table."""
    rows = get_recent_events(limit)
    if not rows:
        return pd.DataFrame(columns=["id","timestamp","event_type","person","confidence","camera_id"])
    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].str[:19].str.replace("T", " ")
    return df


def _today(df: pd.DataFrame) -> pd.DataFrame:
    today = date.today().isoformat()
    return df[df["date"] == today] if not df.empty else df


def _week(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=6)).isoformat()
    return df[df["date"] >= cutoff] if not df.empty else df


def _captures(prefix: str = "") -> list[str]:
    return sorted(
        glob.glob(os.path.join(CAPTURES_DIR, f"{prefix}*.jpg")), reverse=True
    )


def _known_captures() -> list[str]:
    return [p for p in _captures() if not os.path.basename(p).startswith("unknown_")]


def _load_rgb(path: str):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


# ── UI primitives ──────────────────────────────────────────────────────────────

def kpi_card(col, label: str, value, color: str = "#58a6ff") -> None:
    with col:
        st.markdown(
            f'<div class="kpi">'
            f'<p class="val" style="color:{color}">{value}</p>'
            f'<p class="lbl">{label}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


def image_grid(files: list[str], cols: int = 3, max_n: int = 12) -> None:
    files   = files[:max_n]
    columns = st.columns(cols)
    for i, fp in enumerate(files):
        img = _load_rgb(fp)
        if img is None:
            continue
        parts = os.path.basename(fp).replace(".jpg", "").split("_")
        name  = parts[0].title()
        ts    = " ".join(parts[1:])
        with columns[i % cols]:
            st.image(img, caption=f"{name} · {ts}", use_container_width=True)
    if len(files) == max_n:
        st.caption(f"Showing {max_n} most recent captures.")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Smart Hostel AI")
    st.markdown("**Security Panel**")
    st.markdown("---")

    PAGE = st.radio(
        "",
        [
            "🏠  Overview",
            "📷  Live Camera",
            "👥  Recognized Students",
            "🚨  Unknown Visitors",
            "📊  Analytics",
            "📋  Daily Logs",
            "⚙️  Admin Panel",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    # Auto-refresh toggle
    auto_ref = st.checkbox("⚡ Auto-refresh (5 s)", value=True)
    if auto_ref:
        time.sleep(5)
        st.rerun()

    st.markdown(
        f"<small style='color:#8b949e'>Updated: "
        f"{datetime.now().strftime('%H:%M:%S')}</small>",
        unsafe_allow_html=True,
    )

# Pre-load shared data every page render
full_df  = _attendance_df()
today_df = _today(full_df)
sec_df   = _security_events_df()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in PAGE:
    st.title("🛡️ Smart Hostel AI Security Panel")
    st.caption(f"Live dashboard — {date.today().strftime('%A, %d %B %Y')}")
    st.markdown("---")

    students    = list_all_students()
    unk_today   = [p for p in _captures("unknown_")
                   if date.today().isoformat() in p]
    alerts_today = sec_df[
        sec_df["timestamp"].str.startswith(date.today().isoformat())
    ] if not sec_df.empty else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "👥 Total Students Registered",
             len(students))
    kpi_card(c2, "✅ Students Present Today",
             today_df["name"].nunique() if not today_df.empty else 0,
             "#3fb950")
    kpi_card(c3, "🚨 Unknown Visitors Today",
             len(unk_today), "#da3633")
    kpi_card(c4, "⚠️  Security Alerts Today",
             len(alerts_today), "#f0883e")

    st.markdown("---")
    left, right = st.columns([1.7, 1])

    with left:
        st.markdown("### 🕐 Today's Attendance")
        if today_df.empty:
            st.info("No entries recorded today.")
        else:
            disp = today_df[["name", "time"]].head(12).rename(
                columns={"name": "Student", "time": "Time"}
            )
            st.dataframe(disp, use_container_width=True, hide_index=True)

        # Last 5 security events
        if not sec_df.empty:
            st.markdown("### 🔐 Recent Security Events")
            for _, row in sec_df.head(5).iterrows():
                etype = row["event_type"]
                css   = "box-red" if "UNKNOWN" in etype or "FAIL" in etype or "LOITER" in etype else "box-green"
                icon  = {"ENTRY": "✅", "REPEATED_UNKNOWN": "⚠️",
                         "LOITERING": "⏱️", "RAPID_FAIL": "🔴"}.get(etype, "🔔")
                st.markdown(
                    f'<div class="{css}">{icon} <b>{etype}</b> — '
                    f'{row["person"]} '
                    f'<small>conf {row["confidence"]:.2f} | {row["timestamp"]}</small></div>',
                    unsafe_allow_html=True,
                )

    with right:
        st.markdown("### 📸 Latest Recognition")
        kn = _known_captures()
        if kn:
            img = _load_rgb(kn[0])
            if img is not None:
                name_part = os.path.basename(kn[0]).split("_")[0].title()
                st.image(img, caption=f"{name_part} — latest", use_container_width=True)
        else:
            st.info("No student captures yet.")

        unk = _captures("unknown_")
        if unk:
            st.markdown("### 🚨 Latest Unknown")
            img = _load_rgb(unk[0])
            if img is not None:
                ts_part = "_".join(os.path.basename(unk[0]).replace(".jpg","").split("_")[1:])
                st.image(img, caption=f"Unknown · {ts_part}", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE CAMERA
# ═══════════════════════════════════════════════════════════════════════════════
elif "Live Camera" in PAGE:
    st.title("📷 Live Camera Feed")
    st.caption(
        "Each click grabs a fresh frame from the webcam and runs InsightFace recognition. "
        "For continuous detection run `python runtime/camera_runtime.py`."
    )
    st.markdown("---")

    col_main, col_side = st.columns([2.4, 1])

    with col_main:
        snap_btn = st.button("📸 Refresh Frame", use_container_width=True)

        # Persistent frame holder
        frame_placeholder = st.empty()

        if snap_btn or "live_frame" not in st.session_state:
            with st.spinner("Opening camera and running recognition…"):
                from core.face_engine import FaceEngine

                db  = load_all_embeddings()
                eng = FaceEngine()
                cap = cv2.VideoCapture(CAMERA_ID)
                ret, frame = cap.read()
                cap.release()

            if ret and frame is not None:
                faces = eng.process_frame(frame)
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    if db:
                        name, score, _ = find_best_match(face.embedding, db)
                    else:
                        name, score = "No DB", 0.0

                    # Choose colour: green for recognised, red for unknown
                    color = (0, 220, 0) if name != "Unknown" else (0, 0, 220)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label background + text
                    label = f"[{name}]  {score:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(frame,
                                  (x1, y1 - th - 14), (x1 + tw + 8, y1),
                                  color, -1)
                    cv2.putText(frame, label, (x1 + 4, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

                # Clock overlay
                ts_str = datetime.now().strftime("%H:%M:%S")
                cv2.putText(frame, f"Smart Hostel AI  {ts_str}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                st.session_state["live_frame"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.session_state["live_faces"] = len(faces)
            else:
                st.error("❌ Could not capture frame from camera.")
                st.session_state.pop("live_frame", None)

        if "live_frame" in st.session_state:
            frame_placeholder.image(
                st.session_state["live_frame"],
                caption=f"{st.session_state.get('live_faces', 0)} face(s) detected "
                        f"— {datetime.now().strftime('%H:%M:%S')}",
                use_container_width=True,
            )

    with col_side:
        st.markdown("#### ℹ️ Recognition Info")
        st.markdown(
            f'<div class="box-blue">'
            f'Model: <b>InsightFace buffalo_l</b><br>'
            f'Threshold: <b>{SIMILARITY_THRESHOLD}</b><br>'
            f'Camera ID: <b>{CAMERA_ID}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### 👤 Registered Students")
        students = list_all_students()
        if students:
            for s in students:
                st.markdown(
                    f'<div class="box-blue" style="padding:.5rem .8rem;margin:.3rem 0">'
                    f'<b>{s["name"].title()}</b> · {s["room"]} '
                    f'<small>({s["embedding_count"]} embeddings)</small></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No students registered.")

        st.markdown("#### 🖥️ Full Live System")
        st.code("python runtime/camera_runtime.py", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RECOGNIZED STUDENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Recognized Students" in PAGE:
    st.title("👥 Recognized Students")
    st.markdown("---")

    # Attendance table (from SQLite)
    if full_df.empty:
        st.warning("No attendance records yet.")
    else:
        st.markdown(f"### 📋 Attendance Records ({len(full_df)} total)")
        sel_date = st.selectbox(
            "Filter by date",
            ["All"] + sorted(full_df["date"].unique().tolist(), reverse=True),
        )
        filt = full_df if sel_date == "All" else full_df[full_df["date"] == sel_date]
        st.dataframe(
            filt[["name", "date", "time"]].rename(
                columns={"name": "Student", "date": "Date", "time": "Time"}
            ),
            use_container_width=True, hide_index=True,
        )

    # Captures gallery
    st.markdown("---")
    st.markdown("### 📸 Capture Gallery")
    kn = _known_captures()
    if not kn:
        st.info("No student captures yet. Run the live system.")
    else:
        image_grid(kn, cols=4, max_n=16)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — UNKNOWN VISITORS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Unknown Visitors" in PAGE:
    st.title("🚨 Unknown Visitors")
    st.markdown("---")

    unk = _captures("unknown_")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Unknown Captures", len(unk))

    rep_alerts = len(sec_df[sec_df["event_type"] == "REPEATED_UNKNOWN"]) \
        if not sec_df.empty else 0
    c2.metric("Repeated-Unknown Alerts", rep_alerts)

    loit_alerts = len(sec_df[sec_df["event_type"] == "LOITERING"]) \
        if not sec_df.empty else 0
    c3.metric("Loitering Alerts", loit_alerts)

    if not sec_df.empty:
        unk_ev = sec_df[sec_df["event_type"].isin(
            ["REPEATED_UNKNOWN", "LOITERING", "RAPID_FAIL"]
        )]
        if not unk_ev.empty:
            st.markdown("### ⚠️ Threat Events")
            for _, row in unk_ev.head(6).iterrows():
                st.markdown(
                    f'<div class="box-red">🚨 <b>{row["event_type"]}</b> — '
                    f'{row["person"]} | conf {row["confidence"]:.2f} '
                    f'| cam {row["camera_id"]} | {row["timestamp"]}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("### 📷 Captured Images")
    if not unk:
        st.success("No unknown visitors captured. System secure. ✅")
    else:
        image_grid(unk, cols=3, max_n=12)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in PAGE:
    st.title("📊 Attendance Analytics")
    st.markdown("---")

    if full_df.empty:
        st.warning("No attendance data yet.")
        st.stop()

    w_df = _week(full_df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📅 Daily Entries — Last 7 Days")
        daily = (
            w_df.groupby("date").size()
            .reset_index(name="Entries")
            .sort_values("date")
        )
        daily["date"] = pd.to_datetime(daily["date"])
        st.bar_chart(daily.set_index("date")["Entries"])

    with col2:
        st.markdown("#### 👤 Attendance per Student (all time)")
        per_s = (
            full_df.groupby("name").size()
            .reset_index(name="Days")
            .sort_values("Days", ascending=False)
        )
        st.bar_chart(per_s.set_index("name")["Days"])

    st.markdown("---")
    col3, col4 = st.columns([1, 2])

    with col3:
        st.markdown(f"#### ✅ Today ({date.today().isoformat()})")
        st.metric("Students Present",  today_df["name"].nunique() if not today_df.empty else 0)
        st.metric("Total Records",     len(today_df))
        st.metric("Unknown Captures",  len(_captures("unknown_")))
        st.metric("Security Events",   len(sec_df))

    with col4:
        st.markdown("#### 📈 Weekly Attendance Rate")
        total_days = w_df["date"].nunique()
        if total_days:
            rate = (
                w_df.groupby("name").size()
                .reset_index(name="Days Present")
            )
            rate["Attendance %"] = (
                rate["Days Present"] / total_days * 100
            ).round(1)
            rate = rate.sort_values("Attendance %", ascending=False)
            rate.columns = ["Student", "Days Present", "Attendance %"]
            st.dataframe(rate, use_container_width=True, hide_index=True)

    # Unknown visitor trend
    st.markdown("---")
    st.markdown("#### 🚨 Unknown Visitor Count")
    unk_all = _captures("unknown_")
    if unk_all:
        dates = []
        for fp in unk_all:
            parts = os.path.basename(fp).replace(".jpg","").split("_")
            if len(parts) >= 2:
                dates.append(parts[1])  # YYYY-MM-DD
        if dates:
            unk_s = pd.Series(dates).value_counts().sort_index()
            unk_s.index = pd.to_datetime(unk_s.index)
            st.bar_chart(unk_s.rename("Unknown Captures"))
    else:
        st.info("No unknown captures to chart.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DAILY LOGS (Security Events)
# ═══════════════════════════════════════════════════════════════════════════════
elif "Daily Logs" in PAGE:
    st.title("📋 Security Event Logs")
    st.caption("Sourced from `security_events` table in `data/attendance.db`")
    st.markdown("---")

    if sec_df.empty:
        st.info("No security events yet. Run `python runtime/camera_runtime.py`.")
    else:
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            types    = ["All"] + sorted(sec_df["event_type"].unique().tolist())
            sel_type = st.selectbox("Event Type", types)
        with col2:
            persons    = ["All"] + sorted(sec_df["person"].unique().tolist())
            sel_person = st.selectbox("Person", persons)
        with col3:
            cams    = ["All"] + sorted(sec_df["camera_id"].astype(str).unique().tolist())
            sel_cam = st.selectbox("Camera ID", cams)

        filt = sec_df.copy()
        if sel_type   != "All": filt = filt[filt["event_type"] == sel_type]
        if sel_person != "All": filt = filt[filt["person"]     == sel_person]
        if sel_cam    != "All": filt = filt[filt["camera_id"].astype(str) == sel_cam]

        st.markdown(f"**{len(filt)} event(s)**")
        display_cols = ["timestamp", "event_type", "person", "confidence", "camera_id"]
        st.dataframe(
            filt[display_cols].rename(columns={
                "timestamp":  "Timestamp",
                "event_type": "Event",
                "person":     "Person",
                "confidence": "Confidence",
                "camera_id":  "Camera",
            }),
            use_container_width=True, hide_index=True,
        )

        # CSV download
        st.download_button(
            "⬇️ Download CSV",
            filt[display_cols].to_csv(index=False).encode(),
            "security_events.csv", "text/csv",
        )

        # Event type breakdown
        if len(sec_df) >= 2:
            st.markdown("---")
            st.markdown("#### Event Breakdown")
            breakdown = (
                sec_df["event_type"].value_counts()
                .reset_index()
            )
            breakdown.columns = ["Event Type", "Count"]
            st.bar_chart(breakdown.set_index("Event Type"))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ADMIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
elif "Admin Panel" in PAGE:
    st.title("⚙️ Admin Panel")
    st.markdown("---")

    tab_list, tab_add, tab_del = st.tabs(
        ["👤 Student List", "➕ Register Student", "🗑️ Remove Student"]
    )

    # ── Tab 1: Student list ───────────────────────────────────────────────────
    with tab_list:
        students = list_all_students()
        if students:
            df_s = pd.DataFrame(students)
            df_s.columns = ["Name", "Room", "Embeddings"]
            df_s["Name"] = df_s["Name"].str.title()
            st.dataframe(df_s, use_container_width=True, hide_index=True)
            st.metric("Total Registered", len(students))
        else:
            st.info("No students registered yet.")

    # ── Tab 2: Register ───────────────────────────────────────────────────────
    with tab_add:
        st.markdown(
            "Opens an OpenCV window and automatically captures **20 face samples**."
        )
        st.markdown(
            '<div class="box-blue">Move your face around slightly during capture for '
            'better angle coverage and higher accuracy.</div>',
            unsafe_allow_html=True,
        )
        with st.form("register_form"):
            r_name = st.text_input("Full Name", placeholder="e.g. Naman Sharma")
            r_room = st.text_input("Room Number", placeholder="e.g. A-203")
            r_go   = st.form_submit_button("🎥 Start Capture")

        if r_go:
            if not r_name.strip():
                st.error("Name is required.")
            else:
                st.info(f"Opening webcam for '{r_name}'…")
                with st.spinner("Capturing 20 samples — follow the on-screen guide…"):
                    ok = register_student(r_name.strip(), r_room.strip())
                if ok:
                    st.success(f"✅ '{r_name}' registered with 20 embeddings!")
                    st.cache_data.clear()
                else:
                    st.error("Registration cancelled or failed.")

    # ── Tab 3: Remove ─────────────────────────────────────────────────────────
    with tab_del:
        students = list_all_students()
        if not students:
            st.info("No students to remove.")
        else:
            names  = [s["name"].title() for s in students]
            to_del = st.selectbox("Select student to remove", names)
            st.warning(
                f"This permanently deletes **{to_del}'s** embedding file and "
                "all attendance records."
            )
            if st.button("🗑️ Confirm Delete"):
                try:
                    delete_student(to_del)
                    delete_student_records(to_del)
                    st.success(f"✅ '{to_del}' removed successfully.")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
