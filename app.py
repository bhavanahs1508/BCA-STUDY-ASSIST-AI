import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date

import streamlit as st
from google import genai
from google.genai import types
from streamlit_mic_recorder import speech_to_text


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BCA Study Assist",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "study_assist.db"
MODEL = "gemini-3.5-flash-lite"

BCA_SCOPE = """
You are BCA Study Assist, a friendly AI tutor specialized in BCA
(Bachelor of Computer Applications) subjects.

Help with Python, C, Java, DBMS, SQL, data structures and algorithms,
operating systems, computer networks, software engineering, web development,
mathematics, and other BCA subjects.

Adapt your answer to the student's request:
- Simple question: answer clearly and briefly.
- Deep or broad question: explain thoroughly with examples, applications,
  and important related concepts.
- Step-by-step request: use numbered steps and explain why they work.
- Exam answer: use an exam-ready structure with definitions, headings,
  key points, examples, and a conclusion when useful.
- Summary request: provide concise, easy-to-revise notes.
- Code question: provide working code, explain it, and show expected output.
- Timetable/study-plan request: provide a realistic Markdown table with
  Day/Date, Time, Subject, Task, Goal, and Status columns. Include breaks,
  practice, and revision. Respect the student's available time.
- Lecture transcript request: organize the transcript into notes,
  definitions, examples, and a revision summary without inventing facts.

Use beginner-friendly language. Do not make every answer long.
Follow the student's requested format and depth. Stay focused on BCA and
general learning. Never invent facts when uncertain.
"""

NOTES_INSTRUCTION = """
Turn the supplied lecture transcript into well-organized BCA study notes.
Include headings, key definitions, important concepts, examples, and a short
revision summary. Do not add unsupported information.
"""


# =========================================================
# SQLITE DATABASE
# Existing chats/messages are preserved; no database deletion.
# =========================================================

@contextmanager
def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def table_columns(con, table_name):
    return {row["name"] for row in con.execute(f"PRAGMA table_info({table_name})")}


def init_db():
    with db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        cols = table_columns(con, "chats")
        if "title" not in cols:
            con.execute("ALTER TABLE chats ADD COLUMN title TEXT")
        if "created_at" not in cols:
            con.execute("ALTER TABLE chats ADD COLUMN created_at TEXT")
        if "updated_at" not in cols:
            con.execute("ALTER TABLE chats ADD COLUMN updated_at TEXT")

        now = datetime.now().isoformat(timespec="seconds")
        con.execute("UPDATE chats SET title = 'Previous chat' WHERE title IS NULL")
        con.execute("UPDATE chats SET created_at = ? WHERE created_at IS NULL", (now,))
        con.execute("UPDATE chats SET updated_at = created_at WHERE updated_at IS NULL")

        con.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(id)
            )
        """)

        # Migrate older message tables if needed.
        msg_cols = table_columns(con, "messages")
        if "chat_id" not in msg_cols:
            con.execute("ALTER TABLE messages ADD COLUMN chat_id INTEGER")
        if "role" not in msg_cols:
            con.execute("ALTER TABLE messages ADD COLUMN role TEXT")
        if "content" not in msg_cols:
            con.execute("ALTER TABLE messages ADD COLUMN content TEXT")
        if "created_at" not in msg_cols:
            con.execute("ALTER TABLE messages ADD COLUMN created_at TEXT")
        con.execute(
            "UPDATE messages SET created_at = ? WHERE created_at IS NULL",
            (now,),
        )

        con.execute("""
            CREATE TABLE IF NOT EXISTS study_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                subjects TEXT NOT NULL,
                plan TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        plan_cols = table_columns(con, "study_plans")
        if "chat_id" not in plan_cols:
            con.execute("ALTER TABLE study_plans ADD COLUMN chat_id INTEGER")

        con.execute("""
            CREATE TABLE IF NOT EXISTS timetable_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                day TEXT,
                time_slot TEXT,
                subject TEXT,
                task TEXT NOT NULL,
                goal TEXT,
                status INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(id)
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS lecture_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                transcript TEXT NOT NULL,
                notes TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        con.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_tasks_chat ON timetable_tasks(chat_id, id)")


def create_chat():
    now = datetime.now().isoformat(timespec="seconds")
    with db() as con:
        cur = con.execute(
            "INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
            ("New chat", now, now),
        )
        return cur.lastrowid


def list_chats(search_text=""):
    with db() as con:
        if search_text.strip():
            return con.execute(
                """SELECT id, title, updated_at FROM chats
                   WHERE title LIKE ? ORDER BY updated_at DESC, id DESC""",
                (f"%{search_text.strip()}%",),
            ).fetchall()
        return con.execute(
            "SELECT id, title, updated_at FROM chats ORDER BY updated_at DESC, id DESC"
        ).fetchall()


def get_messages(chat_id):
    with db() as con:
        return con.execute(
            "SELECT id, role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()


def save_message(chat_id, role, content):
    now = datetime.now().isoformat(timespec="seconds")
    clean_title = " ".join(content.split())
    with db() as con:
        con.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, now),
        )
        if role == "user":
            row = con.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if row and (row["title"] or "") in ("", "New chat", "Previous chat"):
                title = clean_title[:42] + ("..." if len(clean_title) > 42 else "")
                con.execute("UPDATE chats SET title = ? WHERE id = ?", (title or "New chat", chat_id))
        con.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))


def delete_chat(chat_id):
    with db() as con:
        con.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        con.execute("DELETE FROM timetable_tasks WHERE chat_id = ?", (chat_id,))
        con.execute("DELETE FROM study_plans WHERE chat_id = ?", (chat_id,))
        con.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def save_plan(chat_id, subjects, plan):
    with db() as con:
        con.execute(
            "INSERT INTO study_plans (chat_id, subjects, plan, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, subjects, plan, datetime.now().isoformat(timespec="seconds")),
        )


def parse_markdown_table(text):
    """Extract rows from a Markdown table that looks like a timetable."""
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return []
    split_rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells):
            continue
        split_rows.append(cells)
    if len(split_rows) < 2:
        return []
    headers = [h.lower() for h in split_rows[0]]
    if not any("task" in h or "activity" in h for h in headers):
        return []
    def find_index(words):
        for i, h in enumerate(headers):
            if any(word in h for word in words):
                return i
        return None
    day_i = find_index(("day", "date"))
    time_i = find_index(("time",))
    subject_i = find_index(("subject", "topic"))
    task_i = find_index(("task", "activity"))
    goal_i = find_index(("goal", "target", "outcome"))
    tasks = []
    for cells in split_rows[1:]:
        if task_i is None or task_i >= len(cells):
            continue
        task = cells[task_i].strip()
        if not task or task.lower() in ("-", "none", "n/a"):
            continue
        def cell_at(idx):
            return cells[idx] if idx is not None and idx < len(cells) else ""
        tasks.append({
            "day": cell_at(day_i),
            "time_slot": cell_at(time_i),
            "subject": cell_at(subject_i),
            "task": task,
            "goal": cell_at(goal_i),
        })
    return tasks


def replace_chat_tasks(chat_id, tasks):
    now = datetime.now().isoformat(timespec="seconds")
    with db() as con:
        con.execute("DELETE FROM timetable_tasks WHERE chat_id = ?", (chat_id,))
        for item in tasks:
            con.execute(
                """INSERT INTO timetable_tasks
                   (chat_id, day, time_slot, subject, task, goal, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
                (chat_id, item["day"], item["time_slot"], item["subject"],
                 item["task"], item["goal"], now),
            )


def get_tasks(chat_id):
    with db() as con:
        return con.execute(
            "SELECT * FROM timetable_tasks WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()


def set_task_status(task_id, status):
    with db() as con:
        con.execute("UPDATE timetable_tasks SET status = ? WHERE id = ?", (int(status), task_id))


init_db()


# =========================================================
# GEMINI
# =========================================================

def get_gemini_client():
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to .streamlit/secrets.toml. "
            "Never share the key or upload that file to GitHub."
        )
    return genai.Client(api_key=api_key)


def generate_text(prompt, history=None, extra_instruction=""):
    client = get_gemini_client()
    conversation = []
    for item in (history or [])[-12:]:
        role = "user" if item["role"] == "user" else "model"
        conversation.append(types.Content(role=role, parts=[types.Part.from_text(text=item["content"])]))
    conversation.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    response = client.models.generate_content(
        model=MODEL,
        contents=conversation,
        config=types.GenerateContentConfig(
            system_instruction=BCA_SCOPE + ("\n\n" + extra_instruction if extra_instruction else ""),
            temperature=0.4,
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response. Please try again.")
    return response.text


def looks_like_timetable(prompt):
    p = prompt.lower()
    return any(word in p for word in (
        "timetable", "study plan", "study schedule", "revision plan",
        "study planner", "time table", "schedule my study",
    ))


def looks_like_notes(prompt):
    p = prompt.lower()
    return any(word in p for word in (
        "summarize this lecture", "summarise this lecture", "make notes from",
        "lecture notes", "summarize this transcript", "summarise this transcript",
    ))


# =========================================================
# THEME + SESSION STATE
# =========================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "chat_id" not in st.session_state:
    st.session_state.chat_id = create_chat()
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if "last_mic_text" not in st.session_state:
    st.session_state.last_mic_text = ""


def apply_theme(theme):
    dark = theme == "Dark"
    bg = "#0f1117" if dark else "#ffffff"
    panel = "#191c24" if dark else "#f5f7fb"
    fg = "#edf0f7" if dark else "#202534"
    muted = "#a8b0c0" if dark else "#667085"
    border = "#303543" if dark else "#e3e7ef"
    accent = "#8b9cff" if dark else "#4f63d9"
    st.markdown(
        f"""
        <style>
        :root {{ color-scheme: {"dark" if dark else "light"}; }}
        .stApp {{ background: {bg}; color: {fg}; }}
        [data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {border}; }}
        [data-testid="stSidebar"] * {{ color: {fg}; }}
        [data-testid="stHeader"] {{ background: {bg}; }}
        [data-testid="stChatMessage"] {{
            border: 1px solid {border}; border-radius: 16px;
            padding: 12px 16px; margin-bottom: 12px;
        }}
        .main-title {{ font-size: 2rem; font-weight: 750; letter-spacing: -0.7px; color: {fg}; }}
        .subtitle {{ color: {muted}; font-size: 1rem; margin-bottom: 20px; }}
        .welcome-card {{
            background: {panel}; border: 1px solid {border};
            border-radius: 18px; padding: 22px; margin: 12px 0 18px 0;
        }}
        .welcome-card h3 {{ color: {fg}; margin-top: 0; }}
        .welcome-card p {{ color: {muted}; margin-bottom: 0; }}
        div.stButton > button {{ border-radius: 10px; min-height: 40px; }}
        div.stButton > button:hover {{ border-color: {accent}; }}
        hr {{ border-color: {border}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme(st.session_state.theme)


# =========================================================
# SIDEBAR: HISTORY + APPEARANCE
# =========================================================

with st.sidebar:
    st.markdown("## 🎓 BCA Study Assist")
    st.caption("Your personal BCA AI study companion")
    st.divider()

    selected_theme = st.radio(
        "Appearance", ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
        key="appearance_choice",
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    if st.button("＋ New chat", use_container_width=True):
        st.session_state.chat_id = create_chat()
        st.session_state.pending_question = ""
        st.session_state.last_mic_text = ""
        st.rerun()

    st.divider()
    st.markdown("### 💬 Chat history")
    history_search = st.text_input("Search conversations", placeholder="Search chat titles...")
    chats = list_chats(history_search)
    if not chats:
        st.caption("No matching conversations.")
    else:
        for chat in chats:
            title = (chat["title"] or "Previous chat").strip()
            if len(title) > 32:
                title = title[:32] + "..."
            label = ("● " if chat["id"] == st.session_state.chat_id else "") + title
            if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
                st.session_state.chat_id = chat["id"]
                st.rerun()

    st.divider()
    with st.expander("Manage current chat"):
        confirm = st.checkbox(
            "I want to delete this conversation",
            key="confirm_delete",
        )

        if st.button(
            "Delete current chat",
            disabled=not confirm,
            use_container_width=True,
        ):
            delete_chat(st.session_state.chat_id)
            st.session_state.chat_id = create_chat()
            st.rerun()


    st.caption("Gemini AI · SQLite storage")


# =========================================================
# MAIN CHAT
# =========================================================

st.markdown('<div class="main-title">BCA Study Assist</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask, learn, revise, plan, and understand — all in one chat.</div>',
    unsafe_allow_html=True,
)

chat_id = st.session_state.chat_id
history = get_messages(chat_id)

if not history:
    st.markdown(
        """
        <div class="welcome-card">
          <h3>👋 What would you like to learn today?</h3>
          <p>Ask a BCA question, request an exam answer, get a summary,
          make a timetable, or record a lecture live.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    examples = [
        "Explain DBMS normalization with examples",
        "Teach me Python loops step by step",
        "Give me an exam-ready answer about operating systems",
        "Create a 7-day BCA study timetable for 2 hours per day",
    ]
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()

for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Live browser speech recognition control. The component may depend on browser
# speech-recognition support and internet access. Its transcript is inserted
# into the normal chat flow below.
with st.expander("🎙️ Record a lecture / speak your question", expanded=False):
    st.caption(
        "Allow microphone access when your browser asks. Speak clearly, then stop "
        "recording. Recognition availability depends on your browser and connection."
    )
    mic_text = speech_to_text(
        language="en",
        start_prompt="Start microphone",
        stop_prompt="Stop recording",
        just_once=True,
        key="live_lecture_mic",
    )
    if mic_text and mic_text != st.session_state.last_mic_text:
        st.session_state.last_mic_text = mic_text
        st.session_state.pending_question = (
            "Please organize this live lecture transcript into clear BCA study notes "
            "with headings, definitions, examples, and a short revision summary:\n\n"
            + mic_text
        )
        st.rerun()

prompt = st.chat_input("Ask a BCA question, request a summary, exam answer, or timetable...")

if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = ""

if prompt:
    save_message(chat_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    history_for_ai = get_messages(chat_id)[:-1]
    extra = ""
    if looks_like_timetable(prompt):
        extra = """
Create a goal-based student timetable in a Markdown table with these exact
columns: Day/Date | Time | Subject | Task | Goal | Status.
Use a separate row for each task. Use short, specific goals and put '☐ Not done'
in the Status column. Include breaks, practice, and revision. Make the schedule
realistic and beginner-friendly.
"""
    elif looks_like_notes(prompt):
        extra = NOTES_INSTRUCTION

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = generate_text(prompt, history_for_ai, extra)
                st.markdown(answer)
                save_message(chat_id, "assistant", answer)

                if looks_like_timetable(prompt):
                    tasks = parse_markdown_table(answer)
                    if tasks:
                        replace_chat_tasks(chat_id, tasks)
                        save_plan(chat_id, prompt, answer)
                    else:
                        st.info("The timetable is shown above. You can ask me to reformat it as a table if needed.")

            except Exception as error:
                st.error("I couldn't generate an answer. Check your internet connection and Gemini API key.")
                st.caption(f"Technical details: {error}")


# =========================================================
# TIMETABLE PROGRESS TRACKER
# =========================================================

tasks = get_tasks(st.session_state.chat_id)
if tasks:
    st.divider()
    st.subheader("📅 Timetable goals & progress")
    st.caption("Tick tasks as you complete them. Your progress is saved in SQLite.")
    completed = sum(1 for task in tasks if task["status"])
    st.progress(completed / len(tasks))
    st.write(f"**{completed} of {len(tasks)} tasks completed**")

    for task in tasks:
        label_parts = []
        if task["day"]:
            label_parts.append(task["day"])
        if task["time_slot"]:
            label_parts.append(task["time_slot"])
        if task["subject"]:
            label_parts.append(task["subject"])
        label_parts.append(task["task"])
        if task["goal"]:
            label_parts.append(f"Goal: {task['goal']}")
        label = " · ".join(label_parts)
        checked = st.checkbox(
            label,
            value=bool(task["status"]),
            key=f"task_done_{task['id']}",
        )
        if checked != bool(task["status"]):
            set_task_status(task["id"], checked)
            st.rerun()

st.divider()
st.caption(
    "BCA Study Assist · Streamlit + Gemini AI + SQLite. "
    "Verify important facts and exam answers with your course materials."
)
