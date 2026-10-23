import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from streamlit_mic_recorder import speech_to_text


# =========================================================
# CONFIGURATION (unchanged)
# =========================================================

st.set_page_config(
    page_title="BCA Study Assist",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "study_assist.db"
MODEL = "gemini-3.5-flash-lite"
MAX_MESSAGE_CHARS = 12000

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

TABLE_INSTRUCTION = """
Create a personalized, realistic student timetable in a Markdown table with these
columns: Day/Date | Time | Subject | Task | Goal | Status. Ask a short follow-up
question first if essential information is missing. Include breaks, practice, and
revision. Put '☐ Not done' in the Status column.
"""

TOOL_STEP = "Step-by-step problem solving"
TOOL_SCHEDULE = "Personalized study schedule"
TOOL_NOTES = "Voice transcription → notes"
TOOLS = [TOOL_STEP, TOOL_SCHEDULE, TOOL_NOTES]
THEMES = ["Light", "Dark", "System"]


# =========================================================
# SQLITE DATABASE
# Existing chats/messages are preserved; nothing is deleted or recreated.
# The only addition is a small `app_settings` table (CREATE IF NOT EXISTS)
# used to remember the appearance choice.
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

        # New, additive: tiny key/value table for the appearance preference.
        con.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        con.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_tasks_chat ON timetable_tasks(chat_id, id)")


def get_setting(key, default=""):
    try:
        with db() as con:
            row = con.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row and row["value"] is not None else default
    except Exception:
        return default


def set_setting(key, value):
    try:
        with db() as con:
            con.execute(
                "INSERT INTO app_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
    except Exception:
        pass  # Preference persistence is best-effort.


def create_chat():
    now = datetime.now().isoformat(timespec="seconds")
    with db() as con:
        cur = con.execute(
            "INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
            ("New chat", now, now),
        )
        return cur.lastrowid


def list_chats():
    """Saved conversations (those that have at least one message), newest first."""
    with db() as con:
        return con.execute(
            """SELECT c.id, c.title, c.updated_at FROM chats c
               WHERE EXISTS (SELECT 1 FROM messages m WHERE m.chat_id = c.id)
               ORDER BY c.updated_at DESC, c.id DESC"""
        ).fetchall()


def _like_pattern(text):
    t = text.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{t}%"


def search_chats(query, limit=40):
    """Search by title and by message contents. Returns rows with a `hit` snippet source."""
    if not query.strip():
        return []
    pat = _like_pattern(query)
    with db() as con:
        return con.execute(
            """SELECT c.id, c.title, c.updated_at,
                      (SELECT m.content FROM messages m
                        WHERE m.chat_id = c.id AND m.content LIKE ? ESCAPE '\\'
                        ORDER BY m.id LIMIT 1) AS hit
               FROM chats c
               WHERE EXISTS (SELECT 1 FROM messages m0 WHERE m0.chat_id = c.id)
                 AND (c.title LIKE ? ESCAPE '\\'
                      OR EXISTS (SELECT 1 FROM messages m
                                  WHERE m.chat_id = c.id AND m.content LIKE ? ESCAPE '\\'))
               ORDER BY c.updated_at DESC, c.id DESC LIMIT ?""",
            (pat, pat, pat, limit),
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
# HISTORY GROUPING + SEARCH HELPERS
# =========================================================

GROUP_ORDER = ["Today", "Yesterday", "Previous 7 Days", "Previous 30 Days", "Older"]


def chat_group(updated_at):
    try:
        d = datetime.fromisoformat(updated_at).date()
    except Exception:
        return "Older"
    days = (date.today() - d).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days <= 7:
        return "Previous 7 Days"
    if days <= 30:
        return "Previous 30 Days"
    return "Older"


def group_chats(rows):
    groups = {name: [] for name in GROUP_ORDER}
    for row in rows:
        groups[chat_group(row["updated_at"])].append(row)
    return [(name, groups[name]) for name in GROUP_ORDER if groups[name]]


def short_title(title, limit=28):
    title = " ".join((title or "Previous chat").split())
    return title if len(title) <= limit else title[:limit].rstrip() + "..."


def make_snippet(text, query, width=70):
    flat = " ".join((text or "").split())
    idx = flat.lower().find(query.strip().lower())
    if idx < 0:
        return flat[:width * 2]
    start = max(0, idx - width // 2)
    end = min(len(flat), idx + len(query) + width)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")


# =========================================================
# GEMINI (SDK, model and secrets loading unchanged)
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


def friendly_error(error):
    text = str(error)
    low = text.lower()
    if "gemini_api_key is missing" in low:
        return "The Gemini API key is missing. Add GEMINI_API_KEY to .streamlit/secrets.toml."
    if any(k in low for k in ("api key not valid", "api_key_invalid", "permission_denied", "401", "403")):
        return "Gemini rejected the API key. Please check GEMINI_API_KEY in your secrets."
    if any(k in low for k in ("429", "quota", "resource_exhausted", "rate limit")):
        return "Gemini is rate-limited or the quota is used up. Please wait a minute and try again."
    if any(k in low for k in ("timeout", "timed out", "connection", "network", "name resolution",
                              "unreachable", "getaddrinfo")):
        return "Couldn't reach Gemini. Please check your internet connection and try again."
    if "empty response" in low:
        return "Gemini returned an empty response. Please try again."
    return "Something went wrong while generating the answer. Your message was not lost; please try again."


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


def build_request(tool, prompt):
    """Return (prompt_for_ai, extra_instruction, wants_timetable) for the selected tool."""
    if tool == TOOL_STEP:
        prompt_for_ai = (
            "Solve this BCA study question step by step. Explain each step in "
            "beginner-friendly language, show working/code and expected output "
            "when relevant, and stay focused on the problem.\n\n" + prompt
        )
        extra = (
            "Focus only on step-by-step problem solving for BCA subjects. "
            "Use numbered steps, explain why each step works, and include a "
            "small example when useful."
        )
    elif tool == TOOL_SCHEDULE:
        prompt_for_ai = (
            "Create or refine a personalized study schedule based on this request. "
            "If essential details such as available hours, subjects, or deadline "
            "are missing, ask a short follow-up question instead of guessing.\n\n" + prompt
        )
        extra = (
            "Generate a realistic personalized study schedule. Use a Markdown "
            "table with Day/Date, Time, Subject, Task, Goal, and Status. Include "
            "breaks, practice, and revision. Use '☐ Not done' for Status."
        )
    else:
        prompt_for_ai = (
            "Turn the following spoken/transcribed content into clear BCA study "
            "notes. Preserve the meaning; organize it with headings, definitions, "
            "key points, examples only when supported, and a short revision summary. "
            "Do not invent missing facts. If the text is not lecture/study content, "
            "ask the user to record or provide study-related content.\n\n" + prompt
        )
        extra = NOTES_INSTRUCTION

    wants_timetable = tool == TOOL_SCHEDULE or looks_like_timetable(prompt)
    if wants_timetable:
        extra += "\n" + TABLE_INSTRUCTION
    elif tool != TOOL_NOTES and looks_like_notes(prompt):
        extra += "\n\n" + NOTES_INSTRUCTION
    return prompt_for_ai, extra, wants_timetable


# =========================================================
# SESSION STATE (all keys initialised once, up front)
# =========================================================

if "theme" not in st.session_state:
    saved_theme = get_setting("theme", "System")
    st.session_state.theme = saved_theme if saved_theme in THEMES else "System"
if "appearance_choice" not in st.session_state:
    st.session_state.appearance_choice = st.session_state.theme
if "chat_id" not in st.session_state:
    # None = a fresh, unsaved chat. A row is only created when the first
    # message is successfully answered, so no empty chats pile up.
    st.session_state.chat_id = None
if "delete_target" not in st.session_state:
    st.session_state.delete_target = None
if "show_search" not in st.session_state:
    st.session_state.show_search = False
if "composer_text" not in st.session_state:
    st.session_state.composer_text = ""
if "last_mic_text" not in st.session_state:
    st.session_state.last_mic_text = ""
if "selected_study_tool" not in st.session_state:
    st.session_state.selected_study_tool = TOOL_STEP


# =========================================================
# THEME
# =========================================================

LIGHT = {
    "app-bg": "#ffffff", "panel-bg": "#f7f7f8", "surface": "#ffffff",
    "text": "#111111", "muted": "#6b6b6b", "border": "#e3e3e3",
    "hover": "#ececec", "button-bg": "#111111", "button-text": "#ffffff",
}
DARK = {
    "app-bg": "#0c0c0c", "panel-bg": "#131313", "surface": "#1b1b1b",
    "text": "#f4f4f4", "muted": "#a8a8a8", "border": "#303030",
    "hover": "#262626", "button-bg": "#f2f2f2", "button-text": "#0c0c0c",
}


def _vars(palette):
    return " ".join(f"--{k}: {v};" for k, v in palette.items())


CSS_TEMPLATE = """
<style>
/*ROOT*/

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], [data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background: var(--app-bg) !important;
    color: var(--text) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Hide Streamlit chrome (the header stays so the sidebar toggle keeps working) */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stAppDeployButton"], .stDeployButton { display: none !important; visibility: hidden !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="InputInstructions"] { display: none !important; }

.block-container { max-width: 820px; padding-top: 1.5rem; padding-bottom: 1.5rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--panel-bg) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stSidebar"] .brand { font-weight: 700; font-size: 1.05rem; margin: .2rem 0 .8rem; }
[data-testid="stSidebar"] .group-label {
    color: var(--muted) !important; font-size: .74rem; font-weight: 600;
    margin: 1rem 0 .25rem .3rem; letter-spacing: .02em;
}
[data-testid="stSidebar"] div.stButton > button {
    border: none; background: transparent; box-shadow: none;
    justify-content: flex-start; text-align: left; border-radius: 10px;
    min-height: 38px; padding: .3rem .6rem; font-weight: 400;
}
[data-testid="stSidebar"] div.stButton > button:hover { background: var(--hover); }
[data-testid="stSidebar"] button[kind="primary"] {
    background: var(--button-bg) !important; color: var(--button-text) !important; font-weight: 600;
}
[data-testid="stSidebar"] button[kind="primary"] * { color: var(--button-text) !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 12px; }

/* Greeting / empty state */
.greeting-wrap { text-align: center; margin: 14vh 0 4vh; }
.greeting { font-size: 2rem; font-weight: 650; letter-spacing: -0.5px; color: var(--text); }
.greeting-sub { color: var(--muted); margin-top: .5rem; font-size: .98rem; }

/* Messages */
[data-testid="stChatMessage"] {
    background: transparent; border: none; padding: .5rem .2rem; margin-bottom: .4rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--hover); border-radius: 18px; padding: .7rem 1rem;
}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"] { color: var(--text); }
[data-testid="stCaptionContainer"] { color: var(--muted); }
pre, code { border-radius: 10px; }

/* Generic inputs and buttons */
input, textarea, [data-baseweb="select"] > div { border-color: var(--border) !important; }
[data-testid="stTextInput"] input, [data-baseweb="input"] input, textarea {
    background-color: var(--surface) !important; color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}
[data-baseweb="input"] > div, [data-baseweb="textarea"] > div {
    background: var(--surface) !important; border-color: var(--border) !important;
}
div.stButton > button {
    border: 1px solid var(--border); border-radius: 12px; min-height: 40px;
    background: var(--surface); color: var(--text);
}
div.stButton > button:hover { border-color: var(--text); background: var(--hover); color: var(--text); }
div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label { color: var(--text) !important; }
[data-testid="stDialog"] > div, div[role="dialog"] { background: var(--app-bg) !important; color: var(--text) !important; }
hr { border-color: var(--border) !important; }

/* ChatGPT-style composer: one rounded pill, pinned to the bottom */
[data-testid="stHorizontalBlock"]:has(.composer-anchor) {
    position: sticky; bottom: 12px; z-index: 50;
    background: var(--surface); border: 1px solid var(--border); border-radius: 30px;
    padding: 6px 8px 6px 16px; gap: .25rem !important;
    align-items: flex-end; flex-wrap: nowrap !important; margin-top: 1rem;
    box-shadow: 0 0 0 14px var(--app-bg);
}
[data-testid="stHorizontalBlock"]:has(.composer-anchor) > div { min-width: 0 !important; }
[data-testid="stHorizontalBlock"]:has(.composer-anchor) > div:first-child { flex: 1 1 auto !important; }
[data-testid="stHorizontalBlock"]:has(.composer-anchor) > div:last-child {
    flex: 0 0 auto !important; width: auto !important; min-width: 48px !important;
}
.element-container:has(.composer-anchor) { display: none; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
[data-testid="stForm"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; align-items: flex-end; gap: .4rem !important; }
[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div { min-width: 0 !important; }
[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:first-child { flex: 1 1 auto !important; }
[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:last-child {
    flex: 0 0 44px !important; width: 44px !important; min-width: 44px !important;
}
[data-testid="stForm"] textarea { border: none !important; box-shadow: none !important; background: transparent !important; }
[data-testid="stForm"] [data-baseweb="textarea"], [data-testid="stForm"] [data-baseweb="textarea"] > div,
[data-testid="stForm"] [data-baseweb="base-input"] {
    border: none !important; background: transparent !important; box-shadow: none !important;
}
[data-testid="stFormSubmitButton"] button {
    width: 40px; height: 40px; min-height: 40px; padding: 0; border-radius: 50% !important;
    font-size: 1.25rem; line-height: 1; justify-content: center;
    background: var(--button-bg) !important; color: var(--button-text) !important;
    border: none !important;
}
[data-testid="stFormSubmitButton"] button * { color: var(--button-text) !important; }

@media (max-width: 700px) {
    .block-container { padding: 1rem .8rem 1rem; }
    .greeting { font-size: 1.6rem; }
    .greeting-wrap { margin-top: 8vh; }
    [data-testid="stHorizontalBlock"]:has(.composer-anchor) { padding-left: 12px; }
}
</style>
"""


def apply_theme(theme):
    """Light/Dark are explicit. System follows the browser/OS via prefers-color-scheme."""
    base = DARK if theme == "Dark" else LIGHT
    root = f":root {{ color-scheme: {'dark' if theme == 'Dark' else 'light'}; {_vars(base)} }}"
    if theme == "System":
        root += f" @media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark; {_vars(DARK)} }} }}"
    st.markdown(CSS_TEMPLATE.replace("/*ROOT*/", root), unsafe_allow_html=True)


# Enter sends, Shift+Enter inserts a newline (desktop only). Works by turning a plain
# Enter into the Ctrl+Enter that Streamlit already uses to submit a form.
ENTER_TO_SEND_JS = """
<script>
(function () {
  const win = window.parent;
  if (!win || win.__bcaEnterBound) return;
  win.__bcaEnterBound = true;
  win.document.addEventListener('keydown', function (e) {
    const t = e.target;
    if (!t || t.tagName !== 'TEXTAREA' || !t.closest('[data-testid="stForm"]')) return;
    if (win.matchMedia && win.matchMedia('(pointer: coarse)').matches) return;
    if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.isComposing) {
      e.preventDefault();
      e.stopPropagation();
      t.dispatchEvent(new KeyboardEvent('keydown', {
        key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
        ctrlKey: true, bubbles: true, cancelable: true
      }));
    }
  }, true);
})();
</script>
"""

apply_theme(st.session_state.theme)


# =========================================================
# CALLBACKS
# =========================================================

def open_chat(chat_id):
    st.session_state.chat_id = chat_id
    st.session_state.delete_target = None
    st.session_state.show_search = False


def start_new_chat():
    # Does not touch the database: the old chat stays saved, the new one is
    # created only when its first message is answered.
    st.session_state.chat_id = None
    st.session_state.delete_target = None
    st.session_state.show_search = False


def ask_delete(chat_id):
    st.session_state.delete_target = chat_id


def cancel_delete():
    st.session_state.delete_target = None


def confirm_delete():
    target = st.session_state.delete_target
    if target is not None:
        delete_chat(target)
        if st.session_state.chat_id == target:
            st.session_state.chat_id = None
    st.session_state.delete_target = None


def on_theme_change():
    choice = st.session_state.appearance_choice
    if choice in THEMES:
        st.session_state.theme = choice
        set_setting("theme", choice)


def open_search_panel():
    st.session_state.show_search = True


def close_search_panel():
    st.session_state.show_search = False


# =========================================================
# SEARCH POPUP (st.dialog if available, otherwise an in-page panel)
# =========================================================

def render_search_body():
    query = st.text_input(
        "Search chats",
        placeholder="Search chats by title or message...",
        label_visibility="collapsed",
        key="search_query",
    )
    if query.strip():
        rows = search_chats(query)
        if not rows:
            st.caption("No conversations found.")
    else:
        rows = list_chats()[:15]
        st.caption("Recent conversations" if rows else "No conversations yet.")

    for row in rows:
        if st.button(short_title(row["title"], 46), key=f"sr_{row['id']}", use_container_width=True):
            st.session_state.chat_id = row["id"]
            st.session_state.delete_target = None
            st.session_state.show_search = False
            st.rerun()
        if query.strip() and "hit" in row.keys() and row["hit"]:
            st.caption(make_snippet(row["hit"], query))


_dialog_fn = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
search_dialog = _dialog_fn("Search chats")(render_search_body) if _dialog_fn else None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown('<div class="brand">🎓 BCA Study Assistant</div>', unsafe_allow_html=True)

    st.button("＋  New chat", key="new_chat_btn", on_click=start_new_chat,
              use_container_width=True, type="primary")

    if search_dialog:
        if st.button("🔍  Search chats", key="open_search_btn", use_container_width=True):
            search_dialog()
    else:
        st.button("🔍  Search chats", key="open_search_btn", on_click=open_search_panel,
                  use_container_width=True)

    st.markdown('<div class="group-label">Study tool</div>', unsafe_allow_html=True)
    st.radio("Choose what you want to do", TOOLS, key="selected_study_tool",
             label_visibility="collapsed")

    saved_chats = list_chats()

    if st.session_state.delete_target is not None:
        target_title = next(
            (short_title(c["title"], 34) for c in saved_chats if c["id"] == st.session_state.delete_target),
            "this conversation",
        )
        st.warning(f"Delete “{target_title}”? This cannot be undone.")
        yes_col, no_col = st.columns(2)
        yes_col.button("Delete", key="confirm_delete_btn", on_click=confirm_delete, use_container_width=True)
        no_col.button("Cancel", key="cancel_delete_btn", on_click=cancel_delete, use_container_width=True)

    if not saved_chats:
        st.markdown('<div class="group-label">Chats</div>', unsafe_allow_html=True)
        st.caption("No conversations yet. Ask your first question to get started.")
    else:
        for group_name, group_rows in group_chats(saved_chats):
            st.markdown(f'<div class="group-label">{group_name}</div>', unsafe_allow_html=True)
            for chat in group_rows:
                title_col, del_col = st.columns([6, 1], vertical_alignment="center")
                label = ("●  " if chat["id"] == st.session_state.chat_id else "") + short_title(chat["title"])
                title_col.button(label, key=f"chat_{chat['id']}", on_click=open_chat,
                                 args=(chat["id"],), use_container_width=True)
                del_col.button("🗑", key=f"del_{chat['id']}", on_click=ask_delete,
                               args=(chat["id"],), help="Delete conversation")

    st.divider()
    with st.expander("⚙️  Settings"):
        st.radio(
            "Appearance",
            THEMES,
            horizontal=True,
            key="appearance_choice",
            on_change=on_theme_change,
        )
        st.caption("System follows your browser/OS light or dark setting.")
    st.caption("Gemini AI · SQLite storage")


# =========================================================
# MAIN: CHAT AREA + COMPOSER
# =========================================================

chat_id = st.session_state.chat_id
history = get_messages(chat_id) if chat_id is not None else []
tool = st.session_state.selected_study_tool

# Created first so messages render ABOVE the composer, even though the
# composer's result is needed to decide what to render.
chat_area = st.container()

# Text to restore after a failed send, applied before the composer widget exists.
restored = st.session_state.pop("restore_text", None)
if restored:
    st.session_state.composer_text = restored

composer_cols = st.columns([12, 1], vertical_alignment="bottom")

with composer_cols[1]:
    mic_text = speech_to_text(
        language="en",
        start_prompt="🎙️",
        stop_prompt="■",
        just_once=True,
        key="composer_mic",
    )

# Voice transcript only FILLS the composer. It is never submitted automatically.
if mic_text and mic_text != st.session_state.last_mic_text:
    st.session_state.last_mic_text = mic_text
    existing = (st.session_state.composer_text or "").strip()
    st.session_state.composer_text = f"{existing} {mic_text}".strip() if existing else mic_text

with composer_cols[0]:
    st.markdown('<span class="composer-anchor"></span>', unsafe_allow_html=True)
    with st.form("message_composer", clear_on_submit=True, border=False):
        input_col, send_col = st.columns([10, 1], vertical_alignment="bottom")
        with input_col:
            typed_prompt = st.text_area(
                "Message",
                placeholder="Ask a BCA question or describe your study goal...",
                label_visibility="collapsed",
                key="composer_text",
                height=68,
                max_chars=MAX_MESSAGE_CHARS,
            )
        with send_col:
            send_clicked = st.form_submit_button("↑", help="Send message", use_container_width=True)

components.html(ENTER_TO_SEND_JS, height=0)

prompt = None
if send_clicked:
    cleaned = (typed_prompt or "").strip()
    if cleaned:
        prompt = cleaned
    else:
        st.toast("Type or dictate a message first.")


def render_tracker(tracker_chat_id):
    tasks = get_tasks(tracker_chat_id)
    if not tasks:
        return
    st.divider()
    st.subheader("📅 Timetable goals & progress")
    st.caption("Tick tasks as you complete them. Your progress is saved in SQLite.")
    completed = sum(1 for task in tasks if task["status"])
    st.progress(completed / len(tasks))
    st.write(f"**{completed} of {len(tasks)} tasks completed**")

    for task in tasks:
        label_parts = [p for p in (task["day"], task["time_slot"], task["subject"]) if p]
        label_parts.append(task["task"])
        if task["goal"]:
            label_parts.append(f"Goal: {task['goal']}")
        checked = st.checkbox(
            " · ".join(label_parts),
            value=bool(task["status"]),
            key=f"task_done_{task['id']}",
        )
        if checked != bool(task["status"]):
            set_task_status(task["id"], checked)
            st.rerun()


GREETING_HINTS = {
    TOOL_STEP: "Ask a BCA or programming question and I'll explain it step by step.",
    TOOL_SCHEDULE: "Tell me your subjects, daily study hours and exam date to get a timetable.",
    TOOL_NOTES: "Tap the mic, speak, edit the text if needed, then press send to get notes.",
}

with chat_area:
    # Fallback search panel for Streamlit versions without st.dialog.
    if st.session_state.show_search and not search_dialog:
        with st.container(border=True):
            render_search_body()
            st.button("Close search", key="close_search_btn", on_click=close_search_panel)

    if not history and not prompt:
        st.markdown(
            f'<div class="greeting-wrap"><div class="greeting">How can I help you?</div>'
            f'<div class="greeting-sub">{GREETING_HINTS[tool]}</div></div>',
            unsafe_allow_html=True,
        )

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    flash = st.session_state.pop("flash", None)
    if flash and not prompt:
        kind, message, details = flash
        if kind == "error":
            st.error(message)
            if details:
                with st.expander("Technical details"):
                    st.code(details)
        else:
            st.info(message)

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        prompt_for_ai, extra, wants_timetable = build_request(tool, prompt)
        failed = False
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = generate_text(prompt_for_ai, history, extra)
                except Exception as error:
                    failed = True
                    st.session_state.flash = ("error", friendly_error(error), str(error)[:400])
                    # Nothing was saved, so nothing is duplicated; give the text back.
                    st.session_state.restore_text = prompt

        if not failed:
            # Persist the exchange only after Gemini answered successfully.
            if chat_id is None:
                chat_id = create_chat()
                st.session_state.chat_id = chat_id
            save_message(chat_id, "user", prompt)
            save_message(chat_id, "assistant", answer)

            if wants_timetable:
                tasks = parse_markdown_table(answer)
                if tasks:
                    replace_chat_tasks(chat_id, tasks)
                    save_plan(chat_id, prompt, answer)
                else:
                    st.session_state.flash = (
                        "info",
                        "The timetable is shown above. You can ask me to format it as a table "
                        "to track it with checkboxes.",
                        "",
                    )
        st.rerun()

    if chat_id is not None:
        render_tracker(chat_id)

    if history:
        st.caption("Verify important facts and exam answers with your course materials.")
