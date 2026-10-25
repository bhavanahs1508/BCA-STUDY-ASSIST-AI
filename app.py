import base64
import json
import re
from datetime import datetime, date, timezone

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from streamlit_mic_recorder import speech_to_text

try:
    from supabase import create_client
except ImportError:  # app still works in guest mode
    create_client = None


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto",
)

MODEL = "gemini-3.5-flash-lite"  # unchanged
MAX_MESSAGE_CHARS = 12000
GUEST_LIMIT = 3
MIN_PASSWORD = 8
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

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

STEP_INSTRUCTION = """
Focus on step-by-step problem solving for BCA subjects. Use numbered steps,
explain why each step works, and include a small example when useful.
"""

QUIZ_INSTRUCTION = """
Create a short BCA quiz on the requested topic (5 questions unless the student
asks for another number). Use multiple-choice questions with options A-D.
Put all correct answers with one-line explanations in an 'Answer key' section
at the very end, after all the questions.
"""

THEMES = ["Light", "Dark", "System"]


# =========================================================
# SUPABASE CLIENT (one client per browser session, never shared)
# =========================================================

def _secret(*names):
    for name in names:
        try:
            value = st.secrets.get(name, "")
        except Exception:
            value = ""
        if value:
            return str(value).strip()
    return ""


def _key_is_service_role(key):
    """Refuse powerful keys: they bypass Row Level Security."""
    if key.startswith("sb_secret_"):
        return True
    try:
        payload = key.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload)).get("role") == "service_role"
    except Exception:
        return False


def get_sb():
    if "sb" not in st.session_state:
        if create_client is None:
            raise ValueError("The 'supabase' package is not installed. Add it to requirements.txt.")
        url = _secret("SUPABASE_URL")
        key = _secret("SUPABASE_KEY", "SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing in secrets.")
        if _key_is_service_role(key):
            raise ValueError("Use the anon/publishable Supabase key, not the service_role/secret key.")
        st.session_state.sb = create_client(url, key)
    return st.session_state.sb


def auth_client():
    try:
        return get_sb(), ""
    except Exception as error:
        return None, str(error)


def sb_table(name):
    return get_sb().table(name)


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def safe(fn, *args, default=None):
    """Run a database read; on failure remember the problem and return a default."""
    try:
        return fn(*args)
    except Exception:
        st.session_state.db_error = True
        return default


# =========================================================
# DATABASE (Supabase, protected by RLS: every row belongs to auth.uid())
# =========================================================

def make_title(text):
    clean = " ".join(text.split())
    return (clean[:42] + ("..." if len(clean) > 42 else "")) or "New chat"


def list_chats():
    res = (sb_table("bca_chats").select("id,title,updated_at")
           .order("updated_at", desc=True).limit(200).execute())
    return res.data or []


def _like(text):
    t = text.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{t}%"


def search_chats(query, limit=40):
    pat = _like(query)
    found = (sb_table("bca_chats").select("id,title,updated_at")
             .ilike("title", pat).limit(limit).execute().data or [])
    hits = sb_table("bca_messages").select("chat_id").ilike("content", pat).limit(200).execute().data or []
    extra_ids = {r["chat_id"] for r in hits} - {r["id"] for r in found}
    if extra_ids:
        found += (sb_table("bca_chats").select("id,title,updated_at")
                  .in_("id", list(extra_ids)).execute().data or [])
    found.sort(key=lambda r: r["updated_at"], reverse=True)
    return found[:limit]


def get_messages(chat_id):
    res = (sb_table("bca_messages").select("id,role,content,created_at")
           .eq("chat_id", chat_id).order("id").execute())
    return res.data or []


def create_chat(user_id, title):
    res = sb_table("bca_chats").insert({"user_id": user_id, "title": title}).execute()
    return res.data[0]["id"]


def delete_chat(chat_id):
    # messages and timetable tasks are removed by ON DELETE CASCADE
    sb_table("bca_chats").delete().eq("id", chat_id).execute()


def clear_history(user_id):
    sb_table("bca_chats").delete().eq("user_id", user_id).execute()


def persist_exchange(chat_id, user_id, prompt, answer):
    """Save question + answer together (one insert, so no half-saved or duplicate rows)."""
    created_here = False
    if chat_id is None:
        chat_id = create_chat(user_id, make_title(prompt))
        created_here = True
    try:
        sb_table("bca_messages").insert([
            {"chat_id": chat_id, "user_id": user_id, "role": "user", "content": prompt},
            {"chat_id": chat_id, "user_id": user_id, "role": "assistant", "content": answer},
        ]).execute()
        if not created_here:
            sb_table("bca_chats").update({"updated_at": iso_now()}).eq("id", chat_id).execute()
    except Exception:
        if created_here:
            try:
                delete_chat(chat_id)  # don't leave an empty chat behind
            except Exception:
                pass
        raise
    return chat_id


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


def replace_chat_tasks(chat_id, user_id, tasks):
    sb_table("bca_timetable_tasks").delete().eq("chat_id", chat_id).execute()
    rows = [{"chat_id": chat_id, "user_id": user_id, "day": t["day"], "time_slot": t["time_slot"],
             "subject": t["subject"], "task": t["task"], "goal": t["goal"], "status": False}
            for t in tasks]
    if rows:
        sb_table("bca_timetable_tasks").insert(rows).execute()


def get_tasks(chat_id):
    res = (sb_table("bca_timetable_tasks").select("*")
           .eq("chat_id", chat_id).order("id").execute())
    return res.data or []


def set_task_status(task_id, status):
    sb_table("bca_timetable_tasks").update({"status": bool(status)}).eq("id", task_id).execute()


# =========================================================
# HISTORY GROUPING HELPERS
# =========================================================

GROUP_ORDER = ["Today", "Yesterday", "Previous 7 Days", "Previous 30 Days", "Older"]


def chat_group(updated_at):
    try:
        d = datetime.fromisoformat(str(updated_at)[:19]).date()
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


# ---- Natural-language routing: the user just types what they need ----

def looks_like_timetable(prompt):
    p = prompt.lower()
    return any(w in p for w in (
        "timetable", "study plan", "study schedule", "revision plan",
        "study planner", "time table", "schedule my study",
    ))


def looks_like_notes(prompt):
    p = prompt.lower()
    return any(w in p for w in (
        "summarize this lecture", "summarise this lecture", "make notes from",
        "lecture notes", "summarize this transcript", "summarise this transcript",
        "notes from this", "transcript",
    ))


def looks_like_quiz(prompt):
    p = prompt.lower()
    return any(w in p for w in (
        "quiz", "mcq", "multiple choice", "test me", "practice questions", "practice test",
    ))


def looks_like_steps(prompt):
    p = prompt.lower()
    return any(w in p for w in (
        "step by step", "step-by-step", "solve", "dry run", "trace the", "find the output",
        "how to calculate", "work out",
    ))


def build_request(prompt):
    """Return (prompt_for_ai, extra_instruction, wants_timetable)."""
    if looks_like_timetable(prompt):
        return (
            "Create or refine a personalized study schedule based on this request. "
            "If essential details such as available hours, subjects, or deadline "
            "are missing, ask a short follow-up question instead of guessing.\n\n" + prompt,
            TABLE_INSTRUCTION, True,
        )
    if looks_like_notes(prompt):
        return (
            "Turn the following content into clear BCA study notes. Preserve the "
            "meaning; organize it with headings, definitions, key points, examples "
            "only when supported, and a short revision summary. Do not invent "
            "missing facts.\n\n" + prompt,
            NOTES_INSTRUCTION, False,
        )
    if looks_like_quiz(prompt):
        return prompt, QUIZ_INSTRUCTION, False
    if looks_like_steps(prompt):
        return (
            "Solve this BCA study question step by step. Explain each step in "
            "beginner-friendly language, show working/code and expected output "
            "when relevant.\n\n" + prompt,
            STEP_INSTRUCTION, False,
        )
    return prompt, "", False


# =========================================================
# SESSION STATE (all keys initialised once, up front)
# =========================================================

def init_state():
    defaults = {
        "theme": "System",
        "user": None,            # {"id": ..., "email": ...} when signed in
        "chat_id": None,         # None = fresh chat; a row is created on first saved answer
        "delete_target": None,
        "clear_pending": False,
        "composer_text": "",
        "last_mic_text": "",
        "guest_count": 0,
        "guest_messages": [],    # guest chats live only in this session
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "appearance_choice" not in st.session_state:
        st.session_state.appearance_choice = st.session_state.theme


init_state()

# A theme loaded at sign-in must be applied BEFORE the radio widget exists.
_pending_theme = st.session_state.pop("pending_theme", None)
if _pending_theme in THEMES:
    st.session_state.theme = _pending_theme
    st.session_state.appearance_choice = _pending_theme


# =========================================================
# AUTHENTICATION
# =========================================================

def clear_auth_state():
    st.session_state.user = None
    st.session_state.chat_id = None
    st.session_state.delete_target = None
    st.session_state.clear_pending = False
    st.session_state.pop("sb", None)  # drop this session's client and its tokens


def current_user():
    """Signed-in user, or None. Also refreshes an expiring token."""
    user = st.session_state.get("user")
    if not user:
        return None
    try:
        session = st.session_state.sb.auth.get_session()
    except Exception:
        session = None
    if session is None:
        clear_auth_state()
        return None
    return user


def set_user(sb_user):
    st.session_state.user = {"id": str(sb_user.id), "email": sb_user.email or ""}
    st.session_state.chat_id = None
    st.session_state.guest_messages = []
    meta = getattr(sb_user, "user_metadata", None) or {}
    if meta.get("theme") in THEMES:
        st.session_state.pending_theme = meta["theme"]


def sign_in(email, password):
    res = get_sb().auth.sign_in_with_password({"email": email, "password": password})
    set_user(res.user)


def sign_up(email, password):
    res = get_sb().auth.sign_up({"email": email, "password": password})
    if res.session and res.user:  # email confirmation is turned off in Supabase
        set_user(res.user)
        return "signed_in"
    if res.user is not None and getattr(res.user, "identities", None) == []:
        return "exists"
    return "verify"       # confirmation email sent


def sign_out():
    try:
        st.session_state.sb.auth.sign_out()
    except Exception:
        pass
    clear_auth_state()


def friendly_auth_error(error):
    low = (getattr(error, "message", None) or str(error)).lower()
    if "invalid login" in low or "invalid credentials" in low:
        return "Incorrect email or password."
    if "not confirmed" in low:
        return "Please verify your email first. Check your inbox for the confirmation link."
    if "already registered" in low or "already been registered" in low:
        return "An account with this email already exists. Try signing in."
    if "rate limit" in low or "too many" in low or "429" in low:
        return "Too many attempts. Please wait a few minutes and try again."
    if "password" in low and any(k in low for k in ("least", "weak", "short", "characters")):
        return "That password is too weak. Use at least 8 characters."
    if any(k in low for k in ("timeout", "timed out", "connection", "network", "getaddrinfo", "unreachable")):
        return "Couldn't reach the server. Please check your internet connection."
    return "Couldn't complete that. Please try again."


def render_auth_forms(prefix):
    sb, problem = auth_client()
    if sb is None:
        st.warning("Accounts are unavailable right now. You can keep using guest mode.")
        if problem:
            st.caption(problem)
        return
    tab_in, tab_up = st.tabs(["Sign in", "Sign up"])

    with tab_in:
        with st.form(f"{prefix}_signin_form", border=False):
            email = st.text_input("Email", key=f"{prefix}_si_email")
            password = st.text_input("Password", type="password", key=f"{prefix}_si_pw")
            go = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if go:
            email = email.strip().lower()
            if not EMAIL_RE.match(email) or not password:
                st.error("Enter a valid email and your password.")
            else:
                ok = False
                try:
                    with st.spinner("Signing in..."):
                        sign_in(email, password)
                    ok = True
                except Exception as error:
                    st.error(friendly_auth_error(error))
                if ok:
                    st.rerun()

    with tab_up:
        with st.form(f"{prefix}_signup_form", border=False):
            email = st.text_input("Email", key=f"{prefix}_su_email")
            password = st.text_input("Password (min 8 characters)", type="password", key=f"{prefix}_su_pw")
            confirm = st.text_input("Confirm password", type="password", key=f"{prefix}_su_pw2")
            go = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if go:
            email = email.strip().lower()
            if not EMAIL_RE.match(email):
                st.error("Enter a valid email address.")
            elif len(password) < MIN_PASSWORD:
                st.error(f"Password must be at least {MIN_PASSWORD} characters.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                status = None
                try:
                    with st.spinner("Creating your account..."):
                        status = sign_up(email, password)
                except Exception as error:
                    st.error(friendly_auth_error(error))
                if status == "signed_in":
                    st.rerun()
                elif status == "verify":
                    st.success("Account created! Check your email for a verification link, then sign in.")
                elif status == "exists":
                    st.info("An account with this email already exists. Try signing in.")


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


# NOTE: no blank lines inside this <style> block (Markdown would split the HTML there).
CSS_TEMPLATE = """
<style>
/*ROOT*/
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], [data-testid="stBottom"],
[data-testid="stBottomBlockContainer"], [data-testid="stHeader"] {
    background: var(--app-bg) !important; color: var(--text) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.stApp :is(p, li, h1, h2, h3, h4, h5, h6, label, summary, span, td, th) { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
#MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stDecoration"],
[data-testid="stAppDeployButton"], .stDeployButton { display: none !important; visibility: hidden !important; }
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; opacity: 1 !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stHeader"] button, [data-testid="stHeader"] svg,
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"] svg {
    color: var(--text) !important; fill: var(--text) !important;
}
[data-testid="InputInstructions"] { display: none !important; }
.block-container { max-width: 820px; padding-top: 3.2rem; padding-bottom: 1.5rem; }
/* Three-line menu button (falls back to Streamlit's default arrow on versions with other test ids) */
[data-testid="stExpandSidebarButton"] svg, [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapsedControl"] svg, [data-testid="collapsedControl"] svg { display: none !important; }
[data-testid="stExpandSidebarButton"]::before, [data-testid="stSidebarCollapsedControl"] button::before,
[data-testid="collapsedControl"] button::before { content: "☰"; font-size: 1.45rem; line-height: 1; color: var(--text); }
/* Title and greeting */
.app-title { text-align: center; font-size: 1.35rem; font-weight: 650; letter-spacing: -0.3px; color: var(--text); margin: 0 0 .6rem; }
.greeting-wrap { text-align: center; margin: 10vh 0 4vh; }
.greeting { font-size: 1.9rem; font-weight: 650; letter-spacing: -0.5px; color: var(--text); }
.greeting-sub { color: var(--muted); margin-top: .5rem; font-size: .98rem; }
/* Sidebar */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: var(--panel-bg) !important; }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .25rem !important; align-items: center; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div { min-width: 0 !important; }
[data-testid="stSidebar"] button[kind="secondary"], [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    border: none !important; background: transparent !important; box-shadow: none !important;
    justify-content: flex-start; text-align: left; border-radius: 10px; min-height: 38px; padding: .3rem .6rem;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover, [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: var(--hover) !important;
}
/* Tabs (History / Settings / Profile and Sign in / Sign up) */
[data-baseweb="tab-list"] { background: transparent !important; }
button[data-baseweb="tab"] { background: transparent !important; }
button[data-baseweb="tab"] * { color: var(--muted) !important; }
button[data-baseweb="tab"][aria-selected="true"] * { color: var(--text) !important; }
[data-baseweb="tab-highlight"] { background: var(--text) !important; }
[data-baseweb="tab-border"] { background: var(--border) !important; }
/* Messages */
[data-testid="stChatMessage"] { background: transparent; border: none; padding: .5rem .2rem; margin-bottom: .4rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--hover); border-radius: 18px; padding: .7rem 1rem;
}
pre, code, [data-testid="stCode"] pre { border-radius: 10px; background: var(--panel-bg) !important; color: var(--text) !important; }
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; }
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {
    border: 1px solid var(--border) !important; padding: .35rem .6rem; color: var(--text) !important;
}
[data-testid="stMarkdownContainer"] th { background: var(--panel-bg) !important; }
/* Inputs */
input, textarea, [data-baseweb="select"] > div { border-color: var(--border) !important; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="textarea"] > div,
[data-baseweb="input"] > div { background: var(--surface) !important; border-color: var(--border) !important; }
input, textarea { background-color: var(--surface) !important; color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; }
input::placeholder, textarea::placeholder { color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; opacity: 1; }
[data-baseweb="input"] svg { fill: var(--muted) !important; }
/* Buttons */
button[kind="secondary"], button[kind="secondaryFormSubmit"],
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {
    border: 1px solid var(--border) !important; border-radius: 12px; min-height: 40px;
    background: var(--surface) !important; color: var(--text) !important;
}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover { border-color: var(--text) !important; background: var(--hover) !important; }
button[kind="primary"], button[kind="primaryFormSubmit"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
    background: var(--button-bg) !important; color: var(--button-text) !important; border: none !important;
    border-radius: 12px; min-height: 40px; font-weight: 600;
}
button[kind="primary"] *, button[kind="primaryFormSubmit"] *,
[data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-primaryFormSubmit"] * { color: var(--button-text) !important; }
/* Alerts, expanders, menus, toasts */
[data-testid="stAlert"] { background: var(--panel-bg) !important; border: 1px solid var(--border); }
[data-testid="stAlert"] * { color: var(--text) !important; }
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 12px; background: var(--surface); }
[data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"] { background: var(--surface) !important; color: var(--text) !important; }
[data-testid="stToast"] { background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border); }
hr { border-color: var(--border) !important; }
/* Composer: one rounded pill pinned to the bottom (scoped so other forms are unaffected) */
@CMP@ {
    position: sticky; bottom: 12px; z-index: 50;
    background: var(--surface); border: 1px solid var(--border); border-radius: 30px;
    padding: 6px 8px 6px 16px; gap: .25rem !important;
    align-items: flex-end; flex-wrap: nowrap !important; margin-top: 1rem;
    box-shadow: 0 0 0 14px var(--app-bg);
}
@CMP@ > div { min-width: 0 !important; }
@CMP@ > div:first-child { flex: 1 1 auto !important; }
@CMP@ > div:last-child { flex: 0 0 auto !important; width: auto !important; min-width: 48px !important; }
.element-container:has(.composer-anchor) { display: none; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
@CMP@ [data-testid="stForm"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; align-items: flex-end; gap: .4rem !important; }
@CMP@ [data-testid="stForm"] [data-testid="stHorizontalBlock"] > div { min-width: 0 !important; }
@CMP@ [data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:first-child { flex: 1 1 auto !important; }
@CMP@ [data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:last-child { flex: 0 0 44px !important; width: 44px !important; min-width: 44px !important; }
@CMP@ [data-testid="stForm"] textarea { border: none !important; box-shadow: none !important; background: transparent !important; }
@CMP@ [data-testid="stForm"] [data-baseweb="textarea"], @CMP@ [data-testid="stForm"] [data-baseweb="textarea"] > div,
@CMP@ [data-testid="stForm"] [data-baseweb="base-input"] { border: none !important; background: transparent !important; box-shadow: none !important; }
@CMP@ [data-testid="stFormSubmitButton"] button {
    width: 40px; height: 40px; min-height: 40px; padding: 0; border-radius: 50% !important;
    font-size: 1.25rem; line-height: 1; justify-content: center;
    background: var(--button-bg) !important; color: var(--button-text) !important; border: none !important;
}
@CMP@ [data-testid="stFormSubmitButton"] button * { color: var(--button-text) !important; }
@media (max-width: 700px) {
    .block-container { padding: 3rem .8rem 1rem; }
    .greeting { font-size: 1.6rem; }
    .greeting-wrap { margin-top: 6vh; }
    @CMP@ { padding-left: 12px; }
}
</style>
"""


def apply_theme(theme):
    """Light/Dark are explicit. System follows the OS via prefers-color-scheme."""
    base = DARK if theme == "Dark" else LIGHT
    root = f":root {{ color-scheme: {'dark' if theme == 'Dark' else 'light'}; {_vars(base)} }}"
    if theme == "System":
        root += f" @media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark; {_vars(DARK)} }} }}"
    css = CSS_TEMPLATE.replace("@CMP@", '[data-testid="stHorizontalBlock"]:has(.composer-anchor)')
    st.markdown(css.replace("/*ROOT*/", root), unsafe_allow_html=True)


# Enter sends, Shift+Enter inserts a newline (desktop only).
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

user = current_user()
apply_theme(st.session_state.theme)


# =========================================================
# CALLBACKS
# =========================================================

def open_chat(chat_id):
    st.session_state.chat_id = chat_id
    st.session_state.delete_target = None
    st.session_state.clear_pending = False


def start_new_chat():
    # Nothing is written to the database: a chat row is created only when
    # its first answer is saved. Guest counter is NOT reset.
    st.session_state.chat_id = None
    st.session_state.guest_messages = []
    st.session_state.delete_target = None
    st.session_state.clear_pending = False


def ask_delete(chat_id):
    st.session_state.delete_target = chat_id
    st.session_state.clear_pending = False


def cancel_delete():
    st.session_state.delete_target = None


def confirm_delete():
    target = st.session_state.delete_target
    if target is not None:
        safe(delete_chat, target)
        if st.session_state.chat_id == target:
            st.session_state.chat_id = None
    st.session_state.delete_target = None


def ask_clear():
    st.session_state.clear_pending = True
    st.session_state.delete_target = None


def cancel_clear():
    st.session_state.clear_pending = False


def confirm_clear():
    u = st.session_state.get("user")
    if u:
        safe(clear_history, u["id"])
    st.session_state.chat_id = None
    st.session_state.clear_pending = False


def on_theme_change():
    choice = st.session_state.appearance_choice
    if choice not in THEMES:
        return
    st.session_state.theme = choice
    if st.session_state.get("user") and "sb" in st.session_state:
        try:  # saved on the account so it follows the user to any device
            st.session_state.sb.auth.update_user({"data": {"theme": choice}})
        except Exception:
            pass  # preference saving is best-effort


# =========================================================
# SIDEBAR  (☰ opens it: History, Settings, Profile)
# =========================================================

def render_history_tab(all_chats):
    if not user:
        st.caption("Sign in to save your conversations and find them here later. "
                   "Guest chats are not saved.")
        return

    query = st.text_input("Search history", placeholder="Search history...",
                          label_visibility="collapsed", key="history_search")
    rows = safe(search_chats, query, default=[]) if query.strip() else all_chats

    if st.session_state.delete_target is not None:
        target_title = next(
            (short_title(c["title"], 34) for c in all_chats if c["id"] == st.session_state.delete_target),
            "this conversation",
        )
        st.warning(f"Delete “{target_title}”? This cannot be undone.")
        yes_col, no_col = st.columns(2)
        yes_col.button("Delete", key="confirm_delete_btn", on_click=confirm_delete, use_container_width=True)
        no_col.button("Cancel", key="cancel_delete_btn", on_click=cancel_delete, use_container_width=True)

    if not all_chats:
        st.caption("No conversations yet. Ask your first question to get started.")
        return
    if query.strip() and not rows:
        st.caption("No conversations found.")

    for group_name, group_rows in group_chats(rows):
        st.caption(group_name)
        for chat in group_rows:
            title_col, del_col = st.columns([6, 1], vertical_alignment="center")
            label = ("●  " if chat["id"] == st.session_state.chat_id else "") + short_title(chat["title"])
            title_col.button(label, key=f"chat_{chat['id']}", on_click=open_chat,
                             args=(chat["id"],), use_container_width=True)
            del_col.button("🗑", key=f"del_{chat['id']}", on_click=ask_delete,
                           args=(chat["id"],), help="Delete conversation")

    st.divider()
    if st.session_state.clear_pending:
        st.warning("Delete ALL your conversations? This cannot be undone.")
        yes_col, no_col = st.columns(2)
        yes_col.button("Yes, clear all", key="confirm_clear_btn", on_click=confirm_clear, use_container_width=True)
        no_col.button("Cancel", key="cancel_clear_btn", on_click=cancel_clear, use_container_width=True)
    else:
        st.button("Clear all history", key="clear_all_btn", on_click=ask_clear, use_container_width=True)


def render_settings_tab():
    st.radio("Appearance", THEMES, horizontal=True, key="appearance_choice", on_change=on_theme_change)
    st.caption("System follows your device's light or dark setting.")
    if user:
        st.caption("Your appearance choice is saved to your account.")


def render_profile_tab():
    if user:
        st.text(user["email"])
        st.caption("Signed in. Your chats are saved privately to your account.")
        st.button("Sign out", key="signout_btn", on_click=sign_out, use_container_width=True)
    else:
        left = max(GUEST_LIMIT - st.session_state.guest_count, 0)
        st.caption(f"Guest mode · {left} of {GUEST_LIMIT} free questions left. "
                   "Sign up or sign in to unlock everything and save your history.")
        render_auth_forms("side")


all_chats = safe(list_chats, default=[]) if user else []

with st.sidebar:
    st.button("＋  New chat", key="new_chat_btn", on_click=start_new_chat,
              use_container_width=True, type="primary")
    tab_history, tab_settings, tab_profile = st.tabs(["History", "Settings", "Profile"])
    with tab_history:
        render_history_tab(all_chats)
    with tab_settings:
        render_settings_tab()
    with tab_profile:
        render_profile_tab()


# =========================================================
# MAIN: CHAT AREA + COMPOSER
# =========================================================

st.markdown('<div class="app-title">AI Study Assistant</div>', unsafe_allow_html=True)

chat_id = st.session_state.chat_id
guest_locked = user is None and st.session_state.guest_count >= GUEST_LIMIT

if user:
    history = safe(get_messages, chat_id, default=[]) if chat_id is not None else []
else:
    history = st.session_state.guest_messages

# Created first so messages render ABOVE the composer.
chat_area = st.container()

# Text to restore after a failed send, applied before the composer widget exists.
restored = st.session_state.pop("restore_text", None)
if restored:
    st.session_state.composer_text = restored

composer_cols = st.columns([12, 1], vertical_alignment="bottom")

mic_text = None
if not guest_locked:
    with composer_cols[1]:
        mic_text = speech_to_text(
            language="en", start_prompt="🎙️", stop_prompt="■",
            just_once=True, key="composer_mic",
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
                placeholder=("Sign up or sign in to continue..." if guest_locked
                             else "Ask a BCA question or describe your study goal..."),
                label_visibility="collapsed",
                key="composer_text",
                height=68,
                max_chars=MAX_MESSAGE_CHARS,
                disabled=guest_locked,
            )
        with send_col:
            send_clicked = st.form_submit_button("↑", help="Send message",
                                                 use_container_width=True, disabled=guest_locked)

components.html(ENTER_TO_SEND_JS, height=0)

prompt = None
if send_clicked and not guest_locked:
    cleaned = (typed_prompt or "").strip()
    if cleaned:
        prompt = cleaned
    else:
        st.toast("Type or dictate a message first.")


def render_tracker(tracker_chat_id):
    tasks = safe(get_tasks, tracker_chat_id, default=[])
    if not tasks:
        return
    st.divider()
    st.subheader("📅 Timetable goals & progress")
    st.caption("Tick tasks as you complete them. Your progress is saved to your account.")
    completed = sum(1 for task in tasks if task["status"])
    st.progress(completed / len(tasks))
    st.write(f"**{completed} of {len(tasks)} tasks completed**")
    for task in tasks:
        label_parts = [p for p in (task["day"], task["time_slot"], task["subject"]) if p]
        label_parts.append(task["task"])
        if task["goal"]:
            label_parts.append(f"Goal: {task['goal']}")
        checked = st.checkbox(" · ".join(label_parts), value=bool(task["status"]),
                              key=f"task_done_{task['id']}")
        if checked != bool(task["status"]):
            try:
                set_task_status(task["id"], checked)
            except Exception:
                st.toast("Couldn't save that change. Please try again.")
            st.rerun()


with chat_area:
    if st.session_state.pop("db_error", False):
        st.warning("Couldn't reach your saved chats just now. Please check your connection and try again.")

    if not history and not prompt:
        st.markdown(
            '<div class="greeting-wrap"><div class="greeting">How can I help you today?</div>'
            '<div class="greeting-sub">Ask anything about your BCA subjects.</div></div>',
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

        prompt_for_ai, extra, wants_timetable = build_request(prompt)
        failed = False
        assistant_box = st.chat_message("assistant")
        with assistant_box:
            with st.spinner("Thinking..."):
                try:
                    answer = generate_text(prompt_for_ai, history, extra)
                except Exception as error:
                    failed = True
                    st.session_state.flash = ("error", friendly_error(error), str(error)[:400])
                    # Nothing was saved, so nothing is duplicated; give the text back.
                    st.session_state.restore_text = prompt

        saved = True
        if not failed:
            if user:
                # Persist only after Gemini answered successfully, in a single insert.
                try:
                    new_id = persist_exchange(chat_id, user["id"], prompt, answer)
                    chat_id = new_id
                    st.session_state.chat_id = new_id
                except Exception:
                    saved = False
            else:
                st.session_state.guest_messages.append({"role": "user", "content": prompt})
                st.session_state.guest_messages.append({"role": "assistant", "content": answer})
                st.session_state.guest_count += 1

            if saved and wants_timetable:
                if user:
                    tasks = parse_markdown_table(answer)
                    tracked = False
                    if tasks:
                        try:
                            replace_chat_tasks(chat_id, user["id"], tasks)
                            tracked = True
                        except Exception:
                            pass
                    if not tracked:
                        st.session_state.flash = (
                            "info",
                            "The timetable is shown above. You can ask me to format it as a table "
                            "to track it with checkboxes.",
                            "",
                        )
                else:
                    st.session_state.flash = (
                        "info", "Sign in to track this timetable with checkboxes and keep it saved.", "",
                    )

        if saved:
            st.rerun()
        elif not failed:
            # Answer was generated but could not be saved: show it so it isn't lost.
            with assistant_box:
                st.markdown(answer)
            st.warning("This answer couldn't be saved to your history. Copy it if you need it, "
                       "then check your connection.")

    if guest_locked:
        st.info(f"You've used your {GUEST_LIMIT} free guest questions. "
                "Please sign up or sign in to keep going. It's free, and your chats will be saved.")
        render_auth_forms("main")

    if user and chat_id is not None:
        render_tracker(chat_id)

    if history:
        if user is None and not guest_locked:
            left = GUEST_LIMIT - st.session_state.guest_count
            st.caption(f"Guest mode · {left} free question{'s' if left != 1 else ''} left. "
                       "Sign in from the ☰ menu to save your chats.")
        st.caption("Verify important facts and exam answers with your course materials.")