from flask import Flask, render_template, request, jsonify
import requests
import os
import sqlite3

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_API_URL  = os.getenv("MODEL_API_URL", "http://localhost:8000")
MAX_NEW_TOKENS = 300

MODELS = {
    "qwen":  "Qwen 2.5",
    "llama": "Llama 3.1",
    "gemma": "Gemma 2",
}
DEFAULT_MODEL = "qwen"
# ──────────────────────────────────────────────────────────────────────────────

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "chatbot.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                model      TEXT NOT NULL,
                title      TEXT NOT NULL,
                starred    INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
        """)

init_db()
# ──────────────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("chat.html", models=MODELS, default_model=DEFAULT_MODEL)


# ── Session management ────────────────────────────────────────────────────────
@app.route("/sessions", methods=["GET"])
def get_sessions():
    model = request.args.get("model", DEFAULT_MODEL)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, title, starred FROM sessions WHERE model = ? ORDER BY updated_at DESC",
            (model,)
        ).fetchall()
    return jsonify([{"id": r["id"], "title": r["title"], "starred": bool(r["starred"])} for r in rows])


@app.route("/sessions/new", methods=["POST"])
def new_session():
    data  = request.get_json(silent=True) or {}
    sid   = data.get("id")
    model = data.get("model", DEFAULT_MODEL)
    title = data.get("title", "New Chat")
    if not sid:
        return jsonify({"error": "Missing id"}), 400
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, model, title) VALUES (?, ?, ?)",
            (sid, model, title)
        )
    return jsonify({"status": "ok"})


@app.route("/sessions/<sid>/messages", methods=["GET"])
def get_messages(sid):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (sid,)
        ).fetchall()
    return jsonify([{"role": r["role"], "content": r["content"]} for r in rows])


@app.route("/sessions/<sid>/rename", methods=["POST"])
def rename_session(sid):
    data  = request.get_json(silent=True) or {}
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Empty title"}), 400
    with get_db() as conn:
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, sid))
    return jsonify({"status": "ok"})


@app.route("/sessions/<sid>/star", methods=["POST"])
def star_session(sid):
    with get_db() as conn:
        row = conn.execute("SELECT starred FROM sessions WHERE id = ?", (sid,)).fetchone()
        if not row:
            return jsonify({"error": "Session not found"}), 404
        new_val = 0 if row["starred"] else 1
        conn.execute("UPDATE sessions SET starred = ? WHERE id = ?", (new_val, sid))
    return jsonify({"starred": bool(new_val)})


@app.route("/sessions/<sid>", methods=["DELETE"])
def delete_session(sid):
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    return jsonify({"status": "ok"})


@app.route("/debug/session/<sid>")
def debug_session(sid):
    with get_db() as conn:
        session = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        messages = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id", (sid,)
        ).fetchall()
    return jsonify({
        "session": dict(session) if session else None,
        "message_count": len(messages),
        "messages": [{"role": r["role"], "content": r["content"][:80]} for r in messages],
    })


# ── Chat ──────────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "Empty message"}), 400

    model_key = data.get("model", DEFAULT_MODEL)
    if model_key not in MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    user_message = data["message"].strip()
    session_id   = data.get("session_id", "")

    # Load history from DB
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
        history = [{"role": r["role"], "content": r["content"]} for r in rows]

    try:
        resp = requests.post(
            f"{MODEL_API_URL}/generate",
            json={
                "model":   model_key,
                "message": user_message,
                "history": history,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            return jsonify({"error": f"Model API error: {resp.status_code} {resp.text}"}), 500

        result            = resp.json()
        assistant_message = result.get("response", "").strip()

    except requests.exceptions.Timeout:
        return jsonify({"error": "Model took too long — please try again."}), 503
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach model_api.py — is it running on port 8000?"}), 503
    except Exception as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500

    # Save both messages to DB
    # INSERT OR IGNORE ensures the session row exists even if /sessions/new failed
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, model, title) VALUES (?, ?, ?)",
            (session_id, model_key, user_message[:60])
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "user", user_message)
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "assistant", assistant_message)
        )
        conn.execute(
            "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,)
        )

    return jsonify({"response": assistant_message, "model": model_key})


@app.route("/switch_model", methods=["POST"])
def switch_model():
    data      = request.get_json(silent=True)
    model_key = data.get("model") if data else None

    if model_key not in MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    try:
        resp = requests.post(
            f"{MODEL_API_URL}/switch",
            json={"model": model_key},
            timeout=120,
        )
        if resp.status_code != 200:
            return jsonify({"error": f"Switch failed: {resp.text}"}), 500
        return jsonify({"status": "ok", "model": model_key})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Model switch timed out."}), 503
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach model_api.py — is it running on port 8000?"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    try:
        resp = requests.post(
            f"{MODEL_API_URL}/transcribe",
            files={"audio": (audio_file.filename, audio_file.stream, audio_file.mimetype)},
            timeout=30,
        )
        if resp.status_code != 200:
            return jsonify({"error": f"Transcription failed: {resp.text}"}), 500
        return jsonify(resp.json())
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach model_api.py — is it running on port 8000?"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
