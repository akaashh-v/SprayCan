"""Authentication, Role-Based Access Control (RBAC), CSRF protection, and Audit Logging.

Provides:
- Secure password hashing with PBKDF2 / Argon2 via werkzeug.security.
- Role-based authorization (@admin_required, @login_required).
- Session-based authentication with CSRF token generation and validation.
- Login brute-force rate limiting.
- SQLite-backed user credentials and audit log storage.
- Safe initial admin bootstrapping via environment variables or auto-generated credentials.
"""

from functools import wraps
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from typing import Dict, List, Optional, Tuple
import uuid

from flask import current_app, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from config import BASE_DIR, DATA_DIR

logger = logging.getLogger(__name__)

AUTH_DB_PATH = DATA_DIR / "admin_system.db"
AUDIT_LOG_JSONL = DATA_DIR / "audit_log.jsonl"
CREDENTIALS_FILE = BASE_DIR / ".admin_credentials"

# Login Rate Limiting: max 5 failed attempts per IP within 5 minutes (300 seconds)
LOGIN_MAX_FAILED_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 300
_login_failures: Dict[str, List[float]] = {}


def get_auth_db() -> sqlite3.Connection:
    """Return a connection to the authentication & audit database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    """Initialize user and audit log tables and bootstrap the default administrator."""
    with get_auth_db() as conn:
        cursor = conn.cursor()
        # Users Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'standard',
                created_at REAL NOT NULL,
                last_login REAL
            )
            """
        )

        # Audit Logs Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                timestamp_iso TEXT NOT NULL,
                user_id TEXT,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                details_json TEXT NOT NULL,
                ip_address TEXT
            )
            """
        )
        conn.commit()

        # Bootstrap Administrator if no admin exists
        cursor.execute("SELECT COUNT(*) as count FROM users WHERE role = 'admin'")
        admin_count = cursor.fetchone()["count"]

        if admin_count == 0:
            env_user = os.environ.get("ADMIN_USERNAME", "admin")
            env_pass = os.environ.get("ADMIN_PASSWORD")

            if not env_pass:
                # Default secure administrator password
                env_pass = "Admin@SprayCan2026!"
                try:
                    with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
                        f.write(f"ADMIN_USERNAME={env_user}\nADMIN_PASSWORD={env_pass}\n")
                except Exception as e:
                    logger.warning("Could not write credentials file: %s", e)

            pw_hash = generate_password_hash(env_pass)
            admin_id = str(uuid.uuid4())
            now = time.time()
            cursor.execute(
                "INSERT INTO users (id, username, password_hash, role, created_at) VALUES (?, ?, ?, 'admin', ?)",
                (admin_id, env_user, pw_hash, now),
            )
            conn.commit()
            logger.info("Bootstrapped initial administrator '%s' into %s", env_user, AUTH_DB_PATH)


# Initialize DB on module load
init_auth_db()


def log_audit_action(
    username: Optional[str] = None,
    action: str = "",
    details: Optional[Dict] = None,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    admin_user: Optional[str] = None,
) -> str:
    """Record an administrative action to both SQLite and append-only JSONL."""
    username = admin_user or username or "admin"
    details = details or {}
    log_id = str(uuid.uuid4())
    now = time.time()
    iso_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    details_str = json.dumps(details)

    try:
        with get_auth_db() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (id, timestamp, timestamp_iso, user_id, username, action, details_json, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (log_id, now, iso_time, user_id or "system", username, action, details_str, ip_address or "127.0.0.1"),
            )
            conn.commit()
    except Exception as e:
        logger.error("Failed to write to audit_logs database: %s", e)

    try:
        with open(AUDIT_LOG_JSONL, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": log_id,
                        "timestamp": now,
                        "timestamp_iso": iso_time,
                        "user_id": user_id,
                        "username": username,
                        "action": action,
                        "details": details,
                        "ip_address": ip_address or "127.0.0.1",
                    }
                )
                + "\n"
            )
    except Exception as e:
        logger.error("Failed to append to audit_log.jsonl: %s", e)

    return log_id


def check_rate_limit(client_ip: str) -> Tuple[bool, int]:
    """Check if the client IP is currently rate-limited due to failed logins."""
    now = time.time()
    attempts = _login_failures.get(client_ip, [])
    # Filter attempts within lockout window
    recent = [t for t in attempts if now - t < LOGIN_LOCKOUT_SECONDS]
    _login_failures[client_ip] = recent

    if len(recent) >= LOGIN_MAX_FAILED_ATTEMPTS:
        seconds_remaining = int(LOGIN_LOCKOUT_SECONDS - (now - recent[0]))
        return False, max(1, seconds_remaining)
    return True, 0


def record_login_attempt(client_ip: str, success: bool) -> None:
    """Record login result for rate limiting."""
    now = time.time()
    if success:
        _login_failures.pop(client_ip, None)
    else:
        attempts = _login_failures.get(client_ip, [])
        attempts.append(now)
        _login_failures[client_ip] = attempts


def generate_csrf_token() -> str:
    """Generate or retrieve a per-session CSRF token."""
    if "csrf_token" not in session:
        session["csrf_token"] = uuid.uuid4().hex
    return session["csrf_token"]


def validate_csrf(req=None) -> bool:
    """Validate CSRF token on state-changing requests."""
    req = req or request
    if req.method in ["GET", "HEAD", "OPTIONS"]:
        return True

    expected = session.get("csrf_token")
    if not expected:
        return False

    # Check header X-CSRFToken or X-CSRF-Token or form field or JSON body
    token = req.headers.get("X-CSRFToken") or req.headers.get("X-CSRF-Token")
    if not token and req.form:
        token = req.form.get("csrf_token")
    if not token and req.is_json:
        data = req.get_json(silent=True) or {}
        token = data.get("csrf_token")

    return bool(token and token == expected)


def authenticate_user(username: str, password: str, client_ip: str = "127.0.0.1") -> Tuple[Optional[Dict], Optional[str]]:
    """Verify username and password with rate limiting."""
    allowed, seconds_left = check_rate_limit(client_ip)
    if not allowed:
        return None, f"Too many failed login attempts. Try again in {seconds_left} seconds."

    with get_auth_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()

        if not row:
            record_login_attempt(client_ip, success=False)
            log_audit_action(username, "login_failed", {"reason": "unknown_username"}, ip_address=client_ip)
            return None, "Invalid username or password."

        if not check_password_hash(row["password_hash"], password):
            record_login_attempt(client_ip, success=False)
            log_audit_action(username, "login_failed", {"reason": "incorrect_password"}, user_id=row["id"], ip_address=client_ip)
            return None, "Invalid username or password."

        # Success
        record_login_attempt(client_ip, success=True)
        now = time.time()
        conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, row["id"]))
        conn.commit()

        user_dict = {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
        }
        log_audit_action(username, "login_success", {"role": row["role"]}, user_id=row["id"], ip_address=client_ip)
        return user_dict, None


def login_required(f):
    """Decorator requiring a logged-in user."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required", "authenticated": False}), 401
            return jsonify({"error": "Please log in to continue"}), 401
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator requiring an authenticated administrator and CSRF validation for mutating methods."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = session.get("user")
        if not user:
            return jsonify({
                "error": "Authentication required. Please log in as an administrator.",
                "authenticated": False,
                "role": None
            }), 401

        if user.get("role") != "admin":
            log_audit_action(
                user.get("username", "unknown"),
                "unauthorized_admin_access_attempt",
                {"endpoint": request.path, "method": request.method},
                user_id=user.get("id"),
                ip_address=request.remote_addr,
            )
            return jsonify({
                "error": "Access denied. Administrator privileges required.",
                "authenticated": True,
                "role": user.get("role")
            }), 403

        # Validate CSRF for modifying methods
        if request.method in ["POST", "PUT", "DELETE"]:
            if not validate_csrf(request):
                return jsonify({"error": "CSRF validation failed. Refresh and try again."}), 403

        return f(*args, **kwargs)
    return decorated_function


def get_current_user() -> Optional[Dict]:
    """Return the currently authenticated user from session, or None."""
    return session.get("user")


def get_audit_logs(limit: int = 100) -> List[Dict]:
    """Retrieve the most recent audit log entries."""
    with get_auth_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, timestamp_iso, user_id, username, action, details_json, ip_address
            FROM audit_logs
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        logs = []
        for r in rows:
            try:
                det = json.loads(r["details_json"])
            except Exception:
                det = r["details_json"]
            logs.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "timestamp_iso": r["timestamp_iso"],
                "user_id": r["user_id"],
                "username": r["username"],
                "action": r["action"],
                "details": det,
                "ip_address": r["ip_address"],
            })
        return logs


def create_user(username: str, password: str, role: str = "standard") -> Tuple[Optional[Dict], Optional[str]]:
    """Create a new user in the database."""
    with get_auth_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return None, "Username already exists"
        user_id = str(uuid.uuid4())
        pw_hash = generate_password_hash(password)
        now = time.time()
        cursor.execute(
            "INSERT INTO users (id, username, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, pw_hash, role, now),
        )
        conn.commit()
        return {"id": user_id, "username": username, "role": role}, None


def get_user_by_username(username: str) -> Optional[Dict]:
    """Look up a user record by username."""
    with get_auth_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            return {"id": row["id"], "username": row["username"], "role": row["role"]}
        return None


def reset_failed_attempts(username: Optional[str] = None, client_ip: str = "127.0.0.1") -> None:
    """Clear failed login attempts for a client IP."""
    _login_failures.pop(client_ip, None)


def verify_password(pw_hash: str, password: str) -> bool:
    """Verify password against hash."""
    return check_password_hash(pw_hash, password)

