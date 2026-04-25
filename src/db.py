from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import AppConfig
from .security import decrypt_secret, encrypt_secret, hash_password, verify_password


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(config: AppConfig) -> None:
    with connect(config.database_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            );

            CREATE TABLE IF NOT EXISTS voice_prints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                instructions TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                source_summary TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                voice_print_id INTEGER,
                title TEXT,
                instructions TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                tone_strength INTEGER NOT NULL,
                uploaded_manifest TEXT NOT NULL,
                article TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(voice_print_id) REFERENCES voice_prints(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS user_model_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                provider TEXT NOT NULL DEFAULT 'openai',
                active INTEGER NOT NULL DEFAULT 1,
                api_key_encrypted TEXT,
                base_url TEXT,
                model TEXT NOT NULL,
                voice_model TEXT NOT NULL,
                api_mode TEXT NOT NULL DEFAULT 'responses',
                reasoning_effort TEXT NOT NULL DEFAULT 'medium',
                max_output_tokens INTEGER NOT NULL DEFAULT 30000,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, provider),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )
        table_info = conn.execute("PRAGMA table_info(user_model_settings)").fetchall()
        columns = {row["name"] for row in table_info}
        if "id" not in columns:
            conn.execute("ALTER TABLE user_model_settings RENAME TO user_model_settings_old")
            conn.execute(
                """
                CREATE TABLE user_model_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    provider TEXT NOT NULL DEFAULT 'openai',
                    active INTEGER NOT NULL DEFAULT 1,
                    api_key_encrypted TEXT,
                    base_url TEXT,
                    model TEXT NOT NULL,
                    voice_model TEXT NOT NULL,
                    api_mode TEXT NOT NULL DEFAULT 'responses',
                    reasoning_effort TEXT NOT NULL DEFAULT 'medium',
                    max_output_tokens INTEGER NOT NULL DEFAULT 30000,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, provider),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                INSERT INTO user_model_settings
                (user_id, provider, active, api_key_encrypted, base_url, model, voice_model, api_mode, reasoning_effort, max_output_tokens, updated_at)
                SELECT user_id, provider, 1, api_key_encrypted, base_url, model, voice_model, api_mode, reasoning_effort, max_output_tokens, updated_at
                FROM user_model_settings_old
                """
            )
            conn.execute("DROP TABLE user_model_settings_old")
            table_info = conn.execute("PRAGMA table_info(user_model_settings)").fetchall()
            columns = {row["name"] for row in table_info}
        if "active" not in columns:
            conn.execute("ALTER TABLE user_model_settings ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
        indexes = conn.execute("PRAGMA index_list(user_model_settings)").fetchall()
        if not any(row["name"] == "idx_user_model_settings_user_provider" for row in indexes):
            conn.execute(
                "CREATE UNIQUE INDEX idx_user_model_settings_user_provider ON user_model_settings(user_id, provider)"
            )
        admin = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (config.admin_username,),
        ).fetchone()
        if admin is None:
            now = utc_now()
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, role, active, created_at, updated_at)
                VALUES (?, ?, ?, 'admin', 1, ?, ?)
                """,
                (
                    config.admin_username,
                    None,
                    hash_password(config.admin_password),
                    now,
                    now,
                ),
            )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def create_user(
    config: AppConfig,
    username: str,
    email: str | None,
    password: str,
    role: str = "user",
    active: bool = True,
) -> tuple[bool, str]:
    now = utc_now()
    try:
        with connect(config.database_path) as conn:
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, role, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username.strip(),
                    email.strip() if email else None,
                    hash_password(password),
                    role,
                    1 if active else 0,
                    now,
                    now,
                ),
            )
        return True, "User created."
    except sqlite3.IntegrityError as exc:
        message = str(exc).lower()
        if "username" in message:
            return False, "That username is already taken."
        if "email" in message:
            return False, "That email is already registered."
        return False, "Could not create user."


def get_user_by_username(config: AppConfig, username: str) -> dict[str, Any] | None:
    with connect(config.database_path) as conn:
        return row_to_dict(
            conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        )


def get_user(config: AppConfig, user_id: int) -> dict[str, Any] | None:
    with connect(config.database_path) as conn:
        return row_to_dict(
            conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        )


def authenticate(config: AppConfig, username: str, password: str) -> tuple[bool, str, dict[str, Any] | None]:
    user = get_user_by_username(config, username)
    if user is None or not verify_password(password, user["password_hash"]):
        return False, "Invalid username or password.", None
    if not user["active"]:
        return False, "This account is disabled.", None
    with connect(config.database_path) as conn:
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (utc_now(), user["id"]),
        )
    user.pop("password_hash", None)
    return True, "Signed in.", user


def list_users(config: AppConfig) -> list[dict[str, Any]]:
    with connect(config.database_path) as conn:
        rows = conn.execute(
            """
            SELECT id, username, email, role, active, created_at, updated_at, last_login_at
            FROM users
            ORDER BY created_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def update_user(config: AppConfig, user_id: int, email: str | None, role: str, active: bool) -> None:
    with connect(config.database_path) as conn:
        conn.execute(
            """
            UPDATE users
            SET email = ?, role = ?, active = ?, updated_at = ?
            WHERE id = ?
            """,
            (email.strip() if email else None, role, 1 if active else 0, utc_now(), user_id),
        )


def update_password(config: AppConfig, user_id: int, new_password: str) -> None:
    with connect(config.database_path) as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
            (hash_password(new_password), utc_now(), user_id),
        )


def get_user_model_settings(config: AppConfig, user_id: int, include_secret: bool = False) -> dict[str, Any]:
    with connect(config.database_path) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM user_model_settings
            WHERE user_id = ?
            ORDER BY active DESC, updated_at DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    settings = dict(row) if row else _default_model_settings(config, user_id)
    settings.pop("id", None)
    encrypted_key = settings.pop("api_key_encrypted", None)
    has_api_key = bool(encrypted_key)
    settings["has_api_key"] = has_api_key
    if include_secret:
        settings["api_key"] = decrypt_secret(encrypted_key, config.app_secret_key)
    return settings


def get_user_model_settings_for_provider(
    config: AppConfig,
    user_id: int,
    provider: str,
    include_secret: bool = False,
) -> dict[str, Any]:
    with connect(config.database_path) as conn:
        row = conn.execute(
            "SELECT * FROM user_model_settings WHERE user_id = ? AND provider = ?",
            (user_id, provider),
        ).fetchone()
    settings = dict(row) if row else _default_model_settings(config, user_id, provider)
    settings.pop("id", None)
    encrypted_key = settings.pop("api_key_encrypted", None)
    has_api_key = bool(encrypted_key)
    settings["has_api_key"] = has_api_key
    if include_secret:
        settings["api_key"] = decrypt_secret(encrypted_key, config.app_secret_key)
    return settings


def save_user_model_settings(
    config: AppConfig,
    user_id: int,
    *,
    provider: str,
    api_key: str | None,
    keep_existing_key: bool,
    base_url: str | None,
    model: str,
    voice_model: str,
    api_mode: str,
    reasoning_effort: str,
    max_output_tokens: int,
    active: bool = True,
) -> None:
    existing = get_user_model_settings_for_provider(config, user_id, provider)
    encrypted_key = None
    if keep_existing_key and existing["has_api_key"]:
        with connect(config.database_path) as conn:
            row = conn.execute(
                "SELECT api_key_encrypted FROM user_model_settings WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            ).fetchone()
            encrypted_key = row["api_key_encrypted"] if row else None
    elif api_key:
        encrypted_key = encrypt_secret(api_key, config.app_secret_key)

    clean_provider = provider if provider in {"openai", "ollama", "custom"} else "openai"
    clean_api_mode = api_mode if api_mode in {"responses", "chat", "ollama_native"} else "chat"
    clean_reasoning = reasoning_effort if reasoning_effort in {"none", "low", "medium", "high"} else "medium"
    clean_max_tokens = max(1000, min(100000, int(max_output_tokens)))
    now = utc_now()
    with connect(config.database_path) as conn:
        if active:
            conn.execute("UPDATE user_model_settings SET active = 0 WHERE user_id = ?", (user_id,))
        conn.execute(
            """
            INSERT INTO user_model_settings
            (user_id, provider, active, api_key_encrypted, base_url, model, voice_model, api_mode, reasoning_effort, max_output_tokens, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, provider) DO UPDATE SET
                active = excluded.active,
                api_key_encrypted = excluded.api_key_encrypted,
                base_url = excluded.base_url,
                model = excluded.model,
                voice_model = excluded.voice_model,
                api_mode = excluded.api_mode,
                reasoning_effort = excluded.reasoning_effort,
                max_output_tokens = excluded.max_output_tokens,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                clean_provider,
                1 if active else 0,
                encrypted_key,
                base_url.strip() if base_url else None,
                model.strip() or config.openai_model,
                voice_model.strip() or model.strip() or config.openai_voice_model,
                clean_api_mode,
                clean_reasoning,
                clean_max_tokens,
                now,
            ),
        )


def _default_model_settings(config: AppConfig, user_id: int, provider: str = "openai") -> dict[str, Any]:
    defaults = {
        "openai": {
            "base_url": None,
            "model": config.openai_model,
            "voice_model": config.openai_voice_model,
            "api_mode": "responses",
        },
        "ollama": {
            "base_url": "http://localhost:11434/api/chat",
            "model": "qwen3:8b",
            "voice_model": "qwen3:8b",
            "api_mode": "ollama_native",
        },
        "custom": {
            "base_url": "",
            "model": config.openai_model,
            "voice_model": config.openai_voice_model,
            "api_mode": "chat",
        },
    }
    provider_defaults = defaults.get(provider, defaults["openai"])
    return {
        "user_id": user_id,
        "provider": provider,
        "active": 1 if provider == "openai" else 0,
        "base_url": provider_defaults["base_url"],
        "model": provider_defaults["model"],
        "voice_model": provider_defaults["voice_model"],
        "api_mode": provider_defaults["api_mode"],
        "reasoning_effort": config.reasoning_effort,
        "max_output_tokens": config.max_output_tokens,
        "updated_at": "",
        "has_api_key": False,
    }


def save_voice_print(
    config: AppConfig,
    user_id: int,
    name: str,
    instructions: str,
    sample_count: int,
    source_summary: str,
) -> int:
    now = utc_now()
    with connect(config.database_path) as conn:
        conn.execute(
            "UPDATE voice_prints SET active = 0, updated_at = ? WHERE user_id = ?",
            (now, user_id),
        )
        cursor = conn.execute(
            """
            INSERT INTO voice_prints
            (user_id, name, instructions, sample_count, source_summary, active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (user_id, name.strip() or "Voice Print", instructions, sample_count, source_summary, now, now),
        )
        return int(cursor.lastrowid)


def list_voice_prints(config: AppConfig, user_id: int) -> list[dict[str, Any]]:
    with connect(config.database_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM voice_prints
            WHERE user_id = ?
            ORDER BY active DESC, created_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_voice_print(config: AppConfig, voice_print_id: int, user_id: int | None = None) -> dict[str, Any] | None:
    query = "SELECT * FROM voice_prints WHERE id = ?"
    args: tuple[Any, ...] = (voice_print_id,)
    if user_id is not None:
        query += " AND user_id = ?"
        args = (voice_print_id, user_id)
    with connect(config.database_path) as conn:
        return row_to_dict(conn.execute(query, args).fetchone())


def set_active_voice_print(config: AppConfig, user_id: int, voice_print_id: int) -> None:
    now = utc_now()
    with connect(config.database_path) as conn:
        conn.execute(
            "UPDATE voice_prints SET active = 0, updated_at = ? WHERE user_id = ?",
            (now, user_id),
        )
        conn.execute(
            "UPDATE voice_prints SET active = 1, updated_at = ? WHERE id = ? AND user_id = ?",
            (now, voice_print_id, user_id),
        )


def save_draft(
    config: AppConfig,
    user_id: int,
    voice_print_id: int | None,
    title: str,
    instructions: str,
    word_count: int,
    tone_strength: int,
    uploaded_manifest: Iterable[dict[str, Any]],
    article: str,
) -> int:
    with connect(config.database_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO drafts
            (user_id, voice_print_id, title, instructions, word_count, tone_strength, uploaded_manifest, article, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                voice_print_id,
                title.strip() if title else None,
                instructions,
                word_count,
                tone_strength,
                json.dumps(list(uploaded_manifest), ensure_ascii=False),
                article,
                utc_now(),
            ),
        )
        return int(cursor.lastrowid)


def list_drafts(config: AppConfig, user_id: int) -> list[dict[str, Any]]:
    with connect(config.database_path) as conn:
        rows = conn.execute(
            """
            SELECT d.*, v.name AS voice_print_name
            FROM drafts d
            LEFT JOIN voice_prints v ON v.id = d.voice_print_id
            WHERE d.user_id = ?
            ORDER BY d.created_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]
