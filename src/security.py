from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from itertools import count

from cryptography.fernet import Fernet, InvalidToken


PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 310_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return "$".join(
        [
            PASSWORD_SCHEME,
            str(PASSWORD_ITERATIONS),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        ]
    )


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iterations, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if scheme != PASSWORD_SCHEME:
            return False
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(iterations),
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def password_is_reasonable(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if password.strip() != password:
        return False, "Password cannot start or end with spaces."
    return True, ""


def encrypt_secret(secret: str, app_secret_key: str) -> str:
    if not secret:
        return ""
    return Fernet(_fernet_key(app_secret_key)).encrypt(secret.encode("utf-8")).decode("ascii")


def decrypt_secret(encrypted_secret: str | None, app_secret_key: str) -> str:
    if not encrypted_secret:
        return ""
    try:
        return Fernet(_fernet_key(app_secret_key)).decrypt(encrypted_secret.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError, ValueError):
        return _decrypt_legacy_secret(encrypted_secret, app_secret_key)


def _fernet_key(app_secret_key: str) -> bytes:
    digest = hashlib.sha256(app_secret_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _decrypt_legacy_secret(encrypted_secret: str, app_secret_key: str) -> str:
    try:
        version, salt_b64, nonce_b64, ciphertext_b64, signature_b64 = encrypted_secret.split("$", 4)
        if version != "v1":
            return ""
        salt = base64.b64decode(salt_b64.encode("ascii"))
        nonce = base64.b64decode(nonce_b64.encode("ascii"))
        ciphertext = base64.b64decode(ciphertext_b64.encode("ascii"))
        signature = base64.b64decode(signature_b64.encode("ascii"))
        key = hashlib.pbkdf2_hmac(
            "sha256",
            app_secret_key.encode("utf-8"),
            salt,
            PASSWORD_ITERATIONS,
        )
        expected = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            return ""
        plaintext = _xor_bytes(ciphertext, _legacy_key_stream(key, nonce, len(ciphertext)))
        return plaintext.decode("utf-8")
    except Exception:
        return ""


def _legacy_key_stream(key: bytes, nonce: bytes, length: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for index in count():
        block = hmac.new(key, nonce + index.to_bytes(8, "big"), hashlib.sha256).digest()
        chunks.append(block)
        total += len(block)
        if total >= length:
            break
    return b"".join(chunks)[:length]


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right))
