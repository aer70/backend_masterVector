from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)

import bcrypt
import jwt
import psycopg
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from psycopg.rows import dict_row
from pydantic import BaseModel, EmailStr
from redis import Redis
from rq import Queue

from vectorization_service import parse_settings_json, process_job, validate_settings_payload
from backend.ai_service import suggest_settings, chat_about_settings

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
RESULTS_DIR = STORAGE_DIR / "results"

MAX_BMP_BYTES = 25 * 1024 * 1024
ALLOWED_BMP_CONTENT_TYPES = {"image/bmp", "image/x-ms-bmp", "application/octet-stream"}

JWT_SECRET = os.getenv("BMP2SVG_JWT_SECRET", "change-this-secret-in-production")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:asDUpnyQpQuaQEtPPqNOvCDvXnIazqFX@postgres.railway.internal:5432/railway")
JWT_ALGO = "HS256"
ACCESS_TTL_MINUTES = 30
REFRESH_TTL_DAYS = 7

AUTH_RATE_LIMIT_COUNT = int(os.getenv("BMP2SVG_RATE_LIMIT_AUTH_COUNT", "20"))
AUTH_RATE_LIMIT_WINDOW_SEC = int(os.getenv("BMP2SVG_RATE_LIMIT_AUTH_WINDOW_SEC", "60"))
UPLOAD_RATE_LIMIT_COUNT = int(os.getenv("BMP2SVG_RATE_LIMIT_UPLOAD_COUNT", "10"))
UPLOAD_RATE_LIMIT_WINDOW_SEC = int(os.getenv("BMP2SVG_RATE_LIMIT_UPLOAD_WINDOW_SEC", "60"))
SIGNED_DOWNLOAD_MAX_TTL_SEC = int(os.getenv("BMP2SVG_SIGNED_DOWNLOAD_MAX_TTL_SEC", "3600"))
SIGNED_DOWNLOAD_DEFAULT_TTL_SEC = int(os.getenv("BMP2SVG_SIGNED_DOWNLOAD_DEFAULT_TTL_SEC", "300"))
REDIS_URL = os.getenv("REDIS_URL", "redis://default:tBpcxWxcJmvmOrQnrDDzhjHhTMeTnElJ@redis.railway.internal:6379")
QUEUE_NAME = os.getenv("BMP2SVG_QUEUE_NAME", "bmp2svg-jobs")
JOB_MAX_RETRIES = int(os.getenv("BMP2SVG_JOB_MAX_RETRIES", "2"))
ADMIN_TOKEN = os.getenv("BMP2SVG_ADMIN_TOKEN", "change-this-admin-token")
STUCK_JOB_SEC = int(os.getenv("BMP2SVG_STUCK_JOB_SEC", "600"))
STATE_USE_REDIS = os.getenv("BMP2SVG_STATE_USE_REDIS", "1") == "1"

RATE_LIMIT_KEY_PREFIX = "rl"
DOWNLOAD_USED_KEY_PREFIX = "dl:used"

rate_limit_lock = threading.Lock()
rate_limit_buckets: dict[str, list[float]] = {}

download_once_lock = threading.Lock()
used_download_tokens: dict[str, float] = {}

redis_lock = threading.Lock()
_redis_conn: Redis | None = None


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class JobListItem(BaseModel):
    id: str
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_text: str | None
    duration_ms: int | None


class JobDetails(JobListItem):
    settings_json: dict[str, Any]


class PresetCreateRequest(BaseModel):
    name: str
    settings_json: dict[str, Any]


class PresetUpdateRequest(BaseModel):
    name: str
    settings_json: dict[str, Any]


class PresetItem(BaseModel):
    id: int
    name: str
    settings_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SignedDownloadLinkResponse(BaseModel):
    url: str
    expires_at: datetime
    single_use: bool


class AdminBlockUserRequest(BaseModel):
    blocked: bool
    reason: str | None = None


class AdminStuckJobItem(BaseModel):
    id: str
    status: str
    retry_count: int
    max_retries: int
    started_at: datetime | None
    last_heartbeat_at: datetime | None
    error_text: str | None


class AdminMetricsResponse(BaseModel):
    jobs_total: int
    jobs_running: int
    jobs_failed: int
    jobs_success: int
    avg_duration_ms: float | None
    p95_duration_ms: float | None
    input_bytes_total: int
    svg_bytes_total: int
    log_bytes_total: int
    queue_depth: int


app = FastAPI(title="BMP2SVG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_token(payload: dict[str, Any], expires_delta: timedelta) -> str:
    data = payload.copy()
    data["exp"] = datetime.now(timezone.utc) + expires_delta
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def db_connect() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def assert_db_schema_ready() -> None:
    """Ensure schema is managed externally via Alembic migrations."""
    required_tables = ("users", "refresh_tokens", "jobs", "presets")
    required_columns = {
        "users": ("is_blocked",),
        "jobs": (
            "retry_count",
            "max_retries",
            "queue_job_id",
            "last_heartbeat_at",
            "input_size_bytes",
            "svg_size_bytes",
            "log_size_bytes",
        ),
    }

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.alembic_version') AS reg")
            alembic_table = cur.fetchone()
            if not alembic_table or alembic_table["reg"] is None:
                raise RuntimeError(
                    "Database is not initialized with Alembic. Run: python -m alembic -c alembic.ini upgrade head"
                )

            for table in required_tables:
                cur.execute("SELECT to_regclass(%s) AS reg", (f"public.{table}",))
                row = cur.fetchone()
                if not row or row["reg"] is None:
                    raise RuntimeError(
                        f"Required table '{table}' is missing. Run: python -m alembic -c alembic.ini upgrade head"
                    )

            for table, columns in required_columns.items():
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    """,
                    (table,),
                )
                existing = {r["column_name"] for r in cur.fetchall()}
                missing = [c for c in columns if c not in existing]
                if missing:
                    raise RuntimeError(
                        f"Missing required columns in '{table}': {', '.join(missing)}. "
                        "Run: python -m alembic -c alembic.ini upgrade head"
                    )


def init_runtime() -> None:
    ensure_storage()
    assert_db_schema_ready()


def get_user_by_email(email: str) -> dict[str, Any] | None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email.lower().strip(),))
            return cur.fetchone()


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            return cur.fetchone()


def get_bearer_token(auth_header: str | None) -> str:
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return auth_header.split(" ", 1)[1].strip()


def auth_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = get_bearer_token(authorization)
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid access token")

    user_id = int(payload.get("sub", 0))
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    if bool(user.get("is_blocked", False)):
        raise HTTPException(status_code=403, detail="User is blocked")
    return user


def auth_user_optional(authorization: str | None = Header(default=None)) -> dict[str, Any] | None:
    if not authorization:
        return None
    return auth_user(authorization)


def normalize_guest_token(token: str | None) -> str | None:
    if token is None:
        return None
    normalized = token.strip()
    if not normalized:
        return None
    if len(normalized) > 128:
        raise HTTPException(status_code=400, detail="X-Guest-Token is too long")
    return normalized


def issue_token_pair(user_id: int) -> dict[str, str]:
    access_token = create_token({"sub": str(user_id), "type": "access"}, timedelta(minutes=ACCESS_TTL_MINUTES))
    refresh_token = create_token({"sub": str(user_id), "type": "refresh"}, timedelta(days=REFRESH_TTL_DAYS))

    token_hash = hash_token(refresh_token)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO refresh_tokens (token_hash, user_id, expires_at, revoked, created_at)
                VALUES (%s, %s, %s, FALSE, %s)
                ON CONFLICT (token_hash)
                DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    expires_at = EXCLUDED.expires_at,
                    revoked = FALSE,
                    created_at = EXCLUDED.created_at
                """,
                (
                    token_hash,
                    user_id,
                    datetime.now(timezone.utc) + timedelta(days=REFRESH_TTL_DAYS),
                    datetime.now(timezone.utc),
                ),
            )
        conn.commit()

    return {"access_token": access_token, "refresh_token": refresh_token}


def revoke_refresh_token(refresh_token: str) -> None:
    token_hash = hash_token(refresh_token)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE refresh_tokens SET revoked = TRUE WHERE token_hash = %s", (token_hash,))
        conn.commit()


def assert_active_refresh_token(refresh_token: str, user_id: int) -> None:
    token_hash = hash_token(refresh_token)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT revoked, expires_at, user_id FROM refresh_tokens WHERE token_hash = %s",
                (token_hash,),
            )
            row = cur.fetchone()
    if row is None or int(row["user_id"]) != user_id:
        raise HTTPException(status_code=401, detail="Refresh token not found")
    if bool(row["revoked"]):
        raise HTTPException(status_code=401, detail="Refresh token revoked")
    if row["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token expired")


def ensure_bmp_file(file: UploadFile, payload: bytes) -> None:
    filename = file.filename or ""
    ext_ok = filename.lower().endswith(".bmp")
    mime = (file.content_type or "").lower().strip()
    mime_ok = mime in ALLOWED_BMP_CONTENT_TYPES

    if not ext_ok:
        raise HTTPException(status_code=400, detail="Only .bmp files are allowed")
    if mime and not mime_ok:
        raise HTTPException(status_code=400, detail=f"Unsupported mime type: {mime}")
    if len(payload) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    if len(payload) > MAX_BMP_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large, max is {MAX_BMP_BYTES} bytes")


def normalize_filename_stem(name: str) -> str:
    stem = Path(name).stem.strip()
    if not stem:
        stem = "file"
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._-")
    if not stem:
        stem = "file"
    return stem[:120]


def pick_unique_file_path(directory: Path, stem: str, suffix: str) -> Path:
    idx = 1
    while True:
        candidate_name = f"{stem}{suffix}" if idx == 1 else f"{stem}_{idx}{suffix}"
        candidate_path = directory / candidate_name
        if not candidate_path.exists():
            return candidate_path
        idx += 1


def normalize_preset_name(name: str) -> str:
    normalized = name.strip()
    if len(normalized) < 1:
        raise HTTPException(status_code=400, detail="Preset name must not be empty")
    if len(normalized) > 80:
        raise HTTPException(status_code=400, detail="Preset name must be <= 80 characters")
    return normalized


def to_preset_item(row: dict[str, Any]) -> PresetItem:
    settings_obj = row["settings_json"]
    if isinstance(settings_obj, str):
        settings_obj = json.loads(settings_obj)
    return PresetItem(
        id=int(row["id"]),
        name=str(row["name"]),
        settings_json=settings_obj,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def extract_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        first = forwarded.split(",", 1)[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def apply_rate_limit(bucket: str, request: Request, limit: int, window_sec: int) -> None:
    if limit <= 0 or window_sec <= 0:
        return

    client_ip = extract_client_ip(request)
    now_epoch = int(time.time())

    if STATE_USE_REDIS:
        try:
            redis_conn = get_redis_connection()
            window_slot = now_epoch // int(window_sec)
            redis_key = f"{RATE_LIMIT_KEY_PREFIX}:{bucket}:{client_ip}:{window_slot}"
            count = int(redis_conn.incr(redis_key))
            if count == 1:
                redis_conn.expire(redis_key, int(window_sec) + 5)
            if count > limit:
                raise HTTPException(status_code=429, detail="Rate limit exceeded, try again later")
            return
        except HTTPException:
            raise
        except Exception:
            # Fallback for local development or transient Redis outage.
            pass

    now = time.monotonic()
    key = f"{bucket}:{client_ip}"
    boundary = now - float(window_sec)
    with rate_limit_lock:
        events = rate_limit_buckets.get(key, [])
        events = [ts for ts in events if ts >= boundary]
        if len(events) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded, try again later")
        events.append(now)
        rate_limit_buckets[key] = events


def _cleanup_used_download_tokens() -> None:
    now = time.time()
    expired = [jti for jti, exp_ts in used_download_tokens.items() if exp_ts <= now]
    for jti in expired:
        used_download_tokens.pop(jti, None)


def create_signed_download_token(
    *,
    job_id: str,
    file_key: str,
    kind: str,
    ttl_seconds: int,
    single_use: bool,
) -> str:
    token_id = uuid.uuid4().hex
    payload = {
        "type": "download",
        "job_id": job_id,
        "kind": kind,
        "file_key": file_key,
        "single_use": bool(single_use),
        "jti": token_id,
    }
    return create_token(payload, timedelta(seconds=ttl_seconds))


def consume_single_use_download_token(token_id: str, expires_at_epoch: float) -> None:
    ttl = max(1, int(expires_at_epoch - time.time()))

    if STATE_USE_REDIS:
        try:
            redis_conn = get_redis_connection()
            redis_key = f"{DOWNLOAD_USED_KEY_PREFIX}:{token_id}"
            created = redis_conn.set(redis_key, b"1", ex=ttl, nx=True)
            if not created:
                raise HTTPException(status_code=410, detail="Signed link was already used")
            return
        except HTTPException:
            raise
        except Exception:
            # Fallback to in-memory when Redis state backend is unavailable.
            pass

    with download_once_lock:
        _cleanup_used_download_tokens()
        if token_id in used_download_tokens:
            raise HTTPException(status_code=410, detail="Signed link was already used")
        used_download_tokens[token_id] = float(expires_at_epoch)


def reset_runtime_state_for_tests() -> None:
    """Best-effort cleanup for test isolation (both in-memory and Redis state)."""
    rate_limit_buckets.clear()
    used_download_tokens.clear()

    if not STATE_USE_REDIS:
        return

    try:
        redis_conn = get_redis_connection()
    except Exception:
        return

    for pattern in (f"{RATE_LIMIT_KEY_PREFIX}:*", f"{DOWNLOAD_USED_KEY_PREFIX}:*"):
        cursor = 0
        while True:
            cursor, keys = redis_conn.scan(cursor=cursor, match=pattern, count=200)
            if keys:
                redis_conn.delete(*keys)
            if cursor == 0:
                break


def resolve_keyed_path(file_key: str, kind: str) -> Path:
    path = (BASE_DIR / file_key).resolve()
    if not path.is_file() or BASE_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail=f"{kind.upper()} file not found")
    return path


def auth_admin(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> None:
    if not x_admin_token or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")


def get_redis_connection() -> Redis:
    global _redis_conn
    with redis_lock:
        if _redis_conn is None:
            _redis_conn = Redis.from_url(REDIS_URL, decode_responses=False)
        return _redis_conn


def get_job_queue() -> Queue:
    return Queue(name=QUEUE_NAME, connection=get_redis_connection())


def enqueue_job(job_id: str) -> str:
    queue = get_job_queue()
    job = queue.enqueue("backend.main.run_job_once", job_id, job_timeout=1800)
    return str(job.id)


def run_job_once(job_id: str) -> None:
    now = datetime.now(timezone.utc)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
            job = cur.fetchone()
            if job is None:
                return
            if job["status"] not in {"queued", "running"}:
                return

            cur.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = %s, last_heartbeat_at = %s, worker_id = %s
                WHERE id = %s
                """,
                (now, now, f"pid-{os.getpid()}", job_id),
            )
        conn.commit()

    input_path = str(BASE_DIR / job["input_file_key"])
    output_dir = str(RESULTS_DIR.resolve())
    settings = job["settings_json"]
    if isinstance(settings, str):
        settings = json.loads(settings)
    input_stem = normalize_filename_stem(Path(str(job["input_file_key"])).name)

    result = process_job(
        input_path=input_path,
        settings=settings,
        output_dir=output_dir,
        output_basename=input_stem,
        job_id=job_id,
    )

    svg_key = None
    svg_size = None
    if result.get("svg_path"):
        svg_path = Path(str(result["svg_path"]))
        svg_key = os.path.relpath(svg_path, BASE_DIR)
        if svg_path.is_file():
            svg_size = int(svg_path.stat().st_size)

    log_path = Path(str(result["log_path"]))
    log_key = os.path.relpath(log_path, BASE_DIR)
    log_size = int(log_path.stat().st_size) if log_path.is_file() else None

    if result.get("status") == "success":
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'success', svg_file_key = %s, log_file_key = %s,
                        error_text = NULL, finished_at = %s, duration_ms = %s,
                        last_heartbeat_at = %s, svg_size_bytes = %s, log_size_bytes = %s
                    WHERE id = %s
                    """,
                    (
                        svg_key,
                        log_key,
                        datetime.now(timezone.utc),
                        int(result.get("duration_ms") or 0),
                        datetime.now(timezone.utc),
                        svg_size,
                        log_size,
                        job_id,
                    ),
                )
            conn.commit()
        return

    retry_count = int(job.get("retry_count") or 0)
    max_retries = int(job.get("max_retries") or JOB_MAX_RETRIES)
    retry_next = retry_count + 1
    error_text = str(result.get("error_message") or "Unknown worker error")
    finished_at = datetime.now(timezone.utc)

    if retry_count < max_retries:
        queue_job_id = None
        try:
            queue_job_id = enqueue_job(job_id)
        except Exception as exc:
            error_text = f"{error_text}; retry enqueue failed: {exc}"
            with db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE jobs
                        SET status = 'failed', error_text = %s, finished_at = %s,
                            duration_ms = %s, retry_count = %s, last_error_at = %s,
                            log_file_key = %s, log_size_bytes = %s
                        WHERE id = %s
                        """,
                        (
                            error_text,
                            finished_at,
                            int(result.get("duration_ms") or 0),
                            retry_next,
                            finished_at,
                            log_key,
                            log_size,
                            job_id,
                        ),
                    )
                conn.commit()
            return

        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued', error_text = %s, finished_at = NULL,
                        duration_ms = %s, retry_count = %s, last_error_at = %s,
                        queue_job_id = %s, log_file_key = %s, log_size_bytes = %s,
                        last_heartbeat_at = %s
                    WHERE id = %s
                    """,
                    (
                        error_text,
                        int(result.get("duration_ms") or 0),
                        retry_next,
                        finished_at,
                        queue_job_id,
                        log_key,
                        log_size,
                        finished_at,
                        job_id,
                    ),
                )
            conn.commit()
        return

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'failed', error_text = %s, finished_at = %s,
                    duration_ms = %s, retry_count = %s, last_error_at = %s,
                    log_file_key = %s, log_size_bytes = %s, last_heartbeat_at = %s
                WHERE id = %s
                """,
                (
                    error_text,
                    finished_at,
                    int(result.get("duration_ms") or 0),
                    retry_next,
                    finished_at,
                    log_key,
                    log_size,
                    finished_at,
                    job_id,
                ),
            )
        conn.commit()


def to_job_item(row: dict[str, Any]) -> JobListItem:
    return JobListItem(
        id=str(row["id"]),
        status=row["status"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error_text=row["error_text"],
        duration_ms=row["duration_ms"],
    )


@app.on_event("startup")
def on_startup() -> None:
    init_runtime()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/register", response_model=TokenResponse)
def register(payload: RegisterRequest, request: Request) -> TokenResponse:
    apply_rate_limit("auth", request, AUTH_RATE_LIMIT_COUNT, AUTH_RATE_LIMIT_WINDOW_SEC)

    email = payload.email.lower().strip()
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Email already exists")

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (%s, %s, %s) RETURNING id",
                (email, hash_password(payload.password), datetime.now(timezone.utc)),
            )
            row = cur.fetchone()
        conn.commit()
        user_id = int(row["id"])

    tokens = issue_token_pair(user_id)
    return TokenResponse(**tokens)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, request: Request) -> TokenResponse:
    apply_rate_limit("auth", request, AUTH_RATE_LIMIT_COUNT, AUTH_RATE_LIMIT_WINDOW_SEC)

    user = get_user_by_email(payload.email)
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if bool(user.get("is_blocked", False)):
        raise HTTPException(status_code=403, detail="User is blocked")

    tokens = issue_token_pair(int(user["id"]))
    return TokenResponse(**tokens)


@app.post("/auth/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest) -> TokenResponse:
    claims = decode_token(payload.refresh_token)
    if claims.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user_id = int(claims.get("sub", 0))
    assert_active_refresh_token(payload.refresh_token, user_id)
    revoke_refresh_token(payload.refresh_token)

    tokens = issue_token_pair(user_id)
    return TokenResponse(**tokens)


@app.post("/auth/logout")
def logout(payload: LogoutRequest) -> dict[str, str]:
    claims = decode_token(payload.refresh_token)
    if claims.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    revoke_refresh_token(payload.refresh_token)
    return {"status": "ok"}


@app.post("/ai/suggest")
async def ai_suggest(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Анализирует BMP через Ollama и возвращает рекомендованные настройки векторизации."""
    image_bytes = await file.read()
    if len(image_bytes) > MAX_BMP_BYTES:
        raise HTTPException(status_code=413, detail="Файл слишком большой для анализа ИИ")

    result = await suggest_settings(image_bytes)
    return result.as_dict()


class AIChatRequest(BaseModel):
    messages: list[dict[str, str]]  # [{role: "user"|"assistant", content: "..."}]
    current_settings: dict[str, Any] | None = None
    image_metrics: dict[str, Any] | None = None


@app.post("/ai/chat")
async def ai_chat(payload: AIChatRequest) -> dict[str, Any]:
    """Ведёт диалог с LLM об настройках векторизации."""
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages не может быть пустым")
    return await chat_about_settings(
        messages=payload.messages,
        current_settings=payload.current_settings,
        image_metrics=payload.image_metrics,
    )


@app.get("/presets", response_model=list[PresetItem])
def list_presets(user: dict[str, Any] = Depends(auth_user)) -> list[PresetItem]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, settings_json, created_at, updated_at
                FROM presets
                WHERE owner_user_id = %s
                ORDER BY updated_at DESC, id DESC
                """,
                (int(user["id"]),),
            )
            rows = cur.fetchall()
    return [to_preset_item(row) for row in rows]


@app.post("/presets", response_model=PresetItem)
def create_preset(payload: PresetCreateRequest, user: dict[str, Any] = Depends(auth_user)) -> PresetItem:
    name = normalize_preset_name(payload.name)
    try:
        settings_obj = validate_settings_payload(payload.settings_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with db_connect() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO presets (owner_user_id, name, settings_json, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, name, settings_json, created_at, updated_at
                    """,
                    (
                        int(user["id"]),
                        name,
                        json.dumps(settings_obj),
                        datetime.now(timezone.utc),
                        datetime.now(timezone.utc),
                    ),
                )
            except psycopg.Error as exc:
                if getattr(exc, "sqlstate", None) == "23505":
                    raise HTTPException(status_code=409, detail="Preset with this name already exists") from exc
                raise
            row = cur.fetchone()
        conn.commit()
    return to_preset_item(row)


@app.put("/presets/{preset_id}", response_model=PresetItem)
def update_preset(
    preset_id: int,
    payload: PresetUpdateRequest,
    user: dict[str, Any] = Depends(auth_user),
) -> PresetItem:
    name = normalize_preset_name(payload.name)
    try:
        settings_obj = validate_settings_payload(payload.settings_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with db_connect() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    UPDATE presets
                    SET name = %s,
                        settings_json = %s,
                        updated_at = %s
                    WHERE id = %s AND owner_user_id = %s
                    RETURNING id, name, settings_json, created_at, updated_at
                    """,
                    (
                        name,
                        json.dumps(settings_obj),
                        datetime.now(timezone.utc),
                        int(preset_id),
                        int(user["id"]),
                    ),
                )
            except psycopg.Error as exc:
                if getattr(exc, "sqlstate", None) == "23505":
                    raise HTTPException(status_code=409, detail="Preset with this name already exists") from exc
                raise
            row = cur.fetchone()
        conn.commit()

    if row is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    return to_preset_item(row)


@app.delete("/presets/{preset_id}")
def delete_preset(preset_id: int, user: dict[str, Any] = Depends(auth_user)) -> dict[str, str]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM presets WHERE id = %s AND owner_user_id = %s RETURNING id",
                (int(preset_id), int(user["id"])),
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"status": "ok"}


@app.post("/jobs")
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    settings: str | None = Form(default=None),
    user: dict[str, Any] | None = Depends(auth_user_optional),
    guest_token: str | None = Header(default=None, alias="X-Guest-Token"),
) -> dict[str, str | None]:
    apply_rate_limit("upload", request, UPLOAD_RATE_LIMIT_COUNT, UPLOAD_RATE_LIMIT_WINDOW_SEC)

    raw = await file.read()
    ensure_bmp_file(file, raw)

    try:
        settings_obj = parse_settings_json(settings)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid settings JSON: {exc}") from exc

    job_id = str(uuid.uuid4())
    original_name = Path(file.filename or "").name or "file.bmp"
    upload_stem = normalize_filename_stem(original_name)
    upload_path = pick_unique_file_path(UPLOADS_DIR, upload_stem, ".bmp")
    upload_path.write_bytes(raw)
    input_key = os.path.relpath(upload_path, BASE_DIR)
    log_key = os.path.relpath((RESULTS_DIR / job_id / f"{job_id}.log"), BASE_DIR)
    owner_guest_token = None
    owner_user_id = None
    if user is None:
        owner_guest_token = normalize_guest_token(guest_token) or uuid.uuid4().hex
    else:
        owner_user_id = int(user["id"])

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (
                    id, user_id, guest_token, status, input_file_key, svg_file_key, log_file_key,
                    settings_json, error_text, created_at, started_at, finished_at, duration_ms,
                    retry_count, max_retries, input_size_bytes, queue_job_id
                ) VALUES (%s, %s, %s, 'queued', %s, NULL, %s, %s, NULL, %s, NULL, NULL, NULL, %s, %s, %s, NULL)
                """,
                (
                    job_id,
                    owner_user_id,
                    owner_guest_token,
                    input_key,
                    log_key,
                    json.dumps(settings_obj),
                    datetime.now(timezone.utc),
                    0,
                    max(0, JOB_MAX_RETRIES),
                    len(raw),
                ),
            )
        conn.commit()

    try:
        queue_job_id = enqueue_job(job_id)
    except Exception as exc:
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed', error_text = %s, finished_at = %s
                    WHERE id = %s
                    """,
                    (f"Queue enqueue failed: {exc}", datetime.now(timezone.utc), job_id),
                )
            conn.commit()
        raise HTTPException(status_code=503, detail="Queue is unavailable") from exc

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE jobs SET queue_job_id = %s WHERE id = %s", (queue_job_id, job_id))
        conn.commit()

    return {"id": job_id, "status": "queued", "guest_token": owner_guest_token}


def _where_owner_clause(
    user: dict[str, Any] | None,
    guest_token: str | None,
) -> tuple[str, tuple[Any, ...]]:
    if user is not None:
        return "user_id = %s", (int(user["id"]),)

    token = normalize_guest_token(guest_token)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication or X-Guest-Token required")
    return "guest_token = %s AND user_id IS NULL", (token,)


@app.get("/jobs", response_model=list[JobListItem])
def list_jobs(
    user: dict[str, Any] | None = Depends(auth_user_optional),
    guest_token: str | None = Header(default=None, alias="X-Guest-Token"),
) -> list[JobListItem]:
    owner_clause, owner_args = _where_owner_clause(user, guest_token)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM jobs WHERE {owner_clause} ORDER BY created_at DESC",
                owner_args,
            )
            rows = cur.fetchall()
    return [to_job_item(row) for row in rows]


@app.get("/jobs/{job_id}", response_model=JobDetails)
def get_job(
    job_id: str,
    user: dict[str, Any] | None = Depends(auth_user_optional),
    guest_token: str | None = Header(default=None, alias="X-Guest-Token"),
) -> JobDetails:
    owner_clause, owner_args = _where_owner_clause(user, guest_token)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM jobs WHERE id = %s AND {owner_clause}",
                (job_id, *owner_args),
            )
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")

    settings_obj = row["settings_json"]
    if isinstance(settings_obj, str):
        settings_obj = json.loads(settings_obj)

    return JobDetails(
        **to_job_item(row).model_dump(),
        settings_json=settings_obj,
    )


def _resolve_user_job_file(
    job_id: str,
    user: dict[str, Any] | None,
    guest_token: str | None,
    kind: str,
) -> Path:
    column = "svg_file_key" if kind == "svg" else "log_file_key"
    owner_clause, owner_args = _where_owner_clause(user, guest_token)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {column} AS key FROM jobs WHERE id = %s AND {owner_clause}",
                (job_id, *owner_args),
            )
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    key = row["key"]
    if not key:
        raise HTTPException(status_code=404, detail=f"{kind.upper()} is not available yet")

    path = (BASE_DIR / key).resolve()
    if not path.is_file() or BASE_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail=f"{kind.upper()} file not found")
    return path


def _resolve_user_job_file_key(
    job_id: str,
    user: dict[str, Any] | None,
    guest_token: str | None,
    column: str,
    kind: str,
) -> str:
    owner_clause, owner_args = _where_owner_clause(user, guest_token)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {column} AS key FROM jobs WHERE id = %s AND {owner_clause}",
                (job_id, *owner_args),
            )
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    key = row["key"]
    if not key:
        raise HTTPException(status_code=404, detail=f"{kind.upper()} is not available yet")
    return str(key)


@app.post("/jobs/{job_id}/download-link", response_model=SignedDownloadLinkResponse)
def create_signed_download_link(
    job_id: str,
    request: Request,
    kind: str = Query(pattern="^(svg|log)$"),
    ttl_seconds: int = Query(default=SIGNED_DOWNLOAD_DEFAULT_TTL_SEC, ge=30),
    single_use: bool = Query(default=True),
    user: dict[str, Any] | None = Depends(auth_user_optional),
    guest_token: str | None = Header(default=None, alias="X-Guest-Token"),
) -> SignedDownloadLinkResponse:
    bounded_ttl = min(ttl_seconds, SIGNED_DOWNLOAD_MAX_TTL_SEC)
    column = "svg_file_key" if kind == "svg" else "log_file_key"
    key = _resolve_user_job_file_key(job_id, user, guest_token, column, kind)
    token = create_signed_download_token(
        job_id=job_id,
        file_key=key,
        kind=kind,
        ttl_seconds=bounded_ttl,
        single_use=single_use,
    )

    base = str(request.base_url).rstrip("/")
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=bounded_ttl)
    return SignedDownloadLinkResponse(
        url=f"{base}/downloads/signed?token={token}",
        expires_at=expires_at,
        single_use=single_use,
    )


@app.get("/downloads/signed")
def download_signed(token: str = Query(..., min_length=10)) -> FileResponse:
    claims = decode_token(token)
    if claims.get("type") != "download":
        raise HTTPException(status_code=401, detail="Invalid signed link")

    kind = str(claims.get("kind", "")).lower()
    if kind not in {"svg", "log"}:
        raise HTTPException(status_code=401, detail="Invalid signed link")

    file_key = str(claims.get("file_key", "")).strip()
    job_id = str(claims.get("job_id", "")).strip()
    token_id = str(claims.get("jti", "")).strip()
    single_use = bool(claims.get("single_use", False))
    if not file_key or not job_id or not token_id:
        raise HTTPException(status_code=401, detail="Invalid signed link")

    if single_use:
        expires_raw = claims.get("exp")
        if not isinstance(expires_raw, (int, float)):
            raise HTTPException(status_code=401, detail="Invalid signed link")
        consume_single_use_download_token(token_id, float(expires_raw))

    path = resolve_keyed_path(file_key, kind)
    if kind == "svg":
        return FileResponse(path, media_type="image/svg+xml", filename=path.name)
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.get("/jobs/{job_id}/download/svg")
def download_svg(
    job_id: str,
    user: dict[str, Any] | None = Depends(auth_user_optional),
    guest_token: str | None = Header(default=None, alias="X-Guest-Token"),
) -> FileResponse:
    path = _resolve_user_job_file(job_id, user, guest_token, "svg")
    return FileResponse(path, media_type="image/svg+xml", filename=path.name)


@app.get("/jobs/{job_id}/download/log")
def download_log(
    job_id: str,
    user: dict[str, Any] | None = Depends(auth_user_optional),
    guest_token: str | None = Header(default=None, alias="X-Guest-Token"),
) -> FileResponse:
    path = _resolve_user_job_file(job_id, user, guest_token, "log")
    return FileResponse(path, media_type="text/plain", filename=path.name)


@app.get("/jobs/{job_id}/logs")
def read_logs(
    job_id: str,
    user: dict[str, Any] | None = Depends(auth_user_optional),
    guest_token: str | None = Header(default=None, alias="X-Guest-Token"),
) -> dict[str, str]:
    key = _resolve_user_job_file_key(job_id, user, guest_token, "log_file_key", "log")
    path = (BASE_DIR / key).resolve()
    if BASE_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="Log file not found")
    if not path.exists():
        return {"job_id": job_id, "logs": ""}

    text = path.read_text(encoding="utf-8", errors="replace")
    return {"job_id": job_id, "logs": text}


@app.get("/admin/jobs/stuck", response_model=list[AdminStuckJobItem], dependencies=[Depends(auth_admin)])
def admin_list_stuck_jobs(older_than_sec: int = Query(default=STUCK_JOB_SEC, ge=60)) -> list[AdminStuckJobItem]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, retry_count, max_retries, started_at, last_heartbeat_at, error_text
                FROM jobs
                WHERE status = 'running'
                  AND COALESCE(last_heartbeat_at, started_at, created_at) < NOW() - (%s || ' seconds')::interval
                ORDER BY started_at NULLS LAST, created_at
                """,
                (int(older_than_sec),),
            )
            rows = cur.fetchall()
    return [
        AdminStuckJobItem(
            id=str(r["id"]),
            status=str(r["status"]),
            retry_count=int(r["retry_count"] or 0),
            max_retries=int(r["max_retries"] or 0),
            started_at=r["started_at"],
            last_heartbeat_at=r["last_heartbeat_at"],
            error_text=r["error_text"],
        )
        for r in rows
    ]


@app.post("/admin/jobs/{job_id}/retry", dependencies=[Depends(auth_admin)])
def admin_retry_job(job_id: str, force: bool = Query(default=False)) -> dict[str, str]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Job not found")
            status = str(row["status"])
            if status == "running" and not force:
                raise HTTPException(status_code=409, detail="Job is running; use force=true to retry")

            cur.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    error_text = NULL,
                    finished_at = NULL,
                    started_at = NULL,
                    last_heartbeat_at = NULL,
                    worker_id = NULL
                WHERE id = %s
                """,
                (job_id,),
            )
        conn.commit()

    queue_job_id = enqueue_job(job_id)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE jobs SET queue_job_id = %s WHERE id = %s", (queue_job_id, job_id))
        conn.commit()
    return {"status": "queued", "id": job_id}


@app.post("/admin/jobs/recover-stuck", dependencies=[Depends(auth_admin)])
def admin_recover_stuck_jobs(older_than_sec: int = Query(default=STUCK_JOB_SEC, ge=60)) -> dict[str, int]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM jobs
                WHERE status = 'running'
                  AND COALESCE(last_heartbeat_at, started_at, created_at) < NOW() - (%s || ' seconds')::interval
                """,
                (int(older_than_sec),),
            )
            rows = cur.fetchall()
    recovered = 0
    for row in rows:
        job_id = str(row["id"])
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued',
                        error_text = COALESCE(error_text, 'Recovered after worker crash'),
                        started_at = NULL,
                        finished_at = NULL,
                        worker_id = NULL,
                        last_heartbeat_at = NULL
                    WHERE id = %s
                    """,
                    (job_id,),
                )
            conn.commit()

        queue_job_id = enqueue_job(job_id)
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE jobs SET queue_job_id = %s WHERE id = %s", (queue_job_id, job_id))
            conn.commit()
        recovered += 1
    return {"recovered": recovered}


@app.patch("/admin/users/{user_id}/block", dependencies=[Depends(auth_admin)])
def admin_block_user(user_id: int, payload: AdminBlockUserRequest) -> dict[str, str | bool]:
    reason = payload.reason.strip() if payload.reason else None
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET is_blocked = %s,
                    blocked_at = CASE WHEN %s THEN NOW() ELSE NULL END,
                    blocked_reason = CASE WHEN %s THEN %s ELSE NULL END
                WHERE id = %s
                RETURNING id
                """,
                (payload.blocked, payload.blocked, payload.blocked, reason, int(user_id)),
            )
            row = cur.fetchone()
        conn.commit()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": str(user_id), "blocked": payload.blocked}


@app.get("/admin/metrics", response_model=AdminMetricsResponse, dependencies=[Depends(auth_admin)])
def admin_metrics() -> AdminMetricsResponse:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS jobs_total,
                    COUNT(*) FILTER (WHERE status = 'running') AS jobs_running,
                    COUNT(*) FILTER (WHERE status = 'failed') AS jobs_failed,
                    COUNT(*) FILTER (WHERE status = 'success') AS jobs_success,
                    AVG(duration_ms)::float AS avg_duration_ms,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)::float AS p95_duration_ms,
                    COALESCE(SUM(input_size_bytes), 0) AS input_bytes_total,
                    COALESCE(SUM(svg_size_bytes), 0) AS svg_bytes_total,
                    COALESCE(SUM(log_size_bytes), 0) AS log_bytes_total
                FROM jobs
                """
            )
            row = cur.fetchone()

    queue_depth = 0
    try:
        queue_depth = int(get_job_queue().count)
    except Exception:
        queue_depth = -1

    return AdminMetricsResponse(
        jobs_total=int(row["jobs_total"]),
        jobs_running=int(row["jobs_running"]),
        jobs_failed=int(row["jobs_failed"]),
        jobs_success=int(row["jobs_success"]),
        avg_duration_ms=row["avg_duration_ms"],
        p95_duration_ms=row["p95_duration_ms"],
        input_bytes_total=int(row["input_bytes_total"]),
        svg_bytes_total=int(row["svg_bytes_total"]),
        log_bytes_total=int(row["log_bytes_total"]),
        queue_depth=queue_depth,
    )
