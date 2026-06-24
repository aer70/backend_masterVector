from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from bmp2svg_stable2 import CFG, bmp_to_svg_precise

ALLOWED_SETTINGS = set(CFG.keys())

ENUM_RULES: dict[str, set[str]] = {
    "method": {"rle", "contours"},
    "connect_op": {"close", "dilate", "none"},
    "stair_action": {"average", "remove13"},
    "stair3_policy": {"by_depth", "prefer_external", "prefer_internal", "auto"},
}

NULLABLE_KEYS = {"stair_max_distance_px", "tesseract_cmd"}

NUMERIC_BOUNDS: dict[str, tuple[float | None, float | None, bool]] = {
    "pre_scale_factor": (0.0, None, True),
    "min_area": (1.0, None, False),
    "min_group_area": (1.0, None, False),
    "color_merge_step": (1.0, None, False),
    "connect_radius_px": (0.0, None, False),
    "collinear_tol": (0.0, None, False),
    "stair_max_distance_px": (0.0, None, False),
    "stair_run_min_len": (1.0, None, False),
    "thin_max_width": (1.0, None, False),
    "thin_min_aspect": (1.0, None, False),
    "black_threshold": (0.0, 255.0, False),
    "ra90_max_edge_len_px": (0.0, None, False),
    "ra90_min_edges": (3.0, None, False),
    "stair2_min_len": (3.0, None, False),
    "stair2_exterior_min_len": (3.0, None, False),
    "stair3_min_edges": (4.0, None, False),
    "stair3_max_step_px": (0.0, None, False),
    "stair4_target_angle_deg": (0.1, 179.9, False),
    "stair4_angle_tol_deg": (0.0, 90.0, False),
    "stair4_min_polygon_vertices": (4.0, None, False),
    "simplify_epsilon": (0.0, None, False),
}


@dataclass
class JobResult:
    status: str
    svg_path: str | None
    log_path: str
    duration_ms: int
    error_message: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "svg_path": self.svg_path,
            "log_path": self.log_path,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
        }


class _LoggerWriter:
    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level
        self._buf = ""

    def write(self, message: str) -> int:
        if not message:
            return 0
        self._buf += message
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, line)
        return len(message)

    def flush(self) -> None:
        rest = self._buf.strip()
        if rest:
            self.logger.log(self.level, rest)
        self._buf = ""


def _build_job_logger(job_id: str, log_path: str) -> logging.Logger:
    logger_name = f"vectorization.job.{job_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [job=%(job_id)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    class _JobContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.job_id = job_id
            return True

    file_handler.addFilter(_JobContextFilter())
    logger.addHandler(file_handler)
    return logger


def _is_bool_type(value: Any) -> bool:
    return isinstance(value, bool)


def _is_int_type(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_float_like(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def _validate_numeric_bounds(key: str, value: Any) -> None:
    if key not in NUMERIC_BOUNDS or value is None:
        return
    min_value, max_value, strict_min = NUMERIC_BOUNDS[key]
    num = float(value)
    if min_value is not None:
        if strict_min and num <= min_value:
            raise ValueError(f"{key} must be > {min_value}")
        if not strict_min and num < min_value:
            raise ValueError(f"{key} must be >= {min_value}")
    if max_value is not None and num > max_value:
        raise ValueError(f"{key} must be <= {max_value}")


def _validate_sequence_type(key: str, value: Any, expected: tuple[Any, ...]) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be an array with {len(expected)} elements")
    if len(value) != len(expected):
        raise ValueError(f"{key} must contain exactly {len(expected)} elements")

    normalized: list[Any] = []
    for idx, (item, exp) in enumerate(zip(value, expected)):
        if _is_bool_type(exp):
            if not _is_bool_type(item):
                raise ValueError(f"{key}[{idx}] must be boolean")
            normalized.append(item)
        elif _is_int_type(exp):
            if not _is_int_type(item):
                raise ValueError(f"{key}[{idx}] must be integer")
            normalized.append(int(item))
        elif isinstance(exp, float):
            if not _is_float_like(item):
                raise ValueError(f"{key}[{idx}] must be number")
            normalized.append(float(item))
        elif isinstance(exp, str):
            if not isinstance(item, str):
                raise ValueError(f"{key}[{idx}] must be string")
            normalized.append(item)
        else:
            normalized.append(item)
    return tuple(normalized)


def validate_settings_payload(settings: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise ValueError("settings must be a JSON object")

    unknown_keys = sorted(set(settings.keys()) - ALLOWED_SETTINGS)
    if unknown_keys:
        raise ValueError(f"Unsupported settings keys: {', '.join(unknown_keys)}")

    normalized: dict[str, Any] = {}
    for key, value in settings.items():
        if value is None and key in NULLABLE_KEYS:
            normalized[key] = None
            continue

        expected = CFG[key]
        if _is_bool_type(expected):
            if not _is_bool_type(value):
                raise ValueError(f"{key} must be boolean")
            normalized[key] = value
        elif _is_int_type(expected):
            if not _is_int_type(value):
                raise ValueError(f"{key} must be integer")
            normalized[key] = int(value)
        elif isinstance(expected, float):
            if not _is_float_like(value):
                raise ValueError(f"{key} must be number")
            normalized[key] = float(value)
        elif isinstance(expected, str):
            if not isinstance(value, str):
                raise ValueError(f"{key} must be string")
            normalized[key] = value
        elif isinstance(expected, tuple):
            normalized[key] = _validate_sequence_type(key, value, expected)
        else:
            normalized[key] = value

        if key in ENUM_RULES and normalized[key] not in ENUM_RULES[key]:
            accepted = ", ".join(sorted(ENUM_RULES[key]))
            raise ValueError(f"{key} must be one of: {accepted}")

        _validate_numeric_bounds(key, normalized[key])

    return normalized


def parse_settings_json(raw_settings: str | None) -> dict[str, Any]:
    if not raw_settings:
        return {}

    parsed = json.loads(raw_settings)
    return validate_settings_payload(parsed)


def _resolve_unique_output_paths(output_dir: str, output_basename: str) -> tuple[str, str]:
    idx = 1
    while True:
        stem = output_basename if idx == 1 else f"{output_basename}_{idx}"
        svg_path = os.path.join(output_dir, f"{stem}.svg")
        log_path = os.path.join(output_dir, f"{stem}.log")
        if not os.path.exists(svg_path) and not os.path.exists(log_path):
            return svg_path, log_path
        idx += 1


class _JobProgressTracker:
    def __init__(
        self,
        logger: logging.Logger,
        job_id: str,
        input_path: str,
        svg_path: str,
        log_path: str,
        interval_sec: int = 60,
    ) -> None:
        self.logger = logger
        self.job_id = job_id
        self.input_path = input_path
        self.svg_path = svg_path
        self.log_path = log_path
        self.interval_sec = max(10, int(interval_sec))
        self.started_at = time.perf_counter()
        self._stage = "initializing"
        self._stage_started_at = self.started_at
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name=f"job-heartbeat-{job_id}")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def set_stage(self, stage: str, message: str | None = None) -> None:
        now = time.perf_counter()
        with self._lock:
            elapsed_stage = int(now - self._stage_started_at)
            self._stage = stage
            self._stage_started_at = now

        if message:
            self.logger.info("Stage changed to '%s' (%s)", stage, message)
        else:
            self.logger.info("Stage changed to '%s'", stage)

        if elapsed_stage > 0:
            self.logger.info("Previous stage duration: %ss", elapsed_stage)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            now = time.perf_counter()
            with self._lock:
                stage = self._stage
                stage_elapsed = int(now - self._stage_started_at)
            total_elapsed = int(now - self.started_at)
            self.logger.info(
                "Heartbeat: job is running | stage=%s | stage_elapsed=%ss | total_elapsed=%ss | input=%s | svg_target=%s | log=%s",
                stage,
                stage_elapsed,
                total_elapsed,
                self.input_path,
                self.svg_path,
                self.log_path,
            )


def process_job(
    input_path: str,
    settings: dict[str, Any],
    output_dir: str,
    output_basename: str,
    job_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    svg_path, log_path = _resolve_unique_output_paths(output_dir, output_basename)

    logger = _build_job_logger(job_id, log_path)
    logger.info("Job started")
    logger.info("Input file: %s", input_path)
    logger.info("Target SVG path: %s", svg_path)
    logger.info("Job log path: %s", log_path)

    heartbeat_interval_raw = os.getenv("BMP2SVG_LOG_HEARTBEAT_SEC", "60")
    try:
        heartbeat_interval_sec = int(heartbeat_interval_raw)
    except ValueError:
        heartbeat_interval_sec = 5
        logger.warning(
            "Invalid BMP2SVG_LOG_HEARTBEAT_SEC='%s'. Falling back to %ss.",
            heartbeat_interval_raw,
            heartbeat_interval_sec,
        )
    progress = _JobProgressTracker(
        logger=logger,
        job_id=job_id,
        input_path=input_path,
        svg_path=svg_path,
        log_path=log_path,
        interval_sec=heartbeat_interval_sec,
    )
    progress.start()
    progress.set_stage("settings-validation", "Checking and normalizing settings")

    try:
        settings = validate_settings_payload(settings)
    except ValueError as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        progress.set_stage("failed", "Settings validation failed")
        logger.error("Settings validation failed: %s", exc)
        return JobResult(
            status="failed",
            svg_path=None,
            log_path=log_path,
            duration_ms=duration_ms,
            error_message=str(exc),
        ).as_dict()

    effective_settings = {**CFG, **settings}

    stdout_writer = _LoggerWriter(logger, logging.INFO)
    stderr_writer = _LoggerWriter(logger, logging.ERROR)

    try:
        progress.set_stage("vectorization", "Running bmp_to_svg_precise")
        with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
            bmp_to_svg_precise(input_path, svg_path, **effective_settings)

        progress.set_stage("output-verification", "Checking that output SVG exists")
        if not os.path.isfile(svg_path):
            raise RuntimeError("SVG output was not created")

        duration_ms = int((time.perf_counter() - started) * 1000)
        progress.set_stage("finished", "Job finished successfully")
        logger.info("Job finished successfully in %sms", duration_ms)
        return JobResult(
            status="success",
            svg_path=svg_path,
            log_path=log_path,
            duration_ms=duration_ms,
            error_message=None,
        ).as_dict()
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        progress.set_stage("failed", "Unhandled processing error")
        logger.exception("Job failed: %s", exc)
        return JobResult(
            status="failed",
            svg_path=None,
            log_path=log_path,
            duration_ms=duration_ms,
            error_message=str(exc),
        ).as_dict()
    finally:
        progress.stop()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
