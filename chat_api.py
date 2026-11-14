"""HTTP API for conversation-scoped construction drawing questions.

Run locally with: uvicorn chat_api:app --host 127.0.0.1 --port 8000 --workers 1
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from conversation_pipeline import ConversationPipeline


SESSION_ROOT = Path(
    os.environ.get("STOA_SESSION_DIR", str(Path(tempfile.gettempdir()) / "stoa-conversations"))
).expanduser().resolve()
MAX_UPLOAD_BYTES = max(1, int(os.environ.get("STOA_MAX_UPLOAD_MB", "1024"))) * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024


@dataclass
class SessionRecord:
    pipeline: ConversationPipeline
    directory: Path


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    file_id: Optional[str] = None


app = FastAPI(title="Stoa Drawing QA", version="0.1.0")
_sessions: Dict[str, SessionRecord] = {}
_sessions_lock = threading.RLock()


def _session(session_id: str) -> SessionRecord:
    with _sessions_lock:
        record = _sessions.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Conversation not found or expired.")
    return record


def _safe_filename(filename: Optional[str]) -> str:
    raw = (filename or "upload").replace("\\", "/").split("/")[-1]
    safe = re.sub(r"[^\w.() -]+", "_", raw, flags=re.UNICODE).strip(" .")
    return (safe or "upload")[:180]


def _public_manifest(record: SessionRecord) -> List[dict]:
    private_keys = {"path"}
    return [
        {key: value for key, value in item.items() if key not in private_keys}
        for item in record.pipeline.manifest()
    ]


def _redact_source_paths(value, path_names: Dict[str, str]):
    if isinstance(value, dict):
        return {
            key: (path_names.get(str(child), Path(str(child)).name)
                  if key == "source" and isinstance(child, str)
                  else _redact_source_paths(child, path_names))
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_source_paths(item, path_names) for item in value]
    return value


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/conversations", status_code=201)
def create_conversation():
    conversation_id = uuid4().hex
    directory = SESSION_ROOT / conversation_id
    directory.mkdir(parents=True, exist_ok=False)
    pipeline = ConversationPipeline(
        count_model=os.environ.get("STOA_COUNT_MODEL"),
        geometry_model=os.environ.get("STOA_GEOMETRY_MODEL"),
    )
    with _sessions_lock:
        _sessions[conversation_id] = SessionRecord(pipeline=pipeline, directory=directory)
    return {"conversation_id": conversation_id, "files": []}


@app.post("/conversations/{conversation_id}/files")
def upload_files(conversation_id: str, files: List[UploadFile] = File(...)):
    record = _session(conversation_id)
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one file.")

    saved_paths: List[Path] = []
    created_directories: List[Path] = []
    try:
        for upload in files:
            filename = _safe_filename(upload.filename)
            per_file_directory = record.directory / uuid4().hex
            per_file_directory.mkdir(parents=True, exist_ok=False)
            created_directories.append(per_file_directory)
            destination = per_file_directory / filename
            saved_paths.append(destination)
            size = 0
            with destination.open("wb") as output:
                while chunk := upload.file.read(COPY_CHUNK_BYTES):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{filename} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB upload limit.",
                        )
                    output.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail=f"{filename} is empty.")
        manifests = record.pipeline.add_files(saved_paths)
    except HTTPException:
        for directory in created_directories:
            shutil.rmtree(directory, ignore_errors=True)
        raise
    except Exception as exc:
        for directory in created_directories:
            shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Could not register uploaded files: {exc}") from exc
    finally:
        for upload in files:
            upload.file.close()

    private_keys = {"path"}
    return {"files": [{key: value for key, value in item.items() if key not in private_keys}
                       for item in manifests]}


@app.get("/conversations/{conversation_id}/files")
def list_files(conversation_id: str):
    return {"files": _public_manifest(_session(conversation_id))}


@app.post("/conversations/{conversation_id}/questions")
def ask_question(conversation_id: str, request: QuestionRequest):
    record = _session(conversation_id)
    result = record.pipeline.answer(request.question, file_id=request.file_id)
    path_names = {str(entry.path): entry.path.name for entry in record.pipeline.files.values()}
    return _redact_source_paths(result, path_names)


@app.delete("/conversations/{conversation_id}")
def close_conversation(conversation_id: str):
    with _sessions_lock:
        record = _sessions.pop(conversation_id, None)
    if record is None:
        raise HTTPException(status_code=404, detail="Conversation not found or already closed.")
    record.pipeline.close()
    directory = record.directory.resolve()
    if directory.parent == SESSION_ROOT and directory.exists():
        shutil.rmtree(directory)
    return {"status": "closed", "conversation_id": conversation_id}
