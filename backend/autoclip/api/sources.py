"""Source ingestion endpoints."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import load as load_settings
from ..db import store
from ..pipeline import ingest
from .schemas import SourceOut, YouTubeIngestIn

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sources", tags=["sources"])

#: Uploads stream to disk in chunks rather than being buffered whole — source
#: videos routinely exceed available RAM.
UPLOAD_CHUNK = 4 * 1024 * 1024


@router.get("", response_model=list[SourceOut])
async def list_sources(limit: int = 50) -> list[SourceOut]:
    sources = await asyncio.to_thread(store.list_sources, limit)
    return [SourceOut.of(s) for s in sources]


@router.get("/{source_id}", response_model=SourceOut)
async def get_source(source_id: str) -> SourceOut:
    source = await asyncio.to_thread(store.get_source, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    return SourceOut.of(source)


@router.post("/youtube", response_model=SourceOut, status_code=201)
async def ingest_youtube(payload: YouTubeIngestIn) -> SourceOut:
    """Download a YouTube video and register it as a source."""
    if not ingest.is_youtube_url(payload.url):
        raise HTTPException(status_code=400, detail="That is not a YouTube URL.")

    settings = load_settings().ingest
    if payload.cookies_from_browser is not None:
        settings = settings.model_copy(
            update={"cookies_from_browser": payload.cookies_from_browser}
        )

    try:
        source = await asyncio.to_thread(ingest.ingest_youtube, payload.url, settings)
    except ingest.IngestError as exc:
        # 422 rather than 500: the request was well-formed, but the content
        # can't be fetched, and the hint tells the user what to change.
        raise HTTPException(
            status_code=422, detail={"message": str(exc), "hint": exc.hint}
        ) from exc

    await asyncio.to_thread(store.create_source, source)
    return SourceOut.of(source)


@router.post("/youtube-with-cookies", response_model=SourceOut, status_code=201)
async def ingest_youtube_with_cookies(
    url: str = Form(...),
    cookies_file: UploadFile = File(...),
) -> SourceOut:
    """Download a YouTube video using a temporary uploaded cookies.txt.

    The cookie file is never stored in AutoClip's runtime data or database.
    It exists only for the duration of this request.
    """
    if not ingest.is_youtube_url(url):
        await cookies_file.close()
        raise HTTPException(status_code=400, detail="That is not a YouTube URL.")

    filename = Path(cookies_file.filename or "")
    if filename.suffix.lower() != ".txt":
        await cookies_file.close()
        raise HTTPException(
            status_code=415,
            detail={
                "message": "The cookies file must be a .txt file.",
                "hint": "Export cookies in Netscape/Mozilla cookies.txt format.",
            },
        )

    max_cookie_bytes = 2 * 1024 * 1024
    staging_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt"
        ) as staging:
            staging_path = Path(staging.name)
            staging_path.chmod(0o600)

            total = 0
            header = b""

            while chunk := await cookies_file.read(64 * 1024):
                total += len(chunk)
                if total > max_cookie_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "message": "The cookies file is too large.",
                            "hint": "A normal YouTube cookies.txt file should be much smaller.",
                        },
                    )

                if len(header) < 4096:
                    header += chunk[:4096 - len(header)]

                staging.write(chunk)

        first_line = header.splitlines()[0].strip() if header.splitlines() else b""
        valid_headers = {
            b"# HTTP Cookie File",
            b"# Netscape HTTP Cookie File",
        }
        if first_line not in valid_headers:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid cookies.txt format.",
                    "hint": (
                        "Export the cookies as a Netscape/Mozilla cookies.txt file. "
                        "The first line must identify the HTTP Cookie File format."
                    ),
                },
            )

        settings = load_settings().ingest

        source = await asyncio.to_thread(
            ingest.ingest_youtube,
            url,
            settings,
            cookies_path=staging_path,
        )

    except HTTPException:
        raise
    except ingest.IngestError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc), "hint": exc.hint},
        ) from exc
    finally:
        await cookies_file.close()
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)

    await asyncio.to_thread(store.create_source, source)
    return SourceOut.of(source)


@router.post("/upload", response_model=SourceOut, status_code=201)
async def upload_source(file: UploadFile = File(...)) -> SourceOut:
    """Accept a media upload and register it as a source."""
    filename = Path(file.filename or "upload")
    suffix = filename.suffix.lower()
    if suffix not in ingest.ACCEPTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail={
                "message": f"{suffix or 'This file'} is not a supported format.",
                "hint": f"Accepted: {', '.join(sorted(ingest.ACCEPTED_SUFFIXES))}",
            },
        )

    # Stage to a temp file first so a failed or abandoned upload never leaves a
    # half-written file in the media store.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as staging:
        staging_path = Path(staging.name)
        try:
            while chunk := await file.read(UPLOAD_CHUNK):
                staging.write(chunk)
        except Exception as exc:
            staging_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Upload failed.") from exc

    try:
        source = await asyncio.to_thread(
            ingest.ingest_file, staging_path, move=True, title=filename.stem
        )
    except ingest.IngestError as exc:
        staging_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail={"message": str(exc), "hint": exc.hint}
        ) from exc
    finally:
        shutil.rmtree(staging_path.parent / staging_path.name, ignore_errors=True)

    await asyncio.to_thread(store.create_source, source)
    return SourceOut.of(source)
