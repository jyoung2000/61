import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.pipeline import register_ws_subscriber, unregister_ws_subscriber

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/jobs/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    register_ws_subscriber(job_id, websocket)

    # Replay the CURRENT job state immediately on connect. The live progress
    # broadcasts (including the terminal "complete") are fire-and-forget: a
    # client that connects (or reconnects after a drop on a long run) AFTER
    # the COMPLETE broadcast would otherwise sit forever on the last
    # in-progress message it happened to receive ("Exporting top clips…").
    # Sending the persisted status here means every connect reflects reality.
    try:
        from backend import database
        job = await database.load_job(job_id)
        if job is not None:
            status = job.status.value if hasattr(job.status, "value") else str(job.status)
            terminal = status in ("complete", "failed", "cancelled")
            # For a TERMINAL job, send a clean status line — NOT the stored
            # progress_message, which is often a stale intermediate like
            # "Exporting clips… (59/63)" (clip export emits progress out of order
            # under concurrency, so the last-persisted message isn't the final
            # one). Replaying that on a reconnect hours later showed a bogus
            # "Exporting clips (59/63)" event long after the run finished.
            if status == "complete":
                msg = "Analysis complete"
            elif status == "cancelled":
                msg = "Cancelled"
            elif status == "failed":
                msg = getattr(job, "progress_message", "") or "Analysis failed"
            else:
                msg = getattr(job, "progress_message", "") or ""
            await websocket.send_json({
                "type": "complete" if status == "complete" else "status",
                "status": status,
                "progress": int(getattr(job, "progress", 0) or (100 if terminal else 0)),
                "message": msg,
                # This is a state REPLAY on (re)connect, not a fresh live
                # transition. The client uses it to sync status/UI but must NOT
                # log a second "Analysis complete" stamped at reconnect time —
                # that's the "two different completion times" confusion (the
                # real completion already logged live, minutes earlier).
                "replay": True,
            })
    except Exception as e:  # best-effort — never block the connection
        logger.debug("ws connect status replay failed for %s: %s", job_id, e)

    try:
        while True:
            # Keep connection alive; client can send pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        unregister_ws_subscriber(job_id, websocket)
