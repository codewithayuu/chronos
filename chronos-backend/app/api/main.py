"""
FastAPI Application — Project Chronos API + WebSocket + Background Replay.

Start with:
    python run_server.py
Or:
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..config import load_config
from ..core.manager import PatientManager
from ..data.generator import DataGenerator
from ..data.replay import ReplayService
from .websocket import ConnectionManager
from .routes import create_router


# ──────────────────────────────────────────────
# Background Replay Task
# ──────────────────────────────────────────────

async def replay_loop(app: FastAPI):
    """
    Background task: replays demo data through the pipeline
    and broadcasts updates via WebSocket.

    Runs continuously. When dataset finishes, loops from the start.

    Performance strategy:
      At high speed (e.g., 60x), we process `speed` records per patient
      per second, but only compute entropy on the LAST record of each batch.
      Intermediate records are buffered into the sliding window without
      triggering the expensive SampEn computation.
    """
    manager: PatientManager = app.state.manager
    ws: ConnectionManager = app.state.ws_manager

    speed = int(os.environ.get("CHRONOS_SPEED", "60"))
    loop_replay = os.environ.get("CHRONOS_LOOP", "true").lower() == "true"

    print(f"[Replay] Generating demo dataset...")
    dataset = DataGenerator.generate_demo_dataset()

    replay = ReplayService(manager, manager.config)
    replay.load_cases(dataset)

    print(f"[Replay] Loaded {len(dataset)} patients, "
          f"max duration: {max(c.duration_minutes for c in dataset)} min")
    print(f"[Replay] Speed: {speed}x, Loop: {loop_replay}")

    drugs_given: set = set()
    tick = 0

    # Wait briefly for server to be fully ready
    await asyncio.sleep(1.0)

    while True:
        try:
            if not replay.is_running:
                if loop_replay:
                    print(f"[Replay] Dataset complete. Restarting...")
                    # Manual reset for replay
                    replay._current_minute = 0
                    replay._running = True
                    drugs_given.clear()
                    await asyncio.sleep(1.0)
                    continue
                else:
                    print(f"[Replay] Dataset complete. Stopping.")
                    await ws.broadcast_status({
                        "replay_finished": True,
                        "total_ticks": tick,
                    })
                    break

            # Process `speed` ticks per second (each tick = 1 minute of data)
            last_states = {}
            for step in range(speed):
                if not replay.is_running:
                    break

                # Use the tick() method which processes all patients for 1 minute
                replay.tick()
                
                # Get the latest states for broadcasting
                for case in replay._cases:
                    if replay._current_minute <= len(case.records):
                        state = manager.get_patient_state(case.patient_id)
                        if state:
                            last_states[case.patient_id] = state

            # Broadcast the latest state for each patient
            for pid, state in last_states.items():
                await ws.broadcast_patient_update(
                    state.model_dump(mode="json")
                )

                # If there's a new active alert, broadcast it separately
                if (
                    state.alert.active
                    and state.alert.severity.value in ("WARNING", "CRITICAL")
                ):
                    await ws.broadcast_alert({
                        "patient_id": pid,
                        "severity": state.alert.severity.value,
                        "message": state.alert.message,
                        "timestamp": state.timestamp.isoformat(),
                        "drug_masked": state.alert.drug_masked,
                    })

            tick += 1
            if tick % 30 == 0:
                progress = replay.progress * 100
                active_alerts = len(manager.get_active_alerts())
                print(
                    f"[Replay] Tick {tick}: {progress:.0f}% | "
                    f"{ws.client_count} WS clients | "
                    f"{active_alerts} active alerts"
                )
                await ws.broadcast_status({
                    "tick": tick,
                    "progress": round(replay.progress, 3),
                    "active_patients": len(last_states),
                    "active_alerts": active_alerts,
                    "ws_clients": ws.client_count,
                })

            # Sleep 1 second between ticks (real-time pace for broadcasts)
            await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            print("[Replay] Task cancelled.")
            break
        except Exception as e:
            print(f"[Replay] Error: {e}")
            await asyncio.sleep(2.0)


# ──────────────────────────────────────────────
# FastAPI Lifespan (startup/shutdown)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage background replay task lifecycle."""
    auto_replay = os.environ.get("CHRONOS_AUTO_REPLAY", "true").lower() == "true"

    if auto_replay:
        print("[Server] Starting background replay task...")
        task = asyncio.create_task(replay_loop(app))
        app.state.replay_task = task
    else:
        print("[Server] Auto-replay disabled. Use run_replay.py to feed data manually.")
        app.state.replay_task = None

    yield  # Server runs here

    # Shutdown
    if app.state.replay_task and not app.state.replay_task.done():
        app.state.replay_task.cancel()
        try:
            await app.state.replay_task
        except asyncio.CancelledError:
            pass
    print("[Server] Shutdown complete.")


# ──────────────────────────────────────────────
# App Factory
# ──────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Project Chronos",
        description=(
            "Entropy-based ICU Early Warning System. "
            "Real-time entropy analysis with drug awareness and evidence-based interventions."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize core components
    config = load_config()
    manager = PatientManager(config)
    ws_manager = ConnectionManager()

    app.state.manager = manager
    app.state.ws_manager = ws_manager

    # REST routes
    router = create_router()
    app.include_router(router, prefix="/api/v1")

    # ── WebSocket endpoint ──
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        Main WebSocket endpoint for real-time updates.

        Clients connect here to receive:
          - patient_update events (every ~1 second per patient)
          - new_alert events (when alerts fire)
          - system_status events (every ~30 seconds)
        """
        await ws_manager.connect(websocket)
        try:
            # Send initial state snapshot
            summaries = manager.get_all_summaries()
            if summaries:
                await websocket.send_json({
                    "event": "initial_state",
                    "data": [s.model_dump(mode="json") for s in summaries],
                })

            # Keep connection alive — listen for client messages
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(), timeout=30.0
                    )
                    # Client can send ping/pong or commands
                    if data == "ping":
                        await websocket.send_json({"event": "pong"})
                except asyncio.TimeoutError:
                    # Send keepalive
                    await websocket.send_json({"event": "keepalive"})
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)
        except Exception:
            await ws_manager.disconnect(websocket)

    @app.get("/")
    def root():
        return {
            "name": "Project Chronos",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "websocket": "/ws",
            "api": "/api/v1",
        }

    return app


# Module-level app instance
app = create_app()
