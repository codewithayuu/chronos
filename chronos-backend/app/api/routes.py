"""
REST API Routes — All endpoints from the PRD.

All routes access the PatientManager via request.app.state.manager.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from ..models import VitalSignRecord, DrugEffect


class VitalIngestionResponse(BaseModel):
    status: str = "accepted"
    patient_id: str
    window_size: int


class DrugAdminRequest(BaseModel):
    drug_name: str
    drug_class: Optional[str] = None
    dose: Optional[float] = None
    unit: Optional[str] = None
    start_time: Optional[datetime] = None


class AlertAckRequest(BaseModel):
    acknowledged_by: str = "clinician"


def create_router() -> APIRouter:
    router = APIRouter()

    def _mgr(request: Request):
        return request.app.state.manager

    # ── Vital Signs ──

    @router.post("/vitals", response_model=VitalIngestionResponse)
    def ingest_vital(record: VitalSignRecord, request: Request):
        manager = _mgr(request)
        manager.process_vital(record)
        window = manager.entropy_engine.patients.get(record.patient_id)
        return VitalIngestionResponse(
            patient_id=record.patient_id,
            window_size=window.current_size if window else 0,
        )

    # ── Patients ──

    @router.get("/patients")
    def list_patients(request: Request):
        manager = _mgr(request)
        summaries = manager.get_all_summaries()
        return [s.model_dump() for s in summaries]

    @router.get("/patients/{patient_id}")
    def get_patient(patient_id: str, request: Request):
        manager = _mgr(request)
        state = manager.get_patient_state(patient_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        return state.model_dump()

    @router.get("/patients/{patient_id}/history")
    def get_patient_history(patient_id: str, request: Request, hours: int = 6):
        manager = _mgr(request)
        history = manager.get_patient_history(patient_id, hours)
        if not history:
            raise HTTPException(status_code=404, detail=f"No history for patient {patient_id}")
        return [s.model_dump() for s in history]

    @router.get("/patients/{patient_id}/drugs")
    def get_patient_drugs(patient_id: str, request: Request):
        manager = _mgr(request)
        drugs = manager.get_patient_drugs(patient_id)
        return [d.model_dump() for d in drugs]

    @router.post("/patients/{patient_id}/drugs")
    def add_drug(patient_id: str, drug_req: DrugAdminRequest, request: Request):
        manager = _mgr(request)
        drug = DrugEffect(
            drug_name=drug_req.drug_name,
            drug_class=drug_req.drug_class,
            dose=drug_req.dose,
            unit=drug_req.unit,
            start_time=drug_req.start_time or datetime.utcnow(),
        )
        manager.add_drug(patient_id, drug)
        return {"status": "recorded", "patient_id": patient_id, "drug": drug_req.drug_name}

    # ── Alerts ──

    @router.get("/alerts")
    def get_alerts(request: Request):
        manager = _mgr(request)
        return manager.get_all_alerts()

    @router.post("/alerts/{alert_id}/acknowledge")
    def acknowledge_alert(alert_id: str, ack: AlertAckRequest, request: Request):
        manager = _mgr(request)
        found = manager.acknowledge_alert(alert_id, ack.acknowledged_by)
        if not found:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        return {"status": "acknowledged", "alert_id": alert_id}

    # ── System ──

    @router.get("/system/health")
    def health_check(request: Request):
        manager = _mgr(request)
        health = manager.get_health()
        health["ws_clients"] = request.app.state.ws_manager.client_count
        return health

    return router
