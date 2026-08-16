export interface VitalSign {
  value: number | null;
  unit?: string;
  trend?: 'rising' | 'falling' | 'stable';
}

export interface PatientVitals {
  heart_rate?: VitalSign;
  spo2?: VitalSign;
  bp_systolic?: VitalSign;
  bp_diastolic?: VitalSign;
  resp_rate?: VitalSign;
  temperature?: VitalSign;
  [key: string]: VitalSign | undefined;
}

export interface ActiveDrug {
  drug_name: string;
  dose?: number;
  unit?: string;
}

export interface Recommendation {
  intervention: string;
  success_rate: number;
  evidence_summary: string;
}

export interface PatientData {
  patient_id: string;
  name?: string;
  age?: number;
  room?: string;
  vitals?: PatientVitals;
  composite_entropy?: number;
  status?: string;
  active_drugs?: ActiveDrug[];
  drug_masked?: boolean;
  narrative_summary?: string;
  recommendations?: Recommendation[];
}

export interface AlertData {
  alert_id: string;
  severity: 'CRITICAL' | 'WARNING' | 'INFO';
  timestamp: string;
  patient_id: string;
  message: string;
  acknowledged: boolean;
  acknowledged_by?: string;
}

export interface SystemStatus {
  tick?: number;
  progress?: number;
  active_patients?: number;
  active_alerts?: number;
  ws_clients?: number;
  total_records_processed?: number;
  messages_per_second?: number;
}

export type ViewDensity = 'executive' | 'dense' | 'visualizers';
