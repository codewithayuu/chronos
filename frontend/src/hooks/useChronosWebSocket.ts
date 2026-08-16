import { useEffect, useRef, useState, useCallback } from 'react';
import { PatientData, AlertData, SystemStatus } from '../types';

export const WS_URL = 'ws://localhost:8000/ws';
export const API_BASE = 'http://localhost:8000';
export const MAX_SPARKLINE_POINTS = 120;

export function useChronosWebSocket(url: string = WS_URL) {
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const sparklineData = useRef<Record<string, number[]>>({});

  const [patients, setPatients] = useState<Record<string, PatientData>>({});
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [connected, setConnected] = useState<boolean>(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [sparklines, setSparklines] = useState<Record<string, number[]>>({});

  const updateSparkline = useCallback((patientId: string, cesValue: number | null | undefined) => {
    if (cesValue == null) return;
    if (!sparklineData.current[patientId]) {
      sparklineData.current[patientId] = [];
    }
    const arr = sparklineData.current[patientId];
    arr.push(cesValue);
    if (arr.length > MAX_SPARKLINE_POINTS) {
      arr.shift();
    }
    setSparklines((prev) => ({
      ...prev,
      [patientId]: [...arr],
    }));
  }, []);

  const connect = useCallback(() => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) return;

    try {
      ws.current = new WebSocket(url);
    } catch (err) {
      console.error('[WS] Connection error:', err);
      scheduleReconnect();
      return;
    }

    ws.current.onopen = () => {
      setConnected(true);
      console.log('[WS] Connected to CHRONOS backend');
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
        reconnectTimeout.current = null;
      }
    };

    ws.current.onclose = () => {
      setConnected(false);
      console.log('[WS] Disconnected');
      scheduleReconnect();
    };

    ws.current.onerror = (err) => {
      console.warn('[WS] Connection issue (often expected during hot-reload):', err);
      ws.current?.close();
    };

    ws.current.onmessage = (event) => {
      let msg: any;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }

      switch (msg.event) {
        case 'initial_state': {
          const initial: Record<string, PatientData> = {};
          if (Array.isArray(msg.data)) {
            msg.data.forEach((p: any) => {
              if (p.patient_id) {
                initial[p.patient_id] = p;
                if (p.composite_entropy != null) {
                  updateSparkline(p.patient_id, p.composite_entropy);
                }
              }
            });
          }
          setPatients((prev) => ({ ...prev, ...initial }));
          break;
        }

        case 'initial_sparklines': {
          if (msg.data) {
            setSparklines((prev) => ({ ...prev, ...msg.data }));
            sparklineData.current = { ...sparklineData.current, ...msg.data };
          }
          break;
        }

        case 'patient_update': {
          const data = msg.data;
          if (!data?.patient_id) break;
          setPatients((prev) => {
            const existing = prev[data.patient_id];
            if (existing && data.vitals) {
              const mergedVitals = { ...data.vitals };
              for (const key of ['heart_rate', 'spo2', 'bp_systolic', 'bp_diastolic', 'resp_rate', 'temperature']) {
                if (mergedVitals[key] && mergedVitals[key].value == null && existing.vitals && existing.vitals[key]) {
                  mergedVitals[key] = { ...mergedVitals[key], value: existing.vitals[key].value };
                }
              }
              return {
                ...prev,
                [data.patient_id]: { ...data, vitals: mergedVitals },
              };
            }
            return {
              ...prev,
              [data.patient_id]: data,
            };
          });
          updateSparkline(data.patient_id, data.composite_entropy);
          break;
        }

        case 'new_alert': {
          if (!msg.data) break;
          setAlerts((prev) => [msg.data, ...prev].slice(0, 50));
          break;
        }

        case 'system_status': {
          setSystemStatus(msg.data);
          break;
        }

        case 'keepalive':
        case 'pong':
          break;

        default:
          console.log('[WS] Unknown event:', msg.event);
      }
    };

    function scheduleReconnect() {
      if (reconnectTimeout.current) return;
      reconnectTimeout.current = setTimeout(() => {
        reconnectTimeout.current = null;
        console.log('[WS] Attempting reconnect...');
        connect();
      }, 3000);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, updateSparkline]);

  useEffect(() => {
    connect();

    const pingInterval = setInterval(() => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.send('"ping"');
      }
    }, 25000);

    return () => {
      clearInterval(pingInterval);
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (ws.current) {
        ws.current.onclose = null;
        ws.current.close();
      }
    };
  }, [connect]);

  const acknowledgeAlert = useCallback(
    async (alertId: string, acknowledgedBy = 'Dr. Meera Ravenscroft') => {
      try {
        const res = await fetch(
          `${API_BASE}/api/v1/alerts/${alertId}/acknowledge`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ acknowledged_by: acknowledgedBy }),
          }
        );
        if (res.ok) {
          setAlerts((prev) =>
            prev.map((a) =>
              a.alert_id === alertId
                ? { ...a, acknowledged: true, acknowledged_by: acknowledgedBy }
                : a
            )
          );
        }
        return res.ok;
      } catch (err) {
        console.error('[API] Acknowledge error:', err);
        return false;
      }
    },
    []
  );

  return {
    patients,
    alerts,
    connected,
    systemStatus,
    sparklines,
    acknowledgeAlert,
  };
}
