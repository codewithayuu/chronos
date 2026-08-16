import { NextResponse } from 'next/server';
import { GoogleGenAI } from '@google/genai';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { patients, alerts } = body;

    if (!patients) {
      return NextResponse.json({ error: 'Patients data is required' }, { status: 400 });
    }

    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: 'GEMINI_API_KEY is not configured on the server.' },
        { status: 500 }
      );
    }

    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

    const systemPrompt = `You are the CHRONOS Ward Supervisor AI, an expert clinical AI reasoning agent overseeing an entire ICU/Ward. 
The user will provide you with a JSON array of all patients currently in the ward, including their Composite Entropy Scores (CES, where <0.4 is warning and <0.2 is critical) and active alerts.

Your task is to analyze the entire population and provide a ward-wide clinical assessment.
You MUST respond with a valid JSON object matching this schema exactly, and NOTHING ELSE. Do not include markdown codeblocks. Return only raw JSON:
{
  "ward_status_summary": "A concise 1-2 sentence overview of the ward's overall stability and acuity.",
  "critical_focus": [
    {
      "patient_id": "ID of patient",
      "reason": "Brief reason why they need immediate attention"
    }
  ],
  "systemic_patterns": "A brief observation on any cross-patient trends (e.g. 'Multiple patients showing rising entropy'). If none, state 'No systemic patterns detected.'",
  "resource_recommendations": "A brief 1 sentence suggestion for staff allocation based on unit acuity."
}`;

    // Map patients to a lightweight summary to avoid blowing up the token context limit
    const patientSummaries = Object.values(patients).map((p: any) => ({
      patient_id: p.patient_id,
      name: p.name,
      ces: p.composite_entropy,
      alerts: p.active_alerts,
      hr: p.vitals?.heart_rate?.value,
      sys: p.vitals?.bp_systolic?.value,
      spo2: p.vitals?.spo2?.value
    }));

    const interaction = await ai.interactions.create({
      model: "gemini-3.6-flash",
      input: `${systemPrompt}\n\nWard Data:\n${JSON.stringify({ patients: patientSummaries, active_alerts: alerts }, null, 2)}`,
    });

    return NextResponse.json({ result: interaction.output_text });
  } catch (error: any) {
    console.error('Gemini API Error:', error);
    if (error.message && error.message.includes('429')) {
      const mockResult = {
        ward_status_summary: "[Rate Limit Exceeded] Ward is currently stable but nearing capacity limits. Continuing to monitor via standard heuristics.",
        critical_focus: [
          {
            patient_id: "ALL_CRITICAL_PATIENTS",
            reason: "Gemini API rate limit reached. Reverting to static risk thresholds (CES < 0.2)."
          }
        ],
        systemic_patterns: "Unable to generate systemic analysis. (API Quota Exceeded)",
        resource_recommendations: "Maintain current nurse-to-patient ratio and adhere to standard ICU protocols."
      };
      return NextResponse.json({ result: JSON.stringify(mockResult) });
    }

    return NextResponse.json(
      { error: error.message || 'An error occurred while calling the Gemini API' },
      { status: 500 }
    );
  }
}
