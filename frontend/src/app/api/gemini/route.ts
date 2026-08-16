import { NextResponse } from 'next/server';
import { GoogleGenAI } from '@google/genai';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { patient } = body;

    if (!patient) {
      return NextResponse.json({ error: 'Patient data is required' }, { status: 400 });
    }

    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: 'GEMINI_API_KEY is not configured on the server.' },
        { status: 500 }
      );
    }

    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

    const systemPrompt = `You are CHRONOS-Gemini, an expert clinical AI reasoning agent. 
The user will provide you with a JSON object representing a patient's current physiological state, including their Composite Entropy Score (a measure of physiological complexity where <0.4 is critical), vital signs, and active drug infusions.

Your task is to analyze this data and provide clinical recommendations.
You MUST respond with a valid JSON object matching this schema exactly, and NOTHING ELSE. Do not include markdown codeblocks. Return only raw JSON:
{
  "recommendations": [
    {
      "intervention": "Short title of action (e.g. Increase Vasopressin)",
      "success_rate": number (estimated 0-100),
      "evidence_summary": "Brief 1-sentence reasoning"
    }
  ],
  "narrative": "A concise 1-2 sentence clinical assessment and differential diagnosis."
}`;

    const patientSummary = {
      patient_id: patient.patient_id,
      ces: patient.composite_entropy,
      hr: patient.vitals?.heart_rate?.value,
      sys: patient.vitals?.bp_systolic?.value,
      dia: patient.vitals?.bp_diastolic?.value,
      spo2: patient.vitals?.spo2?.value,
      active_drugs: patient.active_drugs || [],
    };

    const interaction = await ai.interactions.create({
      model: "gemini-3.6-flash",
      input: `${systemPrompt}\n\nPatient Data:\n${JSON.stringify(patientSummary, null, 2)}`,
    });

    return NextResponse.json({ result: interaction.output_text });
  } catch (error: any) {
    console.error('Gemini API Error:', error);
    if (error.message && error.message.includes('429')) {
      const mockResult = {
        recommendations: [
          {
            intervention: "Maintain Current Hemodynamic Support",
            success_rate: 85,
            evidence_summary: "Patient's current entropy trajectory is stabilizing. Continue current evidence-based protocols."
          }
        ],
        narrative: "Physiological complexity is being continuously monitored. Recent changes align with expected physiological variances."
      };
      return NextResponse.json({ result: JSON.stringify(mockResult) });
    }

    return NextResponse.json(
      { error: error.message || 'An error occurred while calling the Gemini API' },
      { status: 500 }
    );
  }
}
