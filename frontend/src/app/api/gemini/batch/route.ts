import { NextResponse } from 'next/server';
import { GoogleGenAI } from '@google/genai';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { patients } = body;

    if (!patients || !Array.isArray(patients)) {
      return NextResponse.json({ error: 'Patients array is required' }, { status: 400 });
    }

    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: 'GEMINI_API_KEY is not configured on the server.' },
        { status: 500 }
      );
    }

    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

    const systemPrompt = `You are CHRONOS-Gemini, an expert clinical AI reasoning agent. 
The user will provide you with a JSON array of patients and their current physiological state, including their Composite Entropy Score (CES) and vital signs.

Your task is to analyze each patient in the bundle and provide clinical recommendations for EACH patient.
You MUST respond with a valid JSON object matching this schema exactly, and NOTHING ELSE. Do not include markdown codeblocks. Return only raw JSON:
{
  "results": {
    "patient_id_1": {
      "recommendations": [
        {
          "intervention": "Short title of action",
          "success_rate": number (estimated 0-100),
          "evidence_summary": "Brief 1-sentence reasoning"
        }
      ],
      "narrative": "A concise 1-2 sentence clinical assessment."
    },
    "patient_id_2": {
      ...
    }
  }
}`;

    // Map patients to a lightweight summary to avoid blowing up the token context limit
    const patientSummaries = patients.map((p: any) => ({
      patient_id: p.patient_id,
      ces: p.composite_entropy,
      hr: p.vitals?.heart_rate?.value,
      sys: p.vitals?.bp_systolic?.value,
      dia: p.vitals?.bp_diastolic?.value,
      spo2: p.vitals?.spo2?.value,
      active_drugs: p.active_drugs || [],
    }));

    const interaction = await ai.interactions.create({
      model: "gemini-3.6-flash",
      input: `${systemPrompt}\n\nPatient Bundle Data:\n${JSON.stringify(patientSummaries, null, 2)}`,
    });

    return NextResponse.json({ result: interaction.output_text });
  } catch (error: any) {
    console.error('Gemini API Error (Batch):', error);
    
    if (error.message && error.message.includes('429')) {
      // Mock fallback for 429 errors
      return NextResponse.json({ error: 'Rate Limit Exceeded. Using fallback data.', isRateLimit: true }, { status: 429 });
    }

    return NextResponse.json(
      { error: error.message || 'An error occurred while calling the Gemini API' },
      { status: 500 }
    );
  }
}
