# AEGIS
### Adaptive Entropy-Guided Intelligence System
*An Agentic ICU Intelligence Platform (formerly "Project Chronos")*

---

## Rebrand Notes (read this first)

This is Project Chronos, rebuilt around an explicit **AI agent layer**. Nothing about the underlying entropy math, drug modeling, or system architecture has changed — the core science is exactly as validated. What's new is *how the system talks to the clinician*.

In the original design, two components were the weakest link from a "why should a judge care" standpoint:

1. The **Evidence Engine** returned the same ranked intervention list to any two patients who happened to land near each other in feature space. It never looked at what *this specific patient* had already been through.
2. The **Clinical Narrative Generator** was explicitly template-based — the doc even said "not an LLM, for safety reasons." That was the right call for a v1, but it's also the most obvious thing to upgrade, and it's the one piece of the pitch that visibly says "AI" to a judge.

AEGIS turns both of these into real agents — LLM calls (Claude, via the Anthropic API) that sit **downstream of deterministic, structured data**, never upstream of it. The math still produces the numbers. The agent's only job is to reason over numbers it's handed and personalize the explanation. This section explains the four-agent architecture; everything after it is the original spec with names and the two relevant chapters updated.

### The Four-Agent Architecture

| Agent | What it does | Powered by |
|---|---|---|
| **Entropy Intelligence Engine** | Computes SampEn, MSE, and the Composite Entropy Score from raw vitals | Deterministic math (NumPy/SciPy) — no AI, and it shouldn't be |
| **Pharmacology Context Agent** | Filters expected drug effects, detects masked deterioration | Deterministic rules + PK simulation — no AI, and it shouldn't be |
| **Clinical Reasoning Agent** | Turns population evidence + *this patient's own trajectory* into a personalized, ranked recommendation | Retrieval (KD-Tree, unchanged) **+ Claude (LLM reasoning layer, new)** |
| **Narrative Agent** | Writes the plain-English handoff summary | **Claude (LLM generation layer, new)**, grounded against structured state |

The rule we hold ourselves to: **the AI never computes a clinical number.** Every number that reaches the AI is already computed by the deterministic pipeline (entropy scores, vitals, drug concentrations, historical success rates). The AI's job is strictly to explain, contextualize, and personalize — never to calculate. This is also why it's safe to say "AI-powered" honestly on a slide: the parts that must be exactly right (the math) are untouched by the LLM, and the parts where an LLM adds real value (turning a table of numbers into "here's what this means for *this* patient, given what's happened to them so far") are exactly where we added it.

---

# Part I: The Problem We Are Solving

## Chapter 1: Why ICUs Fail at Detecting Deterioration

### 1.1 The Core Problem

Every ICU bed in the world has a bedside monitor. That monitor watches heart rate, blood pressure, oxygen saturation, respiratory rate, and temperature. When any of these numbers crosses a fixed line (too high or too low), an alarm sounds. This is the "threshold alarm" paradigm, and it has been the foundation of ICU monitoring for over 40 years.

The problem is that this paradigm is fundamentally broken. Here is why.

**Problem 1: Alarms fire too late.** A patient's heart rate crossing 130 bpm is not the beginning of a crisis — it's the result of one that started hours ago. By the time the number crosses the threshold, the body has already exhausted its compensatory mechanisms.

**Problem 2: Alarms fire too often, and mostly for nothing.** Studies consistently show that 85–95% of ICU alarms are false or clinically irrelevant. A nurse managing two or three ICU beds hears over 200 alarms in a 12-hour shift. The result is "alarm fatigue" — real crises sound exactly like the 187 false alarms that preceded them.

**Problem 3: Alarms have no context.** A heart rate of 55 bpm triggers a low-heart-rate alarm — unless the patient received metoprolol 30 minutes ago, in which case 55 is expected and not dangerous. The monitor doesn't know this. Conversely, a vasopressor can hold blood pressure at a "normal" 120/80 while the underlying cardiovascular system is collapsing underneath it. The monitor can't see that either.

**Problem 4: Alarms tell you WHAT, not HOW.** A heart rate of 88 bpm is "normal" — but whether that 88 is rich and variable (healthy) or flat and robotic (dying) is invisible to a threshold. Two patients can share identical vital sign numbers while one is stable and the other is hours from cardiac arrest.

### 1.2 The Human Cost

These failures aren't theoretical. "Failure to rescue" — missing a deteriorating patient — is one of the leading causes of preventable hospital mortality, contributing to an estimated 400,000 preventable deaths per year worldwide.

### 1.3 What AEGIS Does Differently

AEGIS replaces the binary alarm paradigm with a continuous intelligence layer built on three core ideas — the first two computed by deterministic engines, the third handled by an actual reasoning agent:

1. **Entropy Intelligence** — Instead of "is this number too high or too low?", AEGIS asks "is this signal losing its natural complexity?" This detects deterioration at the pattern level, typically 4–8 hours before threshold alarms fire.
2. **Pharmacology Context** — Every vital sign change is cross-referenced against the patient's active drug list, filtering expected effects and exposing cases where drugs are masking real decline.
3. **Agentic Clinical Reasoning** — Instead of a static ranked list, AEGIS's Clinical Reasoning Agent looks at what worked for similar patients *and* at this patient's own history in this admission, then explains, in plain language, why a given intervention fits — personalized per patient, not templated per population bucket.

---

# Part II: The Science Behind the System

## Chapter 2: Entropy — The Mathematical Foundation

*(Unchanged from the original spec — this is the deterministic core and stays that way.)*

### 2.1 What Entropy Means in This Context

A healthy human body is a remarkably complex system. The heart does not beat like a metronome — there are subtle beat-to-beat variations that are the signature of a healthy autonomic nervous system constantly adjusting to breathing, blood pressure, hormones, and hundreds of other inputs. When the body begins to fail, these variations disappear and the signal becomes regular, predictable, and simple.

**Core insight:** a loss of physiological complexity is an early marker of deterioration, and it can be detected mathematically using entropy measures — often while the vital sign values themselves are still "normal."

### 2.2 Sample Entropy (SampEn) — The Primary Measure

Sample Entropy (Richman & Moorman, 2000) quantifies irregularity in a time series by asking: "given that I've seen a short pattern before, how likely is the next point to also match?" Regular signals → high match probability → low SampEn. Complex, healthy signals → low match probability → high SampEn.

Given a time series 𝑢(1), 𝑢(2), …, 𝑢(𝑁):

**Step 1 — Parameters.** Embedding dimension 𝑚 = 2. Tolerance 𝑟 = 0.2 × SD of the series (makes the measure scale-adaptive across vitals with different natural ranges).

**Step 2 — Template vectors.** x𝑚(𝑖) = [𝑢(𝑖), 𝑢(𝑖+1), …, 𝑢(𝑖+𝑚−1)] for 𝑖 = 1 … 𝑁−𝑚+1.

**Step 3 — Match count at length m.** Two templates match if the max absolute difference between corresponding elements is ≤ 𝑟. Compute 𝐵𝑚 = probability that any two length-𝑚 templates match.

**Step 4 — Match count at length m+1.** Repeat with 𝑚+1 = 3 to get 𝐴𝑚.

**Step 5 — SampEn.**

SampEn(𝑚, 𝑟, 𝑁) = −ln(𝐴𝑚 / 𝐵𝑚)

- SampEn = 0 → perfectly regular signal.
- Typical physiological range: ~0.1 (very regular, concerning) to ~2.5 (very complex, healthy).

| Parameter | Value | Reasoning |
|---|---|---|
| 𝑚 | 2 | Standard in clinical literature; balances sensitivity and reliability |
| 𝑟 | 0.2 × SD | Makes the measure adaptive across vitals with different natural scales |
| 𝑁 | 300 | 5 hours at 1 sample/min; within the recommended 10𝑚–30𝑚 range for reliable estimates |

### 2.3 Multi-Scale Entropy (MSE) — The Deeper View

Introduced by Costa, Goldberger & Peng (2005). Physiological systems operate at multiple time scales — deterioration often shows up first at longer scales before it's visible minute-to-minute.

**Coarse-graining:** for scale factor 𝜏, average consecutive non-overlapping blocks of 𝜏 points into a new, shorter series, then compute SampEn on each coarse-grained series. The result is a 10-value profile: MSE = [SampEn(𝜏=1), …, SampEn(𝜏=10)].

- **Healthy patient:** high SampEn, stable across scales.
- **Deteriorating patient:** SampEn drops at higher scales — long-range regulatory structure breaking down first.
- **Acute crisis:** low SampEn at *all* scales.

Scales 5–10 represent 5–10 minutes of real-time behavior and often fail before minute-to-minute complexity does — this is where MSE buys the earliest warning.

### 2.4 Composite Entropy Score (CES) — The Single Number

CES = Σᵢ 𝑤ᵢ × normalize(SampEnᵢ), where normalization maps each vital's SampEn to 0–1 using min/max observed across the MIMIC-IV training population.

| Vital Sign | Weight | Reasoning |
|---|---|---|
| Heart Rate | 0.25 | Strongest documented correlation between entropy loss and mortality |
| Respiratory Rate | 0.20 | Complexity changes are often the *earliest* sign, sometimes 6–8h ahead |
| Systolic BP | 0.20 | Entropy drops strongly associated with hemodynamic instability, sepsis onset |
| SpO2 | 0.15 | Important but limited dynamic range (90–100) makes entropy less reliable |
| Diastolic BP | 0.10 | Correlated with systolic; down-weighted to avoid double-counting |
| Temperature | 0.10 | Changes slowly, longest lag — still valuable for sepsis patterns |

| CES Range | Severity | Meaning |
|---|---|---|
| 0.60 – 1.00 | NONE (Green) | Normal complexity, regulatory systems functioning well |
| 0.40 – 0.59 | WATCH (Yellow) | Reduced complexity — worth noting, not yet alarming |
| 0.20 – 0.39 | WARNING (Orange) | High probability of a deterioration event within 4–8h without intervention |
| 0.00 – 0.19 | CRITICAL (Red) | Near-complete loss of variability — imminent crisis likely |

### 2.5 Trend Detection — The Trajectory Matters

A CES of 0.45 could mean "naturally lower baseline" or "actively crashing from 0.85." We fit a linear regression over the last 360 CES values (6 hours):

CES(𝑡) = 𝛽₀ + 𝛽₁𝑡 + 𝜖

| Slope | Trend | Meaning |
|---|---|---|
| 𝛽₁ > +0.001/min | Rising ↑ | Complexity increasing, possible recovery |
| −0.001 ≤ 𝛽₁ ≤ +0.001 | Stable → | No significant change |
| 𝛽₁ < −0.001/min | Falling ↓ | Concerning trajectory, even if current CES is still green |

### 2.6 The Sliding Window — How Entropy Stays Current

Entropy is computed on a sliding 300-point window, recomputed every minute as the newest point enters and the oldest drops off. This creates natural smoothing — entropy changes gradually rather than jumping, making sustained trends easy to distinguish from blips.

**Data quality gate:** at least 80% of the window (240/300 points) must be valid, or the entropy computation is flagged "unreliable" and the dashboard shows "Insufficient Data." Gaps ≤5 minutes are forward-filled; longer gaps are excluded from the SampEn calculation entirely (not fabricated).

**Warmup:** during the first 300 minutes after a patient connects, the dashboard shows "Calibrating…" — no entropy score, no alerts, until there's enough real data to trust.

---

## Chapter 3: The Pharmacology Context Agent

*(Formerly "Drug Awareness." The algorithm is unchanged — this is deterministic rule logic, not an LLM — but we've renamed it to reflect its role as one of the four cooperating agents, and it's the layer that feeds the Clinical Reasoning Agent its drug context downstream.)*

### 3.1 Why Drug Context Is Essential

Consider: a patient's heart rate is 95 bpm. The team gives metoprolol. Over 30 minutes, HR drops to 65 and its entropy drops too — because the drug is overriding the heart's natural variability. Without drug context, the entropy system would flag "deteriorating" — a false alarm that erodes clinician trust.

Now the opposite: a patient on norepinephrine shows a stable-looking 115/75 mmHg — but the *entropy* of that BP signal is falling. The value looks fine because the drug is artificially maintaining it, but the underlying cardiovascular system is losing its ability to self-regulate. Without drug context, this masked deterioration goes unflagged.

The Pharmacology Context Agent has two jobs:
1. **Suppress false alarms** when entropy changes are explained by known drug effects.
2. **Expose masked deterioration** when drugs are artificially stabilizing values while true complexity declines.

### 3.2 The Drug Lookup Table

For each of ~15–20 ICU drugs, we store: drug name, class, expected effect direction/magnitude on HR/BP/RR/SpO2, onset and duration, and whether the drug class inherently reduces entropy.

Drug classes covered: **Beta-Blockers** (lower HR/BP, reduce HR entropy directly), **Vasopressors** (raise BP, can mask hemodynamic instability), **Sedatives** (lower HR/BP/RR, reduce entropy broadly via CNS depression), **Opioids** (lower RR/HR, mask respiratory deterioration specifically), **Paralytics** (eliminate spontaneous respiratory effort — RR entropy becomes meaningless), **Antiarrhythmics** (suppress cardiac rhythm complexity directly), **Inotropes** (increase HR, can *increase* HR entropy in some cases).

### 3.3 The Drug Adjustment Algorithm

**Phase 1 — Identify affected vitals** for each active drug.

**Phase 2 — Check if observed changes match expected effects.** actual_change = current_value − baseline (avg of the 30 min pre-dose). If direction matches and magnitude is within 30% tolerance of the expected magnitude, the change is "explained."

**Phase 3 — Adjust CES.** For explained vitals, halve that vital's CES weight, then renormalize so weights still sum to 1.0.

**Phase 4 — Detect drug masking** (the critical, novel part). If (a) values look stable/normal, (b) an active drug would be expected to support those values, and (c) that vital's *entropy* is falling — flag `drug_masked: true`, do **not** reduce that vital's weight, and annotate: "Drug may be masking decline in [vital]." Severity is maintained or elevated.

**Phase 5 — Duration expiry.** Once a drug's onset+duration window passes, its effects drop out of the adjustment entirely.

### 3.4 Raw vs. Adjusted Display

The dashboard always shows both **Raw CES** (what the body is actually doing) and **Adjusted CES** (what's happening that drugs *don't* explain). If raw = 0.32 and adjusted = 0.55, a large chunk of the complexity loss is expected medication effect. If both are 0.32, the drugs aren't explaining it — something else is going on. This raw/adjusted pair is exactly the structured input the Clinical Reasoning Agent uses downstream — it never has to guess whether a drug is involved, it's handed the answer.

---

## Chapter 4: The Clinical Reasoning Agent — Personalized, Evidence-Grounded Guidance

*(Formerly "The Evidence Engine." This is the first of the two chapters that gets the full AI-agent upgrade.)*

### 4.1 The Core Idea, Evolved

The original design answered "what should I do?" purely with population statistics: find the 50 most similar historical patients, rank interventions by success rate, show the top 5. That's a genuinely useful baseline — and it's still exactly how the *retrieval* step works, unchanged (see 4.2–4.5 below). But two identical-looking patients with the same nearest neighbors would get the identical list, worded identically, regardless of anything specific to either of them.

The Clinical Reasoning Agent keeps the retrieval step exactly as-is and adds a reasoning step on top: it hands the retrieved population evidence to Claude **along with this specific patient's own longitudinal record from this admission** — their personal entropy trajectory, which interventions have already been tried on *them* and how they responded, their individual drug tolerance patterns, and any prior deterioration events earlier in this same stay. The output is a recommendation that reads differently for two patients who share the same nearest neighbors but have different histories — because it should.

### 4.2 Building the Historical Case Database

*(Unchanged.)* Source: MIMIC-IV, 300,000+ ICU stays. We pre-extract a feature vector for every stay containing a documented deterioration event — cardiac arrest/code blue, vasopressor initiation or escalation, emergency intubation, rapid response activation, or transfer to a higher level of care — capturing the patient's state in the hours *before* the event.

### 4.3 The Feature Vector

25 features per case: 3 demographic (age, sex, weight), 12 vital-sign summary features (mean, SD, current SampEn for each of HR/BP_sys/RR/SpO2 over the preceding 6h), 3 entropy features (current CES, 6h entropy slope, hours below WATCH threshold), and 7 binary drug-class flags. Each case also stores (not as search features) the interventions applied afterward, whether they succeeded (entropy stopped declining within 2h *and* the patient survived ≥24h post-intervention), and overall outcome.

### 4.4 The Search Algorithm

*(Unchanged — this is the deterministic retrieval step and it stays deterministic.)*

All 25 features are z-score standardized. A KD-Tree is pre-built over the standardized historical vectors (scikit-learn `BallTree`/`KDTree`). At query time: build the current patient's 25-feature vector, standardize it with the training set's saved 𝜇ⱼ/𝜎ⱼ, and retrieve the **K = 50** nearest neighbors by Euclidean distance. K = 50 balances statistical reliability against true similarity — too few (5–10) gives unreliable statistics, too many (500) dilutes what "similar" even means. If the nearest neighbor is farther than a calibrated distance threshold, the system reports "Insufficient similar cases for recommendations" rather than guessing.

### 4.5 Intervention Ranking (the raw evidence table)

Group the 50 neighbors by intervention type (vasopressor initiation/escalation, fluid bolus, intubation, blood product transfusion, antibiotics, sedation adjustment, diuretics). For each group compute success rate, case count, and median time-to-recovery. Drop any intervention with fewer than 5 cases. Rank by success rate and keep the top 5. **This numeric table is computed once, deterministically, and never touched by the LLM — it is the ground truth the agent is required to reason from.**

### 4.6 The Reasoning Step — What the Agent Actually Does

When the Clinical Reasoning Agent is invoked (any time adjusted CES crosses into WATCH or below), it assembles a structured context object — **not free text** — containing:

- The ranked intervention table from 4.5, verbatim (names, success rates, case counts, response times)
- The current patient's raw and adjusted CES, trend slope, and per-vital SampEn breakdown
- Active drugs and any `drug_masked` flags from the Pharmacology Context Agent
- **This patient's own history in the current admission:** prior CES excursions and how each resolved, prior interventions tried on this patient and their observed response, and any drug-tolerance pattern specific to them (e.g., an unusually strong or weak response to a prior fluid bolus)

This object — and only this object — is passed to Claude with a system prompt that constrains it to two things: (1) reason about *which* of the population-evidence interventions best fits given the patient's own history, and (2) explain *why*, in plain clinical language, referencing specific numbers from the object it was given. The prompt explicitly forbids introducing any drug, dose, or statistic that isn't present in the context object — the model is reasoning over facts it's handed, not recalling medical knowledge from training.

**Grounding check:** before the output is shown, every number and drug name in the agent's response is programmatically extracted and checked against the structured context object. Anything that doesn't match is stripped and the affected sentence falls back to a plain templated version. This is the same anti-hallucination guardrail used by the Narrative Agent in Chapter 8 — it's a shared safety component, not duplicated logic.

**Example — same nearest neighbors, two different patients:**

> Patient A (no prior interventions this admission): *"Based on 50 similar historical cases, vasopressor dose adjustment showed the highest stabilization rate (78%, n=39), typically resolving within 45 minutes. Fluid bolus was second (61%, n=28)."*

> Patient B (same 50 neighbors, but already received a fluid bolus 3 hours ago on this admission with a strong, fast response): *"Population evidence favors vasopressor adjustment (78%, n=39), but this patient responded unusually quickly to a fluid bolus earlier today (CES recovered within 20 minutes, faster than the 45-minute historical median). Given that individual response, a second fluid bolus may be worth considering before escalating to vasopressors — though the population data still favors the latter if fluids don't resolve this episode."*

Same retrieval, same numbers available — genuinely different, patient-specific guidance.

### 4.7 Why This Is Personalization, Not Generic Advice

The distinction matters for the pitch: a nearest-neighbor lookup alone is population-level — it tells you what worked for people *like* this patient. Feeding that evidence, plus the patient's own trajectory, into a reasoning step that's required to reconcile the two is what makes the output actually about *this* patient. It's the difference between "here's a leaderboard" and "here's what I'd do for you, specifically, given what's already happened."

### 4.8 The Critical Disclaimer

Unchanged, and non-negotiable — the agent is required to include it verbatim on every recommendation it produces:

> *"These suggestions are based on historical pattern analysis and are provided as decision support only. All clinical decisions remain the sole responsibility of the treating physician."*

The system is a decision-support tool, not a decision-making tool. The AI reasons and explains; a human decides. This distinction doesn't change just because the explanation layer got smarter.

---

# Part III: Extended Features

## Chapter 5: Split-Screen Traditional vs. AEGIS View

*(Unchanged in mechanism — this is a pure visualization feature, no computation.)*

The most persuasive demo device is showing, side by side, what a traditional monitor displays versus what AEGIS displays for the *same patient at the same moment* — especially during "silent decline," when values are still normal but complexity is collapsing.

**Left half — Traditional Monitor View:** HR 82, BP 118/76, SpO2 97%, RR 16, Temp 37.1°C — all green, no alarms, "patient is fine."

**Right half — AEGIS View:** same numbers, plus per-vital SampEn, a CES gauge at 0.38 (WARNING), a 4-hour declining trend, a collapsed MSE profile at higher scales, an alert banner ("Entropy declining — predicted event in ~5.2 hours"), any drug-masking note, and the Clinical Reasoning Agent's personalized suggestion panel.

Standard traditional thresholds used for the left half:

| Vital Sign | Low Alarm | High Alarm |
|---|---|---|
| Heart Rate | < 50 bpm | > 120 bpm |
| SpO2 | < 90% | — |
| Systolic BP | < 90 mmHg | > 180 mmHg |
| Diastolic BP | < 50 mmHg | > 110 mmHg |
| Respiratory Rate | < 8 /min | > 30 /min |
| Temperature | < 35.0°C | > 39.5°C |

---

## Chapter 6: Clinical Scores — NEWS2 and qSOFA

### 6.1 Purpose

Entropy is the primary engine, but clinicians already think in NEWS2 and qSOFA. Displaying both alongside CES speaks the clinician's language *and* shows exactly where entropy catches problems these established scores miss.

### 6.2 NEWS2 (National Early Warning Score 2)

| Vital Sign | 3 | 2 | 1 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| Respiratory Rate | ≤8 | — | 9–11 | 12–20 | 21–24 | — | ≥25 |
| SpO2 (Scale 1) | ≤91 | 92–93 | 94–95 | ≥96 | — | — | — |
| Systolic BP | ≤90 | 91–100 | 101–110 | 111–219 | — | — | ≥220 |
| Heart Rate | ≤40 | — | 41–50 | 51–90 | 91–110 | 111–130 | ≥131 |
| Temperature | ≤35.0 | — | 35.1–36.0 | 36.1–38.0 | 38.1–39.0 | ≥39.1 | — |
| Consciousness | — | — | — | Alert | — | — | CVPU* |

*\*CVPU = Confusion, Voice response, Pain response, Unresponsive. AEGIS's simulated feed has no consciousness assessment, so this defaults to "Alert" (0) — a documented limitation.*

Total NEWS2 (0–20): **0–4 Low** (routine monitoring) · **5–6, or any single 3** — **Medium** (urgent review) · **7+ High** (emergency response).

**How AEGIS uses it:** in the ideal hero case, NEWS2 sits at 2–3 (looks fine by traditional scoring) while CES drops to 0.35 (WARNING) — the side-by-side contrast is the demo's clearest single visual.

### 6.3 qSOFA (Quick Sequential Organ Failure Assessment)

| Criterion | Scores 1 If |
|---|---|
| Respiratory Rate | ≥ 22 /min |
| Systolic BP | ≤ 100 mmHg |
| Altered Mental Status | GCS < 15 (not available in simulation; defaults to 0) |

Total range 0–3; a score ≥2 suggests possible sepsis and warrants further investigation.

### 6.4 Entropy-Enhanced Clinical Score

An experimental "NEWS2+E" score: if CES trend is falling (slope < −0.001/min) *and* current CES < 0.50, add one point to NEWS2. Clearly labeled "Experimental / Research" on the dashboard — this has no formal clinical validation, it's a demonstration of the integration concept.

---

## Chapter 7: Alarm Fatigue Analytics

### 7.1 Purpose

"We reduce alarms" is a claim. Showing the actual numbers, live, is proof. This dashboard quantifies exactly how many alarms AEGIS prevents and whether the remaining ones are meaningful.

### 7.2 Metrics Displayed

**Total alarms, side by side:** "Traditional Alarms (this session): 847" vs. "AEGIS Alerts (this session): 23." Traditional alarms count every threshold crossing on every incoming record, exactly as a real bedside monitor would — the same vital crossing the same threshold 5 times in 10 minutes is 5 alarms. AEGIS alerts are the actual output of the entropy + pharmacology pipeline, which changes slowly by design and so naturally produces far fewer.

**Alarm reduction %:** Reduction = (1 − AEGIS Alerts / Traditional Alarms) × 100%. Target: **≥85% reduction.**

**True positive rate comparison** (for cases with known ground truth): what fraction of real deterioration events were preceded by a threshold alarm within 2 hours, vs. by an AEGIS alert within 8 hours.

**Lead time comparison:** traditional alarms typically give 0–30 minutes of warning (they fire only once a threshold is crossed); AEGIS targets 4–8 hours.

**False alarm breakdown:** a bar chart of *why* traditional alarms fired falsely — drug-induced change, sensor artifact, transient fluctuation, or true positive — computed by retroactively checking whether an active drug explains the change, whether the vital self-corrected within 2 minutes, or whether a real clinical event followed within 4 hours.

### 7.3 Real-Time Counter

A running counter — "Traditional alarms suppressed by AEGIS this session: [incrementing]" — increments every time a change that would have triggered a traditional alarm is determined by AEGIS to be drug-explained or entropy-consistent. A visceral, live demonstration of alarm fatigue reduction during the demo.

---

## Chapter 8: The Narrative Agent — Grounded, LLM-Powered Handoff Summaries

*(Formerly "Auto-Generated Clinical Narrative." This is the second chapter with the full AI-agent upgrade — and it's a direct reversal of the original design decision.)*

### 8.1 Purpose

Numbers and color codes are efficient for experienced users staring at a dashboard. But clinical handoffs between shifts happen in *narrative* — plain clinical English. A well-written summary of a patient's entropy trajectory, drug interactions, and alert status bridges AEGIS's data outputs to how clinicians actually communicate.

### 8.2 Why We Reversed the Original Decision

The original spec deliberately avoided an LLM here, citing "the unpredictability of LLM-generated clinical text for safety reasons," in favor of a fixed decision-tree of templates. That caution was reasonable for a first pass, and the underlying concern — an LLM inventing a number or a drug interaction that isn't real — is still exactly the right thing to worry about. So the Narrative Agent doesn't remove that safeguard; it replaces "avoid the LLM entirely" with "constrain the LLM tightly enough that the failure mode can't reach the clinician."

Concretely: the LLM is given a structured JSON snapshot of the patient's *exact* current state — every number that could possibly appear in the narrative — and a system prompt that forbids introducing any figure, drug name, or claim not present in that JSON. After generation, every number in the output is programmatically extracted and matched against the input JSON. Any sentence containing a number that doesn't match is discarded and replaced with the original template sentence for that slot. The result reads like a clinician wrote it, but every fact in it is provably grounded in the same structured state object the rest of the dashboard uses — the LLM can rephrase and prioritize, but it cannot fabricate.

This also means the agent can do something the old template tree genuinely couldn't: reference the patient's own history across the admission, not just the current snapshot — because that history is just more structured data in the same JSON object, not something the model has to "remember" or invent.

### 8.3 Template Selection / Context Assembly

The structured context handed to the agent still follows the original decision logic for *what to include*, now assembled as data rather than branching template code:

```
IF alert.severity == CRITICAL: include critical-severity context block
ELSE IF alert.severity == WARNING: include warning-severity context block
ELSE IF alert.severity == WATCH: include watch-severity context block
ELSE: include stable context block

IF alert.drug_masked: include masking context
FOR EACH active drug: include drug effect entry
IF interventions available: include Clinical Reasoning Agent's top suggestion
IF prior CES excursions exist this admission: include admission history
```

### 8.4 Example Output — WARNING Level, With Admission History

> *"Patient P003 has shown progressive loss of physiological complexity over the past 4.2 hours. The Composite Entropy Score has declined from 0.71 to 0.34 (WARNING level), driven primarily by declining heart rate entropy (SampEn: 0.42 → 0.18) and respiratory rate entropy (SampEn: 0.38 → 0.15). Current vitals remain within normal ranges (HR 84, BP 112/72, SpO2 96%, RR 17, Temp 37.2°C). This is the patient's second entropy decline this admission — the first, twelve hours ago, resolved with a fluid bolus. Patient is currently receiving Norepinephrine 0.08 mcg/kg/min (started 6h ago), which may be supporting blood pressure values — see drug masking note. Traditional NEWS2 score is 1 (low risk); entropy analysis suggests elevated risk within 4–6 hours. The Clinical Reasoning Agent's top suggestion, based on 142 similar historical cases and this patient's earlier response to fluids, is a repeat fluid bolus before vasopressor escalation."*

### 8.5 Example Output — Stable Patient

> *"Patient P011 is maintaining normal physiological complexity. Composite Entropy Score: 0.78, stable for the past 8 hours. All vital signs within expected parameters. Propofol infusion (active since 10h ago) accounts for a mild, expected reduction in heart rate variability and does not indicate deterioration. No clinical action required at this time."*

### 8.6 Failure Mode

If the Claude API call times out or errors, the Narrative Agent falls back to the original fixed-template output from the pre-agent design — degraded, but never blocked, and never silently wrong. See Chapter 21.3 for the full fallback matrix.

---

## Chapter 9: Cross-Vital Correlation Matrix

*(Unchanged — deterministic statistics, no AI involved.)*

### 9.1 Purpose

Vitals don't operate in isolation — HR affects BP, RR affects SpO2. In a healthy body these interactions form a web of correlations that maintain homeostasis. As the body fails, tightly coupled vitals can decouple, or previously independent vitals can become abnormally coupled (a single overwhelming process, like sepsis, driving everything at once).

### 9.2 Computation

For each vital pair, compute the rolling Pearson correlation over the current 300-point window:

𝑟(𝑋,𝑌) = Σ(𝑋ᵢ−𝑋̄)(𝑌ᵢ−𝑌̄) / √[Σ(𝑋ᵢ−𝑋̄)² Σ(𝑌ᵢ−𝑌̄)²]

Arrange into a symmetric 6×6 matrix (diagonal = 1.0), compare against a baseline matrix (reference population or the patient's own early hours), and compute the deviation matrix Δ𝑟 = 𝑟current − 𝑟baseline. Large |Δ𝑟| flags a clinically significant relationship change.

### 9.3 What to Look For

**Decoupling:** HR and BP normally show a moderate negative correlation (baroreflex compensation). If it weakens, the body may be losing that compensatory ability. Similarly for RR/SpO2.

**Abnormal coupling:** if HR, RR, and Temp all start moving together strongly, it can suggest a single systemic process (e.g., sepsis) driving everything at once.

### 9.4 Visualization

A heatmap on the deep-dive view: deep blue (strong negative), white (none), deep red (strong positive). Cells where |Δ𝑟| exceeds a threshold (e.g., 0.3) are highlighted.

---

## Chapter 10: Pharmacokinetic Drug Simulation

*(Unchanged — deterministic PK math, feeds the Pharmacology Context Agent.)*

### 10.1 Purpose

The basic drug model in Chapter 3 is binary — a drug is "active" or "not." Real drugs are absorbed, peak, and are metabolized continuously. The PK simulation replaces the on/off model with a continuous concentration curve.

### 10.2 The One-Compartment PK Model

**IV bolus:** 𝐶(𝑡) = (𝐷/𝑉𝑑) · 𝑒^(−𝑘𝑒·𝑡)

**Continuous infusion:** 𝐶(𝑡) = (𝑅/𝐶𝐿) · (1 − 𝑒^(−𝑘𝑒·𝑡))

**After infusion stops at 𝑇stop:** 𝐶(𝑡) = 𝐶(𝑇stop) · 𝑒^(−𝑘𝑒·(𝑡−𝑇stop))

Where 𝑘𝑒 = ln(2)/𝑡½ (elimination rate from half-life), 𝐶𝐿 = 𝑘𝑒 × 𝑉𝑑 (clearance), 𝑉𝑑 = volume of distribution, 𝑅 = infusion rate, 𝐷 = dose.

### 10.3 Concentration → Effect (Emax Model)

𝐸(𝐶) = 𝐸max · 𝐶 / (𝐸𝐶50 + 𝐶)

Smooth, realistic: near-linear at low concentration, diminishing returns as concentration rises, plateau at 𝐸max.

### 10.4 Integration

Instead of a binary "is the drug active?" check, the drug adjustment algorithm (3.3) uses this continuous 𝐸(𝐶(𝑡)) as the expected effect magnitude at every timestep — correctly modeling gradual onset, peak timing, gradual wear-off, and dose-dependence.

| Parameter | Norepinephrine | Metoprolol IV |
|---|---|---|
| 𝑉𝑑 (L/kg) | 0.1 | 3.2 |
| 𝑡½ (hours) | 0.04 (2.5 min) | 3.5 |
| 𝐸max (BP mmHg) | +40 | −20 |
| 𝐸𝐶50 (mcg/L) | 5.0 | 0.08 |
| Route | Infusion | Bolus |

Drugs without reliable PK parameters fall back to the simpler binary model from Chapter 3.

---

## Chapter 11: Voice Alerts

*(Unchanged.)*

### 11.1–11.2 Purpose and Behavior

Clinicians aren't always looking at a screen. Voice alerts add an auditory channel, reserved for high-severity transitions to avoid becoming a new source of fatigue:

| Condition | Voice Alert |
|---|---|
| WATCH → WARNING | Chime + "Bed [X], Patient [ID]: Entropy warning. Complexity declining." |
| WARNING → CRITICAL | Triple chime + "Bed [X], Patient [ID]: Critical entropy alert. Immediate review recommended." |
| Drug masking detected (WARNING/CRITICAL) | "Bed [X], Patient [ID]: Drug masking detected. [Drug] may be hiding deterioration." |
| CRITICAL unacknowledged >15 min | Repeats every 5 minutes until acknowledged |

Clear, calm voice; distinct from existing hospital tones (speech, not beeps, avoids the desensitization clinicians already have to alarm tones). Master volume, mute (auto-unmute after 30 min as a safety measure), per-severity trigger selection, and a voice alert log are all user-controllable.

### 11.3 Technical Implementation

Web Speech API (`SpeechSynthesis`) called from the frontend. On a WebSocket severity-transition message: build the alert text, create a `SpeechSynthesisUtterance`, set voice parameters, call `speechSynthesis.speak()`, log it. Falls back to pre-recorded clips if the browser lacks Web Speech API support.

---

## Chapter 12: Digital Twin — 3D Body Visualization

*(Unchanged.)*

### 12.1 Purpose

A 3D human body model, color-coded live by the entropy/vital status of the corresponding organ systems — red glow for danger, green for stable, a blue ring for drug support. Instant, spatial, intuitive.

### 12.2 Body Region Mapping

| Region | Primary Vitals | Entropy Source | Reasoning |
|---|---|---|---|
| Heart / Chest (Cardiac) | HR + BP | HR + BP SampEn | Direct outputs of cardiac function |
| Lungs / Chest (Respiratory) | RR + SpO2 | RR + SpO2 SampEn | RR is lung-generated; SpO2 reflects gas exchange |
| Brain / Head | Combined CES + trend | Overall CES | Most sensitive organ to circulatory/oxygenation change |
| Peripheral Extremities | BP + SpO2 + Temp | BP SampEn + Temp value | Circulation redirected away from extremities during shock |
| Abdomen / Core | Temperature + composite | Temp SampEn + overall CES | Core temp regulation, metabolic state |
| Kidneys (posterior/flank) | BP (+ urine output if available) | BP SampEn | Highly BP-sensitive; sustained decline suggests renal risk |

### 12.3 Color-Coding

Region Health = Σ(𝑤ᵥ × normalize(SampEnᵥ)) / Σ𝑤ᵥ across that region's mapped vitals — essentially a local CES.

| Region Health | Visual |
|---|---|
| 0.60 – 1.00 | Green glow |
| 0.40 – 0.59 | Yellow glow |
| 0.20 – 0.39 | Orange glow, gentle pulse |
| 0.00 – 0.19 | Red glow, pronounced pulse |

A blue ring overlays any region whose mapped vitals are affected by an active drug; if `drug_masked` is true for that region, the ring becomes blue-and-red striped ("support here, but underlying status may be worse than it looks").

### 12.4 Technical Implementation

**Option A (recommended):** Three.js with a pre-built low-poly GLTF/GLB body model (clearly separated mesh groups per region), materials updated on each WebSocket tick, optional rotation, hover tooltips ("Cardiac Region — Health: 0.42 (WATCH). HR SampEn: 0.31, BP SampEn: 0.53. Active Drug: Norepinephrine.").

**Option B (simpler fallback):** a 2D SVG body diagram with clickable, fillable region paths — 80% of the visual impact for 20% of the effort.

Clicking a region can expand the relevant vital-sign charts, highlight contributing vitals in the entropy panel, or show active drug effects for that region.

---

# Part IV: System Architecture & Technical Details

## Chapter 13: Technology Stack

### 13.1 Backend

| Component | Technology | Why |
|---|---|---|
| API Framework | FastAPI (Python) | Native WebSocket + REST support, async by default for concurrent patient streams, and Python is the natural home for NumPy/SciPy/scikit-learn |
| Data Replay Service | Python script | Replays pre-extracted MIMIC-IV data at 60× speed (1 real minute per demo second), simulating a live monitor feed with no medical hardware |
| Entropy Computation | NumPy + SciPy (+ Numba for hot loops) | Vectorized distance computation for SampEn; Numba JIT gives 10–50× speedup on the innermost loop |
| Clinical Reasoning Agent (retrieval) | scikit-learn KDTree/BallTree + StandardScaler | Pre-trained and serialized to disk for instant load at startup |
| **AI Agent Layer** *(new)* | **Anthropic API — Claude, called from the backend** | Powers the Clinical Reasoning Agent's explanation step (4.6) and the Narrative Agent (Ch. 8). Called only *after* all deterministic computation is done, on a structured context object — never used to compute a clinical number, only to explain one. Both call sites share the same grounding/validation guardrail. |
| Database | SQLite | Zero-configuration, file-based; stores the drug lookup table, historical case features, and entropy logs. Can move to PostgreSQL later if concurrency demands it |
| Containerization | Docker + docker-compose | All services packaged; `docker-compose up` starts everything, enabling zero-dependency single-command deployment |

### 13.2 Frontend

| Component | Technology | Why |
|---|---|---|
| UI Framework | React.js | Component-based dashboard: patient cards, entropy gauges, charts, alert panels |
| Charting | D3.js | Full control for entropy curves with drug-event markers, MSE heatmaps, correlation matrices |
| 3D Visualization | Three.js | Standard for browser 3D — model loading, materials, lighting, camera controls |
| WebSocket Client | Native WebSocket API / socket.io-client | Real-time patient state updates drive component re-renders |
| Styling | CSS with CSS Variables | Dark theme via custom properties; CSS Grid for the multi-patient ward layout |

### 13.3 Communication Between Components

| From | To | Protocol | Format |
|---|---|---|---|
| Data Replay Service | Backend Ingestion | HTTP POST `/api/v1/vitals` | JSON Vital Sign Record |
| Frontend | Backend (queries) | HTTP GET various REST endpoints | JSON |
| Backend | Frontend (real-time) | WebSocket `ws://host:8000/ws/patients`, `/ws/alerts` | JSON Patient State / Alert Object |
| Frontend | Backend (drug admin, ack) | HTTP POST | JSON request bodies |

---

## Chapter 14: Data Pipeline — From Raw MIMIC-IV to Working Demo

*(Unchanged — this is data engineering, not AI, and stays deterministic.)*

### 14.1 MIMIC-IV Extraction

Key source tables: `chartevents` (vitals, filtered by item ID: 220045 HR, 220277 SpO2, 220050 Systolic BP, 220051 Diastolic BP, 220210 RR, 223761/223762 Temperature), `inputevents` (drug administration), `icustays` (admission/discharge), `patients` (demographics).

### 14.2 Extraction Process

1. Identify 20–50 hero-candidate ICU stays: duration ≥24h, a documented critical event, <10% missing vitals, medication administration before the event, and — the key criterion — vital sign *values* staying normal for ≥2h before the event while entropy drops (requires running SampEn on candidates to verify).
2. Extract vitals, drug records, demographics, and stay metadata for each.
3. Pivot to one row per timestamp with all six vitals as columns.
4. Resample to uniform 1-minute intervals: forward-fill gaps ≤5 min, mark longer gaps as null (never fabricate).
5. Save each stay as a pseudonymized CSV/Parquet file.

### 14.3 Historical Case Feature Vectors

For every stay with a deterioration event: locate the event timestamp, look back 6 hours, compute all 25 features (14.3 → same as Ch. 4.3), record interventions applied and outcome, save to a structured file.

### 14.4 Pre-Training the KNN Model

Standardize each feature column (save 𝜇ⱼ, 𝜎ⱼ), build the KDTree/BallTree, serialize tree + scaler with joblib/pickle, load once at startup for instant queries.

---

## Chapter 15: API Contract — Complete Specification

### 15.1 REST Endpoints

**`POST /api/v1/vitals`** — Receive a vital sign record from the Data Replay Service; appends to the patient's sliding window, triggers the entropy pipeline once the window fills. Response: `{"status": "accepted", "patient_id": "...", "window_size": 150}`

**`GET /api/v1/patients`** — List all monitored patients with summary status:
```json
[{
  "patient_id": "P001", "bed_number": "ICU-12",
  "composite_entropy": 0.72, "alert_severity": "NONE",
  "last_update": "2026-08-15T14:32:00Z",
  "active_drug_count": 2, "news2_score": 2, "qsofa_score": 0
}]
```

**`GET /api/v1/patients/{id}`** — Full current state, augmented with NEWS2/qSOFA, the Narrative Agent's text, the correlation matrix, Digital Twin region-health scores, and alarm fatigue metrics.

**`GET /api/v1/patients/{id}/history?hours=6`** — Timestamped state snapshots for charting.

**`GET /api/v1/patients/{id}/drugs`** — Active drugs with PK-modeled concentration/effect:
```json
[{
  "drug_name": "Norepinephrine", "drug_class": "vasopressor",
  "dose": 0.08, "unit": "mcg/kg/min", "route": "IV infusion",
  "start_time": "2026-08-15T08:00:00Z",
  "current_concentration": 12.5, "current_effect_fraction": 0.72,
  "expected_bp_effect_mmhg": 28.8, "is_within_effect_window": true
}]
```

**`POST /api/v1/patients/{id}/drugs`** — Record a drug administration event.

**`GET /api/v1/patients/{id}/recommendation`** *(new)* — The Clinical Reasoning Agent's current personalized recommendation, plus the raw evidence table it was grounded on (so the frontend can always show "here's the data behind this" alongside the generated text).

**`GET /api/v1/alerts`** / **`POST /api/v1/alerts/{id}/acknowledge`** — Active alerts across all patients; mark one as seen.

**`GET /api/v1/system/health`** — `{"status": "ok", "active_patients": 12, "uptime_seconds": 3456, "entropy_computations_total": 28490}`

**`GET /api/v1/alarm-fatigue`**
```json
{
  "traditional_alarms_total": 847, "aegis_alerts_total": 23,
  "reduction_percentage": 97.3,
  "traditional_alarms_by_cause": {
    "drug_explained": 412, "transient_fluctuation": 298,
    "sensor_artifact": 89, "true_positive": 48
  },
  "avg_lead_time_traditional_minutes": 18,
  "avg_lead_time_aegis_minutes": 312
}
```

### 15.2 WebSocket Channels

**`ws://host:8000/ws/patients`** — broadcasts on every patient state update (~1/sec/patient during demo): `{"event": "patient_update", "data": { /* Full Patient State Object */ }}`

**`ws://host:8000/ws/alerts`** — broadcasts only on new alerts or severity changes:
```json
{
  "event": "new_alert",
  "data": {
    "alert_id": "A-001", "patient_id": "P003", "severity": "WARNING",
    "message": "Entropy declining — predicted event in ~5.2 hours",
    "timestamp": "2026-08-15T14:32:00Z",
    "voice_alert_text": "Bed 7, Patient P003: Entropy warning. Complexity declining."
  }
}
```

---

## Chapter 16: Frontend Architecture

*(Unchanged in structure — one new leaf component for the agent's grounding data.)*

### 16.1 Component Hierarchy

```
<App>
 <Header> (logo, system status, alarm fatigue counter, voice alert controls)
 <Router>
  <WardView>
   <PatientCard> x N
    <EntropyGauge> <Sparkline> <VitalsStrip> <AlertBadge> <DrugIndicator>
   <AlarmFatiguePanel>

  <PatientDeepDive>
   <SplitScreenToggle>
   <TraditionalView> (when split-screen active)
   <VitalSignPanel>
    <VitalChart> x 6 (D3, SampEn overlay)  <DrugEventMarkers>
   <EntropyDashboard>
    <CESGauge> <MSEProfile> <RawVsAdjustedDisplay> <TrendArrow>
   <ClinicalScoresPanel>
    <NEWS2Display> <qSOFADisplay>
   <CorrelationMatrix> (D3 heatmap)
   <DigitalTwin> (Three.js 3D body model)
   <AlertPanel>
    <AlertBanner> <ContributingVitals> <DrugMaskingWarning>
    <InterventionCard> x 3-5
    <EvidenceGroundingPanel>  <!-- new: shows the raw table the agent reasoned from -->
   <ClinicalNarrative>  <!-- now the Narrative Agent's grounded output -->
   <PharmacokineticChart>

  <AlertTimeline>
   <TimelineChart>
 <VoiceAlertEngine> (non-visual)
 <WebSocketManager> (non-visual)
```

### 16.2 State Management

React Context (or Zustand): **Patient State Store** (map of patient_id → state, updated per WebSocket message), **Alert State Store** (active alerts + acknowledgments), **UI State Store** (current view, selected patient, split-screen toggle, voice mute, time range, filters).

### 16.3 WebSocket Management

A dedicated hook/component connects on mount, parses JSON messages into the state stores, shows "Reconnecting…" on disconnect with 3-second retry, and on reconnect pulls a full REST snapshot to catch up on anything missed.

### 16.4 Chart Rendering Performance

D3 charts re-render only when their specific data changes, using the enter-update-exit data-join pattern. Sparklines use plain SVG paths from the last 120 points (2h) — no complex bindings needed. The Three.js twin runs on `requestAnimationFrame` and only updates materials when region health actually changes. The correlation heatmap recomputes every 60 seconds, not every second, since correlations move slowly.

---

## Chapter 17: Deployment Architecture

### 17.1 Container Structure

**`aegis-backend`** — `python:3.11-slim`; FastAPI app, Entropy Engine, Pharmacology Context Agent, Clinical Reasoning Agent (retrieval + Claude calls), Narrative Agent, WebSocket server. Port 8000. Mounts the data volume (MIMIC-IV extracts, drug database, pre-trained models).

**`aegis-frontend`** — `node:18-alpine` build → `nginx:alpine` serve. Built React app, nginx proxying WebSocket to the backend. Port 3000.

**`aegis-replay`** — `python:3.11-slim`; Data Replay Service. Depends on `aegis-backend` being healthy before starting.

### 17.2 docker-compose.yml (structure)

```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes: ["./data:/data"]
    environment:
      - REPLAY_SPEED=60
      - LOG_LEVEL=info
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/system/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: always

  frontend:
    build: ./frontend
    ports: ["3000:80"]
    depends_on: [backend]
    restart: always

  replay:
    build: ./replay
    depends_on:
      backend:
        condition: service_healthy
    volumes: ["./data:/data"]
    environment:
      - BACKEND_URL=http://backend:8000
      - SPEED_MULTIPLIER=60
      - DATASET=hero_cases
    restart: always
```

### 17.3 Offline Operation

All images, MIMIC-IV extracts, pre-trained models, and the drug database are bundled or volume-mounted — no internet required at runtime *except* for the AI Agent Layer's Claude API calls. If venue internet is unreliable, pre-build and `docker save`/`docker load` the images via USB as before, and rely on the graceful fallback in Chapter 21.3 so a lost connection degrades the agent layer to templates rather than breaking the demo.

---

# Part V: Performance, Quality, and Validation

## Chapter 18: Performance Requirements and Optimization

*(Unchanged — entropy computation is the bottleneck, not the AI layer, which runs async and off the hot path.)*

### 18.1 Computational Cost of SampEn

𝑂(𝑁²) per vital — for 𝑁=300, ~89,000 pairwise comparisons per level. ~5–20ms per vital per patient with NumPy vectorization. For 6 vitals × 20 patients: 6×20×15ms ≈ 1.8s — borderline against a 1Hz target.

### 18.2 Optimization Strategies

1. **Numba JIT** on the innermost max-abs-difference loop — typically 10–50× speedup.
2. **Vectorized distance computation** via NumPy broadcasting (leverages BLAS/LAPACK).
3. **Incremental SampEn** — as the window slides by one point, update match counts by adding the new template's comparisons and removing the dropped template's (~𝑂(𝑁) instead of recomputing all 𝑂(𝑁²)).
4. **Staggered computation** — update 4–5 patients per quarter-second rather than all 20 at once, spreading CPU load and WebSocket traffic.
5. **Reduce window size** (300 → 200) as a last resort — cuts comparisons ~56%, at the cost of slightly less reliable estimates (200 is still above the minimum recommended threshold for 𝑚=2).

### 18.3 Performance Targets

| Metric | Target |
|---|---|
| SampEn per vital per patient | ≤ 25ms |
| Full pipeline per patient (all vitals + CES + drug adjustment + evidence lookup) | ≤ 500ms |
| End-to-end latency (ingestion → dashboard) | ≤ 2 seconds |
| Dashboard frame rate | ≥ 30 FPS |
| Concurrent patient capacity | ≥ 20 |
| Backend memory | ≤ 2 GB |
| Frontend memory | ≤ 500 MB |

---

## Chapter 19: Validation Approach

### 19.1 Hero Case Validation

1. Select 10 MIMIC-IV stays with documented critical events.
2. For each, record when AEGIS first fires a WARNING/CRITICAL alert, when a traditional threshold alarm would have fired, and the actual event time.
3. Compute AEGIS lead time, traditional lead time, and the early-detection advantage between them.
4. **Target:** in ≥7/10 cases, AEGIS lead time ≥4 hours; in ≥3 of those, no traditional alarm would have fired at all before the event.

### 19.2 Drug Awareness Validation

Select 5 stays with a heart-rate-lowering drug administered. Run the pipeline with drug awareness off, count false entropy alerts from the expected drug effect; run it with drug awareness on, verify suppression. **Target:** ≥50% reduction in false alerts.

### 19.3 Clinical Reasoning Agent Validation *(new)*

For a subset of hero cases with multiple deterioration events in the same stay, verify that the agent's recommendation for the second event actually differs from what a pure population lookup would produce, and that every number/drug name in its output passes the grounding check against the structured context object (Chapter 4.6) — i.e., zero hallucinated figures across the validation set.

### 19.4 Alarm Fatigue Validation

Across all 10 hero cases combined: count total traditional alarms, total AEGIS alerts, compute reduction %. **Target:** ≥85% reduction.

---

# Part VI: Configuration & Operations

## Chapter 20: System Configuration

All parameters live in `config.yml` (or environment variables for Docker).

### 20.1 Data Replay

| Parameter | Default | Description |
|---|---|---|
| speed_multiplier | 60 | 60 = 1 real minute per demo second; 1 = real-time; 600 = fast-forward for testing |
| default_dataset | "hero_case_001.csv" | Default patient file |
| loop | true | Restart from the beginning when the dataset ends |
| multi_patient_count | 10 | Simultaneous simulated patients |

### 20.2 Entropy Engine

| Parameter | Default | Description |
|---|---|---|
| sampen_m | 2 | Embedding dimension |
| sampen_r_fraction | 0.2 | Tolerance as fraction of series SD |
| window_size | 300 | Sliding window length |
| min_valid_fraction | 0.8 | Minimum valid-data fraction for a reliable computation |
| warmup_points | 300 | Points collected before first entropy computation |
| mse_scales | [1..10] | MSE scale factors |
| weights | HR .25, SpO2 .15, BP_sys .20, BP_dia .10, RR .20, Temp .10 | CES weights |
| thresholds | none .60, watch .40, warning .20, critical .00 | CES severity thresholds |
| trend_slope_threshold | 0.001 | Slope magnitude for falling/rising classification |
| trend_window_minutes | 360 | Window for the linear trend fit |

### 20.3 Pharmacology Context Agent

| Parameter | Default | Description |
|---|---|---|
| weight_reduction_factor | 0.5 | CES weight reduction for drug-explained vitals |
| tolerance_fraction | 0.30 | Tolerance for matching observed vs. expected drug effect |
| drug_database_path | "/data/drugs.json" | Drug lookup table location |
| pk_model_enabled | true | Continuous PK concentration modeling vs. simple binary |

### 20.4 Clinical Reasoning Agent (Evidence + AI Layer)

| Parameter | Default | Description |
|---|---|---|
| k_neighbors | 50 | Nearest neighbors retrieved |
| min_cases_for_recommendation | 5 | Minimum cases before suggesting an intervention |
| max_interventions_returned | 5 | Max ranked interventions returned |
| max_distance_threshold | 3.0 | Standardized Euclidean distance beyond which neighbors are "too dissimilar" |
| model_path | "/data/knn_model.joblib" | Pre-trained KDTree |
| scaler_path | "/data/scaler.joblib" | Pre-fitted StandardScaler |
| ai_model | "claude-sonnet-4-6" | Model used for the reasoning/narrative steps |
| ai_timeout_ms | 4000 | Timeout before falling back to templated output |
| ai_grounding_check | true | Whether generated numbers/drugs are validated against the structured context object |

### 20.5 API

| Parameter | Default | Description |
|---|---|---|
| host | "0.0.0.0" | Bind address |
| port | 8000 | API port |
| websocket_path | "/ws" | WebSocket base path |
| cors_origins | ["http://localhost:3000"] | Allowed frontend origins |

### 20.6 Frontend

| Parameter | Default | Description |
|---|---|---|
| port | 3000 | Frontend server port |
| websocket_url | "ws://localhost:8000/ws" | Backend WebSocket URL |
| refresh_interval_ms | 1000 | Fallback full-state refresh interval |
| theme | "dark" | UI theme |
| voice_alerts_enabled | true | Voice alerts on by default |
| voice_alert_severity_threshold | "WARNING" | Minimum severity to trigger voice |
| 3d_model_enabled | true | Toggle Digital Twin (disable for low-end machines) |

---

## Chapter 21: Error Handling & Edge Cases

### 21.1 Missing Vital Signs

| Scenario | System Behavior |
|---|---|
| One vital missing from a record | Others processed normally; missing vital's SampEn not updated; CES computed on available vitals with renormalized weights; dashboard shows "—" |
| All vitals missing | Record acknowledged, no processing. Gap >5 min flags the window. Dashboard shows "No data" + timestamp of last valid reading |
| <80% of window valid | Entropy flagged "unreliable"; CES shown in italics with a warning icon; no alerts generated from unreliable entropy |
| Patient data stops entirely | "Data feed lost" after 5 min; moved to a "Disconnected" section after 15 min |

### 21.2 Extreme Vital Sign Values

| Scenario | System Behavior |
|---|---|
| Physiologically impossible (HR = −5, SpO2 = 150) | Rejected at ingestion, treated as null, logged |
| Physiologically extreme but possible (HR = 250, Temp = 42) | Accepted — these are real emergencies; traditional thresholds fire, entropy behaves per the actual pattern |

### 21.3 Service Failures

| Scenario | System Behavior |
|---|---|
| Entropy Engine crashes | Backend returns last known values; dashboard shows stale-data warning; Docker restart policy recovers it |
| Clinical Reasoning Agent's retrieval (KD-Tree) crashes/slow | Entropy alerts still function independently; intervention cards show "Loading recommendations…" or "temporarily unavailable" |
| **AI Agent Layer (Claude API) times out or errors** *(new)* | **Reasoning Agent falls back to the raw ranked evidence table (4.5) with no generated explanation text; Narrative Agent falls back to the original fixed-template narrative (8.2). Nothing blocks; the dashboard simply shows the deterministic version until the next successful call.** |
| WebSocket disconnects | "Reconnecting…" banner, retry every 3s, full REST resync on reconnect |
| Database unavailable | Falls back to in-memory drug table loaded at startup; historical case search unavailable, panel shows "Historical data unavailable" |

---

# Part VII: Building It — Step by Step

## Chapter 22: Build Sequence

Each step builds on the last and produces something testable.

1. **Data Preparation** — Extract, clean, resample MIMIC-IV; identify hero cases via preliminary SampEn analysis. → Clean CSVs + annotated hero cases.
2. **SampEn Implementation** — Test on synthetic data (sine wave → low SampEn, noise → high SampEn), then real HR data; verify SampEn drops before documented events. → `compute_sampen(time_series, m, r)`.
3. **MSE Implementation** — Coarse-graining + SampEn wrapper. → `compute_mse(time_series, m, r, scales)`.
4. **Composite Entropy Score** — Normalization params from training data, weighted average, severity classification, trend regression. → Given a window, produce CES + severity + trend.
5. **Sliding Window & Ingestion** — FastAPI skeleton, `/api/v1/vitals`, in-memory per-patient window, trigger entropy at 300 points. → Running API that turns vitals into entropy.
6. **Data Replay Service** — CSV → POST at 60× speed. → Self-feeding system.
7. **WebSocket Broadcasting** — Broadcast full Patient State Object after each computation. → Live client updates.
8. **Drug Lookup Table** — 15–20 drug profiles, all fields from Chapter 3.2. → Queryable drug database.
9. **Pharmacology Context Agent** — Drug adjustment algorithm, `/api/v1/patients/{id}/drugs` endpoints, raw + adjusted CES pipeline. → Full drug-aware entropy pipeline.
10. **Pharmacokinetic Simulation** — One-compartment PK model replacing the binary drug-effect check. → Realistic ramp/peak/decay drug effects.
11. **Historical Case Feature Vectors** — 25-feature vectors for every deterioration-event stay. → KNN-ready database.
12. **Clinical Reasoning Agent — Retrieval + AI Layer** *(updated)* — Standardize features, build the KDTree, implement the KNN query + intervention ranking (4.2–4.5) exactly as before. **Then add the reasoning step:** assemble the structured context object (evidence table + patient's own admission history + drug/masking flags), write the constrained system prompt, call the Claude API, and implement the grounding-check validator that strips any ungrounded number/drug name before display. → Given a patient state, the system returns both a raw evidence table *and* a personalized, grounded explanation.
13. **Alert Generation & Full Backend Pipeline** — Wire Entropy → Pharmacology Context Agent → Clinical Reasoning Agent into one pipeline; generate alert objects on threshold crossings; invoke the Reasoning Agent for WARNING/CRITICAL; broadcast complete state via WebSocket. → Complete backend.
14. **NEWS2 and qSOFA** — Implement the scoring tables (Chapter 6); add to the Patient State Object. → Clinical scores alongside entropy.
15. **Narrative Agent** *(updated)* — Build the structured context assembly (8.3), write the grounded-generation system prompt, call Claude, implement the same grounding-check validator as Step 12 (shared component), wire the template fallback for failures. → Every state update includes a grounded, personalized plain-English summary.
16. **Cross-Vital Correlation Matrix** — Rolling Pearson correlations (Chapter 9); add the 6×6 matrix to the state object. → Inter-vital data ready for the heatmap.
17. **Alarm Fatigue Metrics** — Traditional alarm counter, AEGIS alert counter, all Chapter 7.2 metrics. → Quantified reduction data.
18. **React Project Setup** — Dark theme, layout shells, `EntropyGauge`/`Sparkline`/`AlertBadge`. → Frontend skeleton, no data yet.
19. **WebSocket Client & State Management** — Connection manager, state stores, wire messages to updates. → Frontend receiving live data.
20. **Ward View** — Patient cards wired to state, CES color coding, sparklines, badges, drug indicators, severity auto-sort. → Live multi-patient overview.
21. **Deep Dive — Vital Sign Charts** — D3 line charts with SampEn overlay, drug event markers, traditional threshold lines for contrast. → Detailed single-patient charts.
22. **Deep Dive — Entropy Dashboard** — CES gauge, MSE bar chart, raw-vs-adjusted display, trend arrow. → Entropy analysis view.
23. **Deep Dive — Alert & Recommendation Panel** — Alert banner, contributing vitals, drug masking indicator, intervention cards + disclaimer, plus the new evidence-grounding panel showing the raw table behind the agent's text. → Clinical decision support view.
24. **Split-Screen View** — Toggle between traditional and AEGIS views side by side. → The demo's most visually compelling comparison.
25. **Clinical Scores, Narrative, Correlation Display** — Wire NEWS2/qSOFA, render the Narrative Agent's text, build the D3 correlation heatmap. → All analytical outputs visible.
26. **Alarm Fatigue Dashboard** — Counters, percentages, cause breakdown, live suppression counter. → Quantified impact demonstration.
27. **Digital Twin** — Load the 3D model, map entropy to regions, color-coded materials, click interaction, drug overlays. → The visually stunning 3D view.
28. **Voice Alerts** — Web Speech API integration, trigger conditions, volume/mute/log controls. → Audio alerts for critical situations.
29. **Pharmacokinetic Visualization** — D3 chart of drug concentration over time with current point + effect annotation. → Visual explanation of drug modeling.
30. **Docker Packaging** — Dockerfiles for backend/frontend/replay, `docker-compose.yml`, verify `docker-compose up` boots everything, pre-build and save images. → Deployable system.
31. **Testing and Hardening** — Run all validation cases (Chapter 19), verify lead times, alarm reduction, and the AI grounding-check pass rate; load-test with 20 concurrent patients; fix bottlenecks; test on a clean machine. → Validated, robust demo.
32. **Demo Preparation** — Pick the best hero case, record a backup demo video, prepare the pitch, practice. → Ready to present.

---

*This document contains the complete technical, mathematical, and architectural specification for AEGIS. The entropy math, drug modeling, and system architecture are unchanged from the original Project Chronos design and remain fully deterministic. What's new is an explicit AI agent layer — the Clinical Reasoning Agent and the Narrative Agent — that sits strictly downstream of that deterministic pipeline, reasoning over structured, grounded data to turn population evidence into per-patient guidance, with a shared validation step that prevents either agent from introducing a fact that isn't already in the data. AEGIS transforms ICU monitoring from reactive threshold alarms to proactive, entropy-based, personalized intelligence — running on existing hardware, with a human clinician as the final decision-maker at every step.*

**End of Document**
