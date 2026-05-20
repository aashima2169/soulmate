"""
Frequency - Relationship Pattern Intelligence
Backend API using FastAPI + Gemini Vision + Supabase
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
from supabase import create_client, Client
import base64
import json
import uuid
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Frequency API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Clients ────────────────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-pro")

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)

# ── System Prompt ──────────────────────────────────────────────────────────────
COUNSELOR_PROMPT = """
You are a behaviorally-informed relationship counselor with deep expertise in 
attachment theory, communication dynamics, and emotional pattern recognition.

Your job is NOT to judge or make decisions for the user.
Your job is to surface patterns they cannot see clearly because they are 
emotionally invested.

ANALYZE the conversation screenshots and user responses provided.

IDENTIFY the following:

1. USER_ATTACHMENT_STYLE
   - Observed functional style (from their actual messages — NOT what they claim)
   - Key behavioral signals you noticed
   - Confidence level (low/medium/high)

2. THEIR_ATTACHMENT_STYLE  
   - Observed functional style (from the other person's behavior)
   - Key behavioral signals you noticed
   - Confidence level (low/medium/high)
   - Gap between stated vs actual (if user mentioned what they claimed)

3. POWER_DYNAMIC
   - Who is accommodating whom
   - Investment balance (who is investing more energy)
   - Pacing compatibility (are both moving at the same speed)

4. SHRINKAGE_SIGNALS
   - Specific moments where the user made themselves smaller
   - Over-explaining, over-apologizing, over-accommodating patterns
   - Moments where user's needs were minimized or dismissed

5. DYNAMIC_TRAJECTORY  
   - What this dynamic looks like in 3 months if the pattern continues
   - What the user is likely to feel (not what might happen externally)

6. WORTH_CALIBRATION
   - Where the user is giving themselves away too early
   - Whether this person is meeting the user's actual worth
   - One specific pattern to watch

7. ONE_QUESTION
   - A single honest question for the user to sit with
   - Not an answer. Not advice. Just a question that cuts to the truth.

OUTPUT FORMAT — respond ONLY in this exact JSON structure:

{
  "user_attachment": {
    "style": "string (e.g. Secure with anxious activation under stress)",
    "signals": ["signal 1", "signal 2", "signal 3"],
    "confidence": "low|medium|high"
  },
  "their_attachment": {
    "style": "string",
    "signals": ["signal 1", "signal 2", "signal 3"],
    "confidence": "low|medium|high",
    "stated_vs_actual_gap": "string or null"
  },
  "power_dynamic": {
    "who_accommodates_more": "user|them|balanced",
    "investment_balance": "string description",
    "pacing_compatibility": "compatible|mismatched|too early to tell"
  },
  "shrinkage_signals": [
    {"moment": "string", "pattern": "string"}
  ],
  "dynamic_trajectory": "string — honest 2-3 sentence projection",
  "worth_calibration": {
    "giving_away_too_early": "string or null",
    "are_they_meeting_your_worth": "yes|partially|no|too early to tell",
    "watch_for": "string"
  },
  "verdict": "green|yellow|red",
  "verdict_reason": "string — one sentence, warm but honest",
  "one_question": "string — the single most important question",
  "prediction_confidence": "low|medium|high",
  "data_quality": "string — note any limitations from screenshot quality or limited data"
}

TONE RULES:
- Warm, direct, non-judgmental
- Never tell them to leave or stay
- Always return agency to the user
- Specific over generic — reference actual moments from their screenshots
- If data is limited, say so honestly in data_quality
"""


# ── Request Models ─────────────────────────────────────────────────────────────
class UserContext(BaseModel):
    # Behavioural scenario answers (not self-report)
    no_reply_reaction: str           # what they do when no reply for a day
    conflict_style: str              # how they handle disagreement
    early_investment: str            # how quickly they open up
    boundary_response: str           # what they do when needs are dismissed
    repair_pattern: str              # who initiates repair after tension

    # Optional self-reported style for recalibration delta
    self_reported_style: Optional[str] = None

    # Additional context
    how_long_talking: Optional[str] = None
    have_met_in_person: Optional[bool] = None
    user_notes: Optional[str] = None


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze_conversation(
    screenshots: list[UploadFile] = File(...),
    user_context: str = Form(...),        # JSON string of UserContext
    session_id: Optional[str] = Form(None),
):
    """
    Main analysis endpoint.
    Accepts up to 10 screenshots + user context answers.
    Returns full attachment/pattern analysis.
    """
    if len(screenshots) > 10:
        raise HTTPException(400, "Maximum 10 screenshots allowed")

    # Parse user context
    try:
        ctx = UserContext(**json.loads(user_context))
    except Exception as e:
        raise HTTPException(400, f"Invalid user context: {e}")

    # Convert screenshots to Gemini parts
    image_parts = []
    for screenshot in screenshots:
        content = await screenshot.read()
        image_parts.append({
            "mime_type": screenshot.content_type or "image/jpeg",
            "data": base64.b64encode(content).decode("utf-8"),
        })

    # Build context prompt
    context_text = f"""
USER CONTEXT (behavioral self-assessment):
- When no reply for a day: {ctx.no_reply_reaction}
- Conflict style: {ctx.conflict_style}
- How quickly they open up: {ctx.early_investment}
- When needs are dismissed: {ctx.boundary_response}
- Who initiates repair: {ctx.repair_pattern}
- How long talking: {ctx.how_long_talking or 'not specified'}
- Met in person: {ctx.have_met_in_person if ctx.have_met_in_person is not None else 'not specified'}
- Self-reported attachment style: {ctx.self_reported_style or 'not provided'}
- Additional notes: {ctx.user_notes or 'none'}

SCREENSHOTS: {len(image_parts)} conversation screenshot(s) attached.
Analyze them in chronological order.

{COUNSELOR_PROMPT}
"""

    # Call Gemini
    try:
        parts = []
        for img in image_parts:
            parts.append({
                "inline_data": {
                    "mime_type": img["mime_type"],
                    "data": img["data"],
                }
            })
        parts.append({"text": context_text})

        response = gemini.generate_content(parts)
        raw_text = response.text.strip()

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        analysis = json.loads(raw_text)

    except json.JSONDecodeError:
        raise HTTPException(500, "AI response could not be parsed. Please try again.")
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    # Save to Supabase for metrics tracking
    analysis_id = str(uuid.uuid4())
    session_id = session_id or str(uuid.uuid4())

    try:
        supabase.table("analyses").insert({
            "id": analysis_id,
            "session_id": session_id,
            "screenshot_count": len(screenshots),
            "self_reported_style": ctx.self_reported_style,
            "derived_user_style": analysis.get("user_attachment", {}).get("style"),
            "derived_their_style": analysis.get("their_attachment", {}).get("style"),
            "verdict": analysis.get("verdict"),
            "prediction_confidence": analysis.get("prediction_confidence"),
            "power_dynamic": analysis.get("power_dynamic", {}).get("who_accommodates_more"),
            "pacing_compatibility": analysis.get("power_dynamic", {}).get("pacing_compatibility"),
            "worth_meeting": analysis.get("worth_calibration", {}).get("are_they_meeting_your_worth"),
            "has_stated_vs_actual_gap": bool(
                analysis.get("their_attachment", {}).get("stated_vs_actual_gap")
            ),
            "created_at": datetime.utcnow().isoformat(),
            # user_outcome filled in later via /api/feedback
        }).execute()
    except Exception:
        pass  # Don't fail the request if DB write fails

    return {
        "analysis_id": analysis_id,
        "session_id": session_id,
        "analysis": analysis,
    }


@app.post("/api/feedback")
async def record_outcome(
    analysis_id: str,
    outcome: str,               # "prediction_correct" | "prediction_wrong" | "still_dating" | "ended"
    user_rating: Optional[int] = None,   # 1-5
    notes: Optional[str] = None,
):
    """
    User reports back on whether the prediction was accurate.
    This powers the prediction accuracy metrics.
    """
    try:
        supabase.table("analyses").update({
            "user_outcome": outcome,
            "user_rating": user_rating,
            "outcome_notes": notes,
            "outcome_at": datetime.utcnow().isoformat(),
        }).eq("id", analysis_id).execute()
    except Exception as e:
        raise HTTPException(500, f"Feedback save failed: {e}")

    return {"status": "recorded", "analysis_id": analysis_id}


@app.get("/api/metrics")
async def get_metrics():
    """
    Internal metrics dashboard.
    Tracks prediction accuracy, style detection patterns, verdict distribution.
    """
    try:
        # Overall stats
        total = supabase.table("analyses").select("id", count="exact").execute()
        with_outcome = supabase.table("analyses")\
            .select("id", count="exact")\
            .not_.is_("user_outcome", "null")\
            .execute()

        correct = supabase.table("analyses")\
            .select("id", count="exact")\
            .eq("user_outcome", "prediction_correct")\
            .execute()

        # Verdict distribution
        verdicts = supabase.table("analyses")\
            .select("verdict")\
            .execute()

        verdict_counts = {"green": 0, "yellow": 0, "red": 0}
        for row in verdicts.data:
            v = row.get("verdict")
            if v in verdict_counts:
                verdict_counts[v] += 1

        # Style deviation (self-report vs derived)
        with_gap = supabase.table("analyses")\
            .select("id", count="exact")\
            .eq("has_stated_vs_actual_gap", True)\
            .execute()

        total_count = total.count or 0
        outcome_count = with_outcome.count or 0
        correct_count = correct.count or 0

        return {
            "total_analyses": total_count,
            "analyses_with_outcome": outcome_count,
            "prediction_accuracy": round(
                correct_count / outcome_count * 100, 1
            ) if outcome_count > 0 else None,
            "verdict_distribution": verdict_counts,
            "stated_vs_actual_gap_rate": round(
                (with_gap.count or 0) / total_count * 100, 1
            ) if total_count > 0 else None,
        }
    except Exception as e:
        raise HTTPException(500, f"Metrics fetch failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
