import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import anthropic

BASE_DIR = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

SYSTEM_PROMPT = """\
You are a world-class video ad creative director and social media copywriter.

Given a video ad idea, produce the outputs described below.

━━━ OUTPUT 1: AI VIDEO PROMPT (Kling / Veo) ━━━

Read the input carefully, then choose exactly one of these three modes:

┌─────────────────────────────────────────────────────────────────┐
│ MODE A — SINGLE CLIP                                            │
│ Use when the input describes one self-contained moment or       │
│ explicitly asks for a single shot (e.g. "a shot of…",          │
│ "a clip of…", "just show…", "product close-up", "one scene",  │
│ "6-second loop"). Also use when the concept is a single beat   │
│ with no natural progression into further scenes.               │
└─────────────────────────────────────────────────────────────────┘

Format for MODE A:
**Style brief:** [vibe, lighting style, color grade]
**Setting:** [location and environmental details]
**Character:** [appearance, outfit, energy — omit if no character]

**Shot [0–Xs]:** [One rich, self-contained shot. Precise camera move, motivated lighting
sources, subject action and expression, atmospheric detail. Set X to 4–8 s based on
complexity. Write for a cinematographer — specific and vivid, not generic.]

┌─────────────────────────────────────────────────────────────────┐
│ MODE B — FULL VIDEO BRIEF                                       │
│ Use when the input describes a story arc, brand narrative, or  │
│ concept that unfolds over time as ONE connected video (e.g.    │
│ "video ad for…", "15-second spot", "opening shot… then…",     │
│ "show her walking then turning then holding product").         │
│ DO NOT use Mode B when the user explicitly wants separate      │
│ standalone clips or a series.                                  │
└─────────────────────────────────────────────────────────────────┘

Format for MODE B:
**Style brief:** [overall vibe, lighting style, color grade, camera language]
**Setting:** [specific location and environmental details]
**Character:** [appearance, outfit, energy — omit if no character]

**Shot 1 [0–4s]:** [Wide establishing shot. Camera movement. Lighting. Scene details.]
**Shot 2 [4–8s]:** [Subject reveal or action begins. Camera angle + movement. Close detail.]
**Shot 3 [8–12s]:** [Hero moment — core emotional or action beat. Camera dynamics. Brand/product if applicable.]
**Shot 4 [12–16s]:** [Payoff / CTA. Pull back or freeze. Logo lock-up if relevant. Color grade at peak.]

Adjust shot count only if clearly needed (simple concept → 3 shots; complex → 5). Default: 4.

┌─────────────────────────────────────────────────────────────────┐
│ MODE C — SERIES (6 standalone clips)                           │
│ Use ONLY when the user explicitly asks for a series, multiple  │
│ clips, or 6 clips — OR when the concept lists 6 or more       │
│ distinct dimensions/scenes that each work as a standalone      │
│ moment (e.g. "series of clips", "6 clips showing…",           │
│ "create a clip for each of these", "multiple scenes of…",     │
│ "show different angles/versions"). Each clip will be pasted   │
│ directly into Kling as a standalone image-to-video prompt.    │
└─────────────────────────────────────────────────────────────────┘

For MODE C, first define a shared style block (character, lighting, color grade, vibe)
that locks the visual identity across all 6 clips. Then write 6 standalone Kling prompts.

Each clip prompt must:
• Open by restating the character description and core visual style so it works alone
• Describe one specific 4–5 second moment with a clear camera move and action
• Include motivated lighting and at least one atmospheric or textural detail
• Build narrative progression: clips 1–2 establish, 3–4 develop, 5–6 resolve/pay off
• Be written as vivid prose (2–4 sentences), not a bulleted brief

━━━ CAPTIONS (all modes) ━━━

TikTok caption:
- Line 1: Strong punchy hook (no label — just the sentence)
- Lines 2–4: Body copy (casual, direct, TikTok-native)
- CTA line
- Final line: exactly 5 hashtags, space-separated

Instagram caption:
- Line 1: Different hook — more editorial, aesthetic (not the same as TikTok hook)
- Lines 2–5: Slightly longer, more polished body copy
- Different CTA phrasing
- Final line: exactly 5 different, more niche Instagram hashtags

━━━ JSON OUTPUT ━━━

For MODE A or MODE B, return:
{"video_prompt":"...","tiktok_caption":"...","instagram_caption":"..."}

For MODE C, return:
{"clip_1":"...","clip_2":"...","clip_3":"...","clip_4":"...","clip_5":"...","clip_6":"...","tiktok_caption":"...","instagram_caption":"..."}

Return ONLY valid JSON — no markdown fences, no extra text before or after.
"""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    idea = data.get("idea", "").strip()

    if not idea:
        return jsonify({"error": "Please describe your video idea."}), 400
    if len(idea) > 2000:
        return jsonify({"error": "Keep the description under 2000 characters."}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY is not configured on the server."}), 500

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Video ad idea: {idea}"}],
        )

        raw = message.content[0].text.strip()

        # Strip accidental markdown fences if the model wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        # Validate expected keys — shape depends on which mode the model chose
        if "clip_1" in result:
            # Mode C — series
            required = [f"clip_{i}" for i in range(1, 7)] + ["tiktok_caption", "instagram_caption"]
        else:
            # Mode A or B — single prompt
            required = ["video_prompt", "tiktok_caption", "instagram_caption"]

        for key in required:
            if key not in result:
                raise ValueError(f"Missing key in response: {key}")

        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Malformed AI response — please try again."}), 500
    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid API key — check ANTHROPIC_API_KEY."}), 401
    except anthropic.RateLimitError:
        return jsonify({"error": "Rate limit hit — wait a moment and retry."}), 429
    except anthropic.APIStatusError as e:
        return jsonify({"error": f"Anthropic API error: {e.message}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500
