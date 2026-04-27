import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import anthropic

BASE_DIR = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

SYSTEM_PROMPT = """\
You are a world-class video ad creative director and social media copywriter.

Given a video ad idea, produce exactly three outputs formatted as described below.

━━━ OUTPUT 1: AI VIDEO PROMPT (Kling / Veo) ━━━

First, read the input carefully and decide which mode fits:

MODE A — SINGLE CLIP
Use when the input describes one contained moment, one action, one scene, or explicitly asks
for a short clip (e.g. "a shot of…", "a clip of…", "just show…", "6-second loop",
"product close-up", "one scene"). Also use when the concept is inherently a single beat
with no natural narrative progression.

Format for MODE A:
**Style brief:** [vibe, lighting style, color grade]
**Setting:** [location and environmental details]
**Character:** [appearance, outfit, energy — omit if no character]

**Shot [0–Xs]:** [One rich, self-contained shot. Include: precise camera move (e.g. slow
push-in, low-angle dolly, overhead crane descent), motivated lighting sources, subject
action and expression, atmospheric details. Duration X should be 4–8 s based on the
complexity of the action described. Be specific and vivid — write for a cinematographer,
not a brief.]

MODE B — FULL VIDEO BRIEF
Use when the input describes a story arc, multiple scenes, a brand narrative, a product
journey, or any concept that naturally unfolds over time (e.g. "video ad for…",
"a 15-second spot", "show her walking then turning then…", "opening shot… then…").

Format for MODE B:
**Style brief:** [overall vibe, lighting style, color grade, camera language]
**Setting:** [specific location and environmental details]
**Character:** [appearance, outfit, energy — omit if no character]

**Shot 1 [0–4s]:** [Wide establishing shot. Camera movement. Lighting. Scene details.]
**Shot 2 [4–8s]:** [Subject reveal or action begins. Camera angle + movement. Close detail.]
**Shot 3 [8–12s]:** [Hero moment — the core emotional or action beat. Camera dynamics. Brand/product if applicable.]
**Shot 4 [12–16s]:** [Payoff / CTA. Pull back or freeze. Logo lock-up or title card if relevant. Color grade at peak intensity.]

Add or remove shots only if the concept clearly calls for it (e.g. a very simple concept
may need 3 shots; a complex narrative may need 5). Default to 4.

For both modes: be specific and cinematic — not generic. Include motivated lighting
sources, exact camera moves, and emotional direction in every shot.

━━━ OUTPUT 2: TIKTOK CAPTION ━━━
- Line 1: A strong, punchy hook (no label — just the hook sentence itself)
- Lines 2–4: Body copy (casual, direct, native to TikTok)
- CTA line
- Final line: exactly 5 hashtags (space-separated)

━━━ OUTPUT 3: INSTAGRAM CAPTION ━━━
- Line 1: A different hook — more editorial, considered, aesthetic (not the same as TikTok)
- Lines 2–5: Slightly longer, more polished body copy
- Different CTA phrasing
- Final line: exactly 5 different, more niche Instagram hashtags

Return ONLY valid JSON — no markdown fences, no extra commentary:
{"video_prompt":"...","tiktok_caption":"...","instagram_caption":"..."}
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
            max_tokens=2048,
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

        # Validate expected keys
        for key in ("video_prompt", "tiktok_caption", "instagram_caption"):
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
