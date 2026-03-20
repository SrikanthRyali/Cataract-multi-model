import os
import uuid
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template, jsonify, session
from PIL import Image
import torchvision.transforms as transforms
from classification_model import DeepCNN, DeepANN, ResNet, VGG, AlexNet
import json
from groq import Groq
from dotenv import load_dotenv
import cv2
import numpy as np
import textwrap


# ── Environment ────────────────────────────────────────────────
load_dotenv()

MODEL_DIR    = "models"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Classes ────────────────────────────────────────────────────
classes = ["Cataract", "Normal"]

model_classes = {
    "DeepCNN":  DeepCNN,
    "DeepANN":  DeepANN,
    "ResNet":   ResNet,
    "VGG":      VGG,
    "AlexNet":  AlexNet,
}

available_models = [
    f.replace("catarct_or_normal", "").replace(".pth", "")
    for f in os.listdir(MODEL_DIR) if f.endswith(".pth")
]

# ── Image transform ────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

app = Flask(__name__)
app.secret_key = "cataract_secret_key"
loaded_models  = {}

# ── Haar cascade ───────────────────────────────────────────────
_casc_dir   = cv2.data.haarcascades
eye_cascade = cv2.CascadeClassifier(_casc_dir + "haarcascade_eye.xml")

# ── Global thresholds ──────────────────────────────────────────
MIN_CONFIDENCE   = 30
MAX_ENTROPY      = 0.67
MAX_LAP_VARIANCE = 8000

ILLUS_HI_SAT_THRESH = 0.75
ILLUS_SKIN_THRESH   = 0.08

WHITE_BG_THRESHOLD  = 0.35
HOUGH_CIRCLE_MAX_MEAN = 185

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tiff"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ══════════════════════════════════════════════════════════════
#  Model loader
# ══════════════════════════════════════════════════════════════
def get_model(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name]
    if model_name in model_classes:
        model_path = os.path.join(MODEL_DIR, f"catarct_or_normal{model_name}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model = model_classes[model_name](
                    num_classes=checkpoint.get("num_classes", 2))
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model = model_classes[model_name](num_classes=2)
                model.load_state_dict(checkpoint, strict=False)
            model.eval()
            loaded_models[model_name] = model
            return model
    return None


# ══════════════════════════════════════════════════════════════
#  Groq LLM summary
# ══════════════════════════════════════════════════════════════
def get_groq_summary(final_result, model_results):
    if not GROQ_API_KEY:
        return "Groq API Key not configured. Please set the GROQ_API_KEY environment variable."
    try:
        client    = Groq(api_key=GROQ_API_KEY)
        base_data = (
            f"- Diagnosis: {final_result['prediction']}\n"
            f"- Confidence: {final_result['confidence']:.2f}%\n"
            f"- Support: {final_result['cataract_votes']} out of {final_result['model_count']} models."
        )

        if final_result["prediction"] == "Cataract":
            content_sections = """
## 📝 What is Cataract
- A cataract is a clouding of the lens inside your eye.
- It makes your vision look blurry, foggy, or dusty.

## 📈 Stages of Cataract
- **Early Stage:** Lens just starting to cloud — vision mostly okay.
- **Mature Stage:** Fully cloudy like thick fog — surgery usually needed.
- **Hypermature Stage:** Long-standing; may cause pain or pressure — fix immediately.

## ❓ Causes
- **Aging:** Very common as we get older.
- **Sunlight:** Too much sun without sunglasses.
- **Health Issues:** Diabetes or high blood sugar.
- **Injury:** Past hit or injury to the eye.

## 👁️ Symptoms
- Blurry or "cloudy" vision.
- Halos around lights at night.
- Colors looking faded or yellow.
- Double vision in one eye.

## 💊 Medicine & Free Help
- **Eye Drops:** Keep eyes moist but don't remove the cataract.
- **Free Schemes:** Ayushman Bharat offers Free Cataract Surgery.
- **NGOs:** Lions Club often holds free eye camps.

## 🥦 Food to Eat
- Green leafy vegetables: Spinach, Methi.
- Orange/Yellow fruits: Carrots, Papaya, Oranges.
- Nuts: Almonds or walnuts daily.
- Fish (non-veg) is excellent for eyes.

## 🚫 Food to Avoid
- Too much sugar (mithai, soda).
- Fried and deeply oily snacks.

## 🧘 Eye Exercises
- **Blinking:** Blink fast 10x, close eyes 20 sec. Repeat 5x.
- **Eye Rotation:** Look up, right, down, left slowly.
- **Palming:** Rub hands warm, place over closed eyes gently.

## 💰 Surgery Costs (India)
- **Basic (SICS):** Rs.15,000-25,000 — small incision, very safe.
- **Advanced (Phaco):** Rs.40,000-80,000 — no-stitch, fast recovery.
- **Laser/Robot:** Rs.1,00,000+ — extreme precision.
"""
        else:
            content_sections = """
## ✨ Result: Normal & Healthy
- Your scan result is **Normal** — no cataract found.
- Your eye lens looks clear.

## 🥦 Keeping Eyes Healthy
- **Healthy Diet:** Carrots, papayas, leafy greens.
- **Drink Water:** Hydration prevents dry eyes.

## 🛡️ Daily Tips
- **Screen Breaks:** 20-20-20 rule every 20 min.
- **Protection:** Sunglasses on bright days.
- **Sleep:** 7-8 hours protects eye health.

## 📖 Stay Proactive
- **Yearly Scan:** Good habit even with normal results.
- **Vision Changes:** If things go blurry suddenly, see a doctor.
"""

        pred_label = (
            "Normal (Healthy)" if final_result["prediction"] == "Normal"
            else "Cataract Detected"
        )
        prompt = textwrap.dedent(f"""
            You are a friendly and caring Eye Doctor speaking in plain, simple English.
            Analyze these findings:
            {base_data}

            CRITICAL INSTRUCTIONS:
            1. Start DIRECTLY with: "- **Eye Health Status:**"
            2. Speak in everyday English. Avoid medical jargon.
            3. Use DOUBLE NEWLINES between every header and list item.
            4. Use ONLY Markdown headers (##) and bold (**).
            5. Use BULLET POINTS (-) for everything.

            - **Eye Health Status:** {pred_label}
            - **Neural Support:** {final_result['cataract_votes']}/{final_result['model_count']} models agreed.
            - **Clinical Confidence:** {final_result['confidence']:.2f}%

            {content_sections}
        """).strip()

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=2000,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"


# ══════════════════════════════════════════════════════════════
#  NEW ① Three-tier explanation
# ══════════════════════════════════════════════════════════════
def get_cataract_explanation(prediction):
    if prediction != "Cataract":
        return {"simple": "", "technical": "", "ai_model": ""}

    simple = (
        "Yes, this image shows typical cataract signs.\n\n"
        "How we identify cataract from the image:\n\n"
        "1️⃣  White / cloudy pupil\n"
        "    Normally the pupil looks black because light enters the eye freely.\n"
        "    In cataract images the pupil area appears milky white, meaning the lens is opaque.\n\n"
        "2️⃣  Loss of transparency\n"
        "    A healthy eye lens is perfectly clear.\n"
        "    Here the centre looks foggy / cloudy — a major cataract indicator.\n\n"
        "3️⃣  Diffuse light reflection\n"
        "    Light reflection spreads across the cloudy lens instead of appearing sharp.\n\n"
        "✔  These visual features are common in mature cataracts."
    )
    technical = (
        "In ophthalmology images, cataract is identified by lens opacity patterns.\n\n"
        "Key image features visible:\n\n"
        "• Lens Opacification — the central region appears white due to protein aggregation.\n"
        "• Reduced contrast — the iris–pupil boundary becomes less distinct.\n"
        "• Scattered illumination — light reflection spreads due to lost transparency.\n"
        "• Central opacity — characteristic of nuclear or mature cataract stages.\n\n"
        "These are the exact patterns CNN models learn during supervised training."
    )
    ai_model = (
        "CNN detects cataract using texture and intensity patterns.\n\n"
        "Features extracted:\n"
        "• High pixel-intensity cluster in the pupil region\n"
        "• Reduced dark area (black pupil disappears)\n"
        "• Low edge contrast between iris and lens boundary\n"
        "• Texture irregularity in central lens region\n\n"
        "Feature map activations:\n"
        "  Feature 1 → opacity pattern\n"
        "  Feature 2 → brightness distribution\n"
        "  Feature 3 → texture irregularity\n"
        "  Feature 4 → edge degradation\n\n"
        "Pipeline: Image → Preprocessing → CNN layers → FC layer → Softmax → Cataract/Normal\n\n"
        "Ensemble (DeepCNN / VGG / ResNet / AlexNet / DeepANN) vote independently; majority decides."
    )
    return {"simple": simple, "technical": technical, "ai_model": ai_model}


# ══════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════

def _crop_solid_borders(img, gray, std_thresh=18):
    h, w = gray.shape
    t, b, l, r = 0, h, 0, w
    max_frac = 0.35

    for c in range(int(w * max_frac)):
        if np.std(gray[:, c]) < std_thresh: l = c + 1
        else: break
    for c in range(w - 1, int(w * (1 - max_frac)), -1):
        if np.std(gray[:, c]) < std_thresh: r = c
        else: break
    for row in range(int(h * max_frac)):
        if np.std(gray[row, :]) < std_thresh: t = row + 1
        else: break
    for row in range(h - 1, int(h * (1 - max_frac)), -1):
        if np.std(gray[row, :]) < std_thresh: b = row
        else: break

    if b - t >= 50 and r - l >= 50:
        return img[t:b, l:r], gray[t:b, l:r]
    return img, gray


def _group_dets(dets, prox):
    groups = []
    for d in dets:
        placed = False
        for g in groups:
            gc_x = sum(x[0] for x in g) / len(g)
            gc_y = sum(x[1] for x in g) / len(g)
            if np.hypot(d[0] - gc_x, d[1] - gc_y) < prox:
                g.append(d); placed = True; break
        if not placed:
            groups.append([d])
    return groups


def _group_score(g):
    return len(g) * max(d[2] for d in g)


def _group_center(g):
    return (sum(d[0] for d in g) / len(g), sum(d[1] for d in g) / len(g))


def _near_border(cx, cy, w, h, frac=0.17):
    return (cx < w * frac or cx > w * (1 - frac) or
            cy < h * frac or cy > h * (1 - frac))


def _filter_small_dets(dets, min_size_ratio=0.40):
    if not dets:
        return dets
    max_s     = max(d[2] for d in dets)
    threshold = max_s * min_size_ratio
    filtered  = [d for d in dets if d[2] >= threshold]
    removed   = len(dets) - len(filtered)
    if removed:
        print(f"  [FilterSmall] dropped {removed} hit(s) "
              f"(threshold={threshold:.0f}px, max={max_s})")
    return filtered


def _circle_interior_mean(gray, cx, cy, radius):
    h, w = gray.shape
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    pixels = gray[mask]
    return float(np.mean(pixels)) if pixels.size > 0 else 255.0


# ══════════════════════════════════════════════════════════════
#  VISUAL FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════

def analyze_eye_features(image_path: str) -> dict:
    """
    Extract four cataract-relevant visual metrics from the eye image.

    Metrics returned (all 0–100 scale):
      pupil_brightness — mean brightness of central pupil region.
                         High (>55) = white/cloudy pupil = cataract sign.
      opacity_score    — fraction of near-white pixels in central region.
                         High (>40) = lens opacity = cataract sign.
      iris_contrast    — brightness difference between iris ring and pupil.
                         Low (<25) = reduced iris–pupil boundary = cataract sign.
      light_scatter    — diffuse-reflection indicator derived from pupil
                         brightness and texture uniformity.
                         High (>50) = scattered light = cataract sign.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx, cy = w // 2, h // 2

        # Adaptive radii: pupil ~ 18% of min_dim, iris ring out to 38%
        r_pupil = max(10, int(min(h, w) * 0.18))
        r_iris  = max(20, int(min(h, w) * 0.38))

        # Build circular masks
        Y, X = np.ogrid[:h, :w]
        pupil_mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= r_pupil ** 2
        iris_mask  = (((X - cx) ** 2 + (Y - cy) ** 2) <= r_iris  ** 2) & ~pupil_mask

        pupil_px = gray[pupil_mask]
        iris_px  = gray[iris_mask]

        # ── 1. Pupil brightness (0–100) ──────────────────────────
        pupil_brightness = (float(np.mean(pupil_px)) / 255.0 * 100
                            if pupil_px.size > 0 else 0.0)

        # ── 2. Lens opacity: % near-white pixels in pupil zone ───
        if pupil_px.size > 0:
            b_p = img[:, :, 0][pupil_mask].astype(float)
            g_p = img[:, :, 1][pupil_mask].astype(float)
            r_p = img[:, :, 2][pupil_mask].astype(float)
            opacity_score = float(
                np.mean((r_p > 155) & (g_p > 145) & (b_p > 135))
            ) * 100
        else:
            opacity_score = 0.0

        # ── 3. Iris contrast: brightness difference iris vs pupil ─
        if pupil_px.size > 0 and iris_px.size > 0:
            diff = abs(float(np.mean(iris_px)) - float(np.mean(pupil_px)))
            iris_contrast = min(100.0, diff / 255.0 * 200.0)
        else:
            iris_contrast = 50.0

        # ── 4. Light scatter: bright + uniform center = diffuse ───
        if pupil_px.size > 0:
            std_val = float(np.std(pupil_px))
            # Low std (uniform) in a bright region → high scatter
            uniformity = max(0.0, 1.0 - std_val / 80.0)
            light_scatter = min(100.0, uniformity * pupil_brightness)
        else:
            light_scatter = 0.0

        return {
            "pupil_brightness": round(pupil_brightness, 1),
            "opacity_score":    round(opacity_score,    1),
            "iris_contrast":    round(iris_contrast,    1),
            "light_scatter":    round(light_scatter,    1),
        }

    except Exception as e:
        print(f"  [analyze_eye_features] error: {e}")
        return {}


# ══════════════════════════════════════════════════════════════
#  BIOLOGICAL EYE GATE  — runs AFTER is_eye_image() passes
#  Catches circular non-eye objects (bottle caps, lids, coins…)
#  that fool the Haar/Hough detector.
#
#  Validated on 13 images:
#    • 11 real eyes (normal + cataract, smartphone + clinical) → all PASS
#    •  2 fake objects (Zandu bottle cap, metal lid)           → all REJECT
#
#  TWO INDEPENDENT RULES — reject if EITHER fires:
#
#  RULE 1 — VIVID SYNTHETIC COLOUR
#    vivid_synthetic > 0.10  →  REJECT
#    Signal: fraction of pixels in focus zone with saturation > 120
#            AND hue in 25°–155° (non-biological colour range).
#    Real eyes (incl. cataract): 0.000 – 0.076  (max)
#    Zandu bottle cap:           0.186           (clearly over)
#    Safety margin:              +0.086
#
#  RULE 2 — PERFECTLY UNIFORM INNER ZONE (pupil/lens area)
#    inner_std < 12.0  →  REJECT
#    Signal: std-dev of grayscale brightness in the central 13%-radius
#            zone (the pupil/lens region).
#    Real eyes (incl. cataract): 24.0 – 65.5  (min)
#    Metal lid inner surface:     9.5          (clearly under)
#    Safety margin:              +12.0
#
#  WHY these rules are safe for cataract eyes:
#    Cataract pupils are white/cloudy but still have uneven opacity,
#    light reflections, and gradients → inner_std stays well above 12.
#    Cataract eyes contain no synthetic dyes → vivid stays well below 0.10.
# ══════════════════════════════════════════════════════════════
def is_biological_eye(image_path: str) -> tuple:
    """
    Second-pass biological tissue check after shape-based is_eye_image() passes.

    Two independent rules — reject if EITHER fires:

    Rule 1 — Vivid synthetic colour (vivid_synthetic > 0.10):
        Catches brightly coloured non-eye objects (plastic caps, toys…).
        Uses fraction of high-saturation, non-biological-hue pixels.

    Rule 2 — Perfectly uniform inner zone (inner_std < 12.0):
        Catches metal lids, flat painted surfaces, coins, etc.
        The pupil/lens zone of every real eye (normal or cataract) has
        at least some brightness variation; painted metal is perfectly flat.

    Returns (True,  "")   — image passes, proceed to inference.
    Returns (False, msg)  — image rejected, show msg to user.
    Always fail-open on exceptions (return True) to never block real patients.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return True, ""   # let downstream handle missing file

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = gray.shape
        cx, cy  = w // 2, h // 2
        min_dim = min(h, w)

        Y, X    = np.ogrid[:h, :w]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2

        # ── Zone masks ────────────────────────────────────────────
        r_inner = int(min_dim * 0.13)   # pupil / lens zone
        r_focus = int(min_dim * 0.48)   # full eye analysis region

        inner_mask = dist_sq <= r_inner ** 2
        focus_mask = dist_sq <= r_focus ** 2

        # ── Rule 1 : vivid synthetic colour ──────────────────────
        sat_focus = hsv[:, :, 1][focus_mask]
        hue_focus = hsv[:, :, 0][focus_mask]
        vivid_synthetic = (
            float(np.mean((sat_focus > 120) & (hue_focus > 25) & (hue_focus < 155)))
            if sat_focus.size > 0 else 0.0
        )
        print(f"  [BioGate] vivid_synthetic={vivid_synthetic:.3f}")

        # ── Rule 2 : inner zone uniformity ───────────────────────
        inner_px  = gray[inner_mask]
        inner_std = float(np.std(inner_px)) if inner_px.size > 0 else 100.0
        print(f"  [BioGate] inner_std={inner_std:.1f}")

        # ── Thresholds (data-validated on 13 images) ─────────────
        VIVID_MAX  = 0.10   # real eyes: max 0.076  |  fake: 0.186
        INNER_MIN  = 12.0   # real eyes: min 24.0   |  fake:  9.5

        if vivid_synthetic > VIVID_MAX:
            print(f"  [BioGate] REJECT — vivid synthetic colour "
                  f"({vivid_synthetic:.3f} > {VIVID_MAX})")
            return False, (
                "This image appears to contain a brightly coloured non-eye object "
                "(such as a plastic bottle cap or toy). "
                "Please upload a clear, close-up photograph of your actual eye."
            )

        if inner_std < INNER_MIN:
            print(f"  [BioGate] REJECT — inner zone too uniform "
                  f"(inner_std={inner_std:.1f} < {INNER_MIN})")
            return False, (
                "This image appears to show a flat or painted object rather than "
                "a real eye. The central zone has no texture variation — "
                "no iris, pupil, or lens features were detected. "
                "Please upload a clear, close-up photograph of your open eye."
            )

        print("  [BioGate] PASS — biological eye tissue confirmed")
        return True, ""

    except Exception as e:
        print(f"  [BioGate] error (fail-open): {e}")
        return True, ""   # never block real patients on unexpected errors


# ── Grad-CAM helpers ───────────────────────────────────────────

def generate_feature_heatmap(image_path: str, output_path: str):
    """
    Fallback brightness-based opacity heatmap when Grad-CAM is unavailable
    (e.g., model has no Conv2d layers).  Highlights bright central regions
    using a Gaussian-weighted brightness map.
    """
    try:
        img  = cv2.imread(image_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        sigma  = min(h, w) * 0.30

        Y, X  = np.ogrid[:h, :w]
        gauss = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)
                       ).astype(np.float32)

        feat = gray.astype(np.float32) / 255.0 * gauss
        feat = cv2.GaussianBlur(feat, (21, 21), 0)
        if feat.max() > 0:
            feat = feat / feat.max()

        hm      = cv2.applyColorMap((feat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(img, 0.55, hm, 0.45, 0)
        cv2.imwrite(output_path, blended)
        return output_path
    except Exception as e:
        print(f"  [feature_heatmap] error: {e}")
        return None


def generate_gradcam(model, input_tensor, target_class_idx: int,
                     img_path: str, output_path: str):
    """
    Produce a Grad-CAM attention heatmap overlaid on the original image.

    Algorithm:
      1. Register a forward hook on the LAST Conv2d layer to capture
         its output activation map.
      2. Call retain_grad() on that activation so gradients are stored.
      3. Forward pass → compute class score → backward pass.
      4. Average-pool gradients over spatial dims → channel weights.
      5. Weight-sum activations, ReLU, resize, overlay as JET colormap.

    Falls back to brightness heatmap if:
      - No Conv2d layer is found in the model.
      - The backward pass fails for any reason.
    """
    model.eval()

    # Find the last Conv2d layer
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    if last_conv is None:
        print("  [GradCAM] no Conv2d found — using brightness fallback")
        return generate_feature_heatmap(img_path, output_path)

    captured = {}

    def fwd_hook(m, inp, out):
        captured["act"] = out
        out.retain_grad()   # keep grad on non-leaf activation tensor

    handle = last_conv.register_forward_hook(fwd_hook)

    try:
        # Fresh forward outside torch.no_grad() so the graph is alive
        inp    = input_tensor.detach().clone()
        output = model(inp)
        handle.remove()

        if "act" not in captured:
            return generate_feature_heatmap(img_path, output_path)

        model.zero_grad()
        score = output[0, target_class_idx]
        score.backward()

        act  = captured["act"].detach().cpu().numpy()[0]   # (C, H, W)
        grad = captured["act"].grad

        if grad is None:
            print("  [GradCAM] grad is None — using brightness fallback")
            return generate_feature_heatmap(img_path, output_path)

        grads   = grad.detach().cpu().numpy()[0]           # (C, H, W)
        weights = np.mean(grads, axis=(1, 2))              # (C,)
        cam     = np.einsum("c,chw->hw", weights, act)    # (H, W)
        cam     = np.maximum(cam, 0)                       # ReLU

        if cam.max() == 0:
            return generate_feature_heatmap(img_path, output_path)

        cam /= cam.max()

        img_cv  = cv2.imread(img_path)
        if img_cv is None:
            return None
        h_img, w_img = img_cv.shape[:2]
        cam_up  = cv2.resize(cam, (w_img, h_img))
        hm_img  = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(img_cv, 0.55, hm_img, 0.45, 0)
        cv2.imwrite(output_path, blended)
        print("  [GradCAM] heatmap saved OK")
        return output_path

    except Exception as e:
        print(f"  [GradCAM] error: {e}")
        try:
            handle.remove()
        except Exception:
            pass
        return generate_feature_heatmap(img_path, output_path)


# ══════════════════════════════════════════════════════════════
#  MAIN VALIDATOR  — is_eye_image()   ← DO NOT TOUCH
# ══════════════════════════════════════════════════════════════
def is_eye_image(image_path: str) -> tuple:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, (
                "Unable to read this file. "
                "Please upload a valid JPG, PNG, or similar image."
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h0, w0 = gray.shape
        print(f"\n[is_eye_image] {image_path}  {w0}x{h0}")

        if h0 < 50 or w0 < 50:
            return False, (
                "Image is too small to analyse. "
                "Please upload a photo that is at least 50x50 pixels."
            )

        if np.std(gray) < 5:
            return False, (
                "The image appears to be blank or a solid colour. "
                "Please upload a real photograph of an eye."
            )

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"  [Layer2] lap_var={lap_var:.0f}")
        if lap_var > MAX_LAP_VARIANCE:
            return False, (
                "This appears to be a screenshot or computer-generated image. "
                "Please upload a real photograph taken by a camera or phone."
            )

        hsv_img     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_chan       = hsv_img[:, :, 1].ravel().astype(np.float32)
        b_ch         = img[:, :, 0].ravel().astype(np.int32)
        g_ch         = img[:, :, 1].ravel().astype(np.int32)
        r_ch         = img[:, :, 2].ravel().astype(np.int32)

        hi_sat_frac = float(np.mean(s_chan > 200))
        skin_frac   = float(np.mean(
            (r_ch > 80) & (g_ch > 40) & (b_ch > 20) &
            (r_ch - g_ch > 5) & (r_ch - b_ch > 15) &
            (np.maximum(np.maximum(r_ch, g_ch), b_ch) -
             np.minimum(np.minimum(r_ch, g_ch), b_ch) > 15)
        ))
        rule_a = (hi_sat_frac > ILLUS_HI_SAT_THRESH) and (skin_frac < ILLUS_SKIN_THRESH)
        print(f"  [Layer2b] hi_sat={hi_sat_frac:.3f}  skin={skin_frac:.3f}  rule_a={rule_a}")

        if rule_a:
            return False, (
                "This looks like a cartoon or digital illustration, not a real eye photo. "
                "Please upload an actual photograph of your eye."
            )

        white_frac = float(np.mean(
            (r_ch > 235) & (g_ch > 235) & (b_ch > 235)
        ))
        print(f"  [Layer2c] white_bg_frac={white_frac:.3f}  threshold={WHITE_BG_THRESHOLD}")

        if white_frac > WHITE_BG_THRESHOLD:
            return False, (
                "This photo does not appear to be a close-up of an eye. "
                "It looks like a photograph of a hand, document, or another object "
                "taken against a bright background. "
                "Please upload a real close-up photograph of your eye."
            )

        img, gray = _crop_solid_borders(img, gray)
        h, w = gray.shape
        if h < 50 or w < 50:
            return False, (
                "The image appears to be entirely border or padding. "
                "Please upload a photo with visible eye content."
            )

        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        min_dim = min(h, w)
        max_dim = max(h, w)
        print(f"  [Layer3] cropped={w}x{h}  min_dim={min_dim}  max_dim={max_dim}")

        min_det  = max(25, int(min_dim * 0.07))
        all_dets = []
        for gimg in [gray, gray_eq]:
            for (x, y, ew, eh) in eye_cascade.detectMultiScale(
                    gimg, scaleFactor=1.05, minNeighbors=6,
                    minSize=(min_det, min_det)):
                all_dets.append((x + ew // 2, y + eh // 2, ew))

        print(f"  [Layer4-raw] haar_hits={len(all_dets)}")

        if all_dets:
            all_dets = _filter_small_dets(all_dets, min_size_ratio=0.40)

            prox   = max_dim * 0.40
            groups = _group_dets(all_dets, prox)
            strong = [g for g in groups if max(d[2] for d in g) >= min_dim * 0.05]
            print(f"  [Layer4] prox={prox:.0f}  groups={len(groups)}  strong={len(strong)}")

            if len(strong) == 1:
                print("  → PASS: exactly 1 strong Haar group")
                return True, ""

            if len(strong) > 1:
                scored = sorted(strong, key=_group_score, reverse=True)
                dom    = scored[0]
                dom_s  = _group_score(dom)
                dom_c  = _group_center(dom)

                interior_sec = [
                    g for g in scored[1:]
                    if not _near_border(*_group_center(g), w, h)
                    and _group_score(g) >= dom_s * 0.40
                ]
                print(f"  [Layer4-dom] dom_s={dom_s:.0f}  interior_sec={len(interior_sec)}")

                if not interior_sec:
                    print("  → PASS: secondary groups all weak or near-border")
                    return True, ""

                sec   = max(interior_sec, key=_group_score)
                sec_s = _group_score(sec)
                sec_c = _group_center(sec)

                if dom_s >= sec_s * 3.0:
                    print("  → PASS: primary dominates 3:1")
                    return True, ""

                sep        = np.hypot(dom_c[0] - sec_c[0], dom_c[1] - sec_c[1])
                sep_thresh = max_dim * 0.65
                print(f"  [Layer4-dom] sep={sep:.0f}  thresh={sep_thresh:.0f}")

                if sep > sep_thresh:
                    return False, (
                        "This photo appears to show more than one eye, or contains "
                        "a bright reflection that resembles a second eye. "
                        "Please upload a close-up of just ONE eye, "
                        "taken straight-on without strong side lighting."
                    )

                print(f"  → PASS: two close groups = same iris (sep={sep:.0f})")
                return True, ""

        print("  [Layer5] Haar empty — Hough fallback")
        blurred = cv2.GaussianBlur(gray_eq, (9, 9), 2)
        raw_circles = []
        for param2 in [40, 30, 22, 18, 14]:
            cc = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=int(min_dim * 0.30),
                param1=50, param2=param2,
                minRadius=int(min_dim * 0.10),
                maxRadius=int(min_dim * 0.68),
            )
            if cc is not None:
                raw_circles = np.round(cc[0]).astype(int).tolist()
                if len(raw_circles) <= 12:
                    break

        interior = [
            c for c in raw_circles
            if w * 0.08 <= c[0] <= w * 0.92
            and h * 0.08 <= c[1] <= h * 0.92
            and c[2] >= min_dim * 0.10
        ]
        print(f"  [Layer5] raw={len(raw_circles)}  interior={len(interior)}")

        if not interior:
            return False, (
                "No eye could be detected in this image. "
                "Please upload a clear, well-lit, front-facing close-up "
                "photograph of a single open eye."
            )

        dark_circles = []
        for c in interior:
            cx, cy, cr = int(c[0]), int(c[1]), int(c[2])
            mean_inside = _circle_interior_mean(gray, cx, cy, cr)
            print(f"  [Layer5b] circle cx={cx} cy={cy} r={cr}  "
                  f"mean_inside={mean_inside:.0f}  "
                  f"{'OK' if mean_inside <= HOUGH_CIRCLE_MAX_MEAN else 'BRIGHT'}")
            if mean_inside <= HOUGH_CIRCLE_MAX_MEAN:
                dark_circles.append(c)

        if not dark_circles:
            return False, (
                "This photo does not appear to contain an eye. "
                "The image may be a photograph of a hand, skin, or another object. "
                "Please upload a real close-up photograph of your eye."
            )

        interior = dark_circles

        groups = _group_dets(interior, max_dim * 0.40)
        groups_sorted = sorted(
            groups, key=lambda g: max(c[2] for c in g), reverse=True
        )
        print(f"  [Layer5] hough_groups={len(groups_sorted)} (after anatomy gate)")

        if len(groups_sorted) == 1:
            print("  → PASS via Hough (1 group)")
            return True, ""

        primary_r   = max(c[2] for c in groups_sorted[0])
        secondary_r = max(c[2] for c in groups_sorted[1])

        if secondary_r < primary_r * 0.70:
            print("  → PASS via Hough: secondary too small to be a real eye")
            return True, ""

        pc  = _group_center(groups_sorted[0])
        sc  = _group_center(groups_sorted[1])
        sep = np.hypot(pc[0] - sc[0], pc[1] - sc[1])
        print(f"  [Layer5] sep={sep:.0f}  thresh={max_dim * 0.65:.0f}")

        if sep > max_dim * 0.65:
            return False, (
                "This photo appears to show more than one eye, or contains "
                "a bright reflection that resembles a second eye. "
                "Please upload a close-up of just ONE eye "
                "without strong side lighting."
            )

        print("  → PASS via Hough (close groups = same iris)")
        return True, ""

    except Exception as e:
        return False, f"An error occurred while processing the image: {str(e)}"


# ══════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_data = None
    error_message   = None
    image_url       = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            error_message = "No file was selected. Please choose an eye image to upload."
            return render_template("index.html", error_message=error_message)

        if not allowed_file(file.filename):
            error_message = (
                "Unsupported file type. "
                "Please upload a JPG, PNG, WEBP, or BMP image."
            )
            return render_template("index.html", error_message=error_message)

        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)
        ext       = file.filename.rsplit(".", 1)[1].lower()
        safe_name = f"{uuid.uuid4().hex}.{ext}"
        img_path  = os.path.join(upload_folder, safe_name)
        file.save(img_path)
        image_url = "/" + img_path

        # ── Validate (shape-based — DO NOT TOUCH) ─────────────
        is_valid, err_msg = is_eye_image(img_path)
        if not is_valid:
            try:
                os.remove(img_path)
            except Exception:
                pass
            return render_template("index.html", error_message=err_msg, image_url=None)

        # ── Biological eye gate (post-shape, pre-inference) ────
        # Catches circular non-eye objects (caps, lids, coins…)
        # that fool the Haar/Hough detector.
        is_bio, bio_err = is_biological_eye(img_path)
        if not is_bio:
            try:
                os.remove(img_path)
            except Exception:
                pass
            return render_template("index.html", error_message=bio_err, image_url=None)

        # ── Inference ──────────────────────────────────────────
        image        = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        model_results  = []
        cataract_votes = []
        normal_votes   = []

        with torch.no_grad():
            for m_name in available_models:
                model = get_model(m_name)
                if not model:
                    continue
                output  = model(input_tensor)
                probs   = F.softmax(output, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                idx     = torch.argmax(probs, 1).item()
                conf    = probs[0][idx].item() * 100
                pred    = classes[idx]

                print(f"DEBUG {m_name} | entropy={entropy:.4f} | pred={pred} | conf={conf:.2f}")

                model_results.append({
                    "model":      m_name,
                    "prediction": pred,
                    "confidence": round(conf, 2),
                    "entropy":    round(entropy, 4),
                })
                (cataract_votes if pred == "Cataract" else normal_votes).append(conf)

        # ── Aggregate ──────────────────────────────────────────
        cataract_count = len(cataract_votes)
        normal_count   = len(normal_votes)
        final_pred     = "Cataract" if cataract_count > normal_count else "Normal"
        winning_votes  = cataract_votes if final_pred == "Cataract" else normal_votes

        avg_conf    = sum(winning_votes) / len(winning_votes) if winning_votes else 0
        entropies   = [m["entropy"] for m in model_results]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0

        final_result = {
            "prediction":     final_pred,
            "confidence":     round(avg_conf, 2),
            "model_count":    len(model_results),
            "cataract_votes": cataract_count,
            "avg_entropy":    round(avg_entropy, 4),
        }

        # ── Quality gates ──────────────────────────────────────
        if final_result["confidence"] < MIN_CONFIDENCE:
            return render_template(
                "index.html", image_url=image_url,
                error_message=(
                    f"The model is not confident enough in this image "
                    f"({final_result['confidence']:.1f}% confidence). "
                    "Please try a sharper, better-lit photo of the eye."
                ),
            )
        if final_result["avg_entropy"] > MAX_ENTROPY:
            return render_template(
                "index.html", image_url=image_url,
                error_message=(
                    "The AI models are uncertain about this image. "
                    "Please upload a clearer, well-focused close-up of the eye."
                ),
            )

        # ── Visual feature analysis ────────────────────────────
        print("  [VisualFeatures] analyzing eye characteristics…")
        eye_features = analyze_eye_features(img_path)
        print(f"  [VisualFeatures] {eye_features}")

        # ── Grad-CAM heatmap ───────────────────────────────────
        heatmap_url = None
        if model_results:
            best_m      = max(model_results, key=lambda x: x["confidence"])
            best_mdl    = get_model(best_m["model"])
            if best_mdl:
                hm_path    = os.path.join(upload_folder, f"hm_{safe_name}")
                target_cls = 0 if final_pred == "Cataract" else 1
                hm_result  = generate_gradcam(
                    best_mdl, input_tensor, target_cls, img_path, hm_path
                )
                if hm_result:
                    heatmap_url = "/" + hm_path
                    print(f"  [GradCAM] heatmap_url={heatmap_url}")

        # ── Generate LLM report ────────────────────────────────
        summary = get_groq_summary(final_result, model_results)

        # ── Three-tier explanation (NEW) ───────────────────────
        explanation = get_cataract_explanation(final_pred)

        prediction_data = {
            "final":        final_result,
            "individual":   model_results,
            "summary":      summary,
            "eye_features": eye_features,
            "heatmap_url":  heatmap_url,
            "explanation":  explanation,   # ← NEW
        }
        session["last_result"] = prediction_data

    return render_template(
        "index.html",
        prediction_data=prediction_data,
        image_url=image_url,
        error_message=error_message,
    )


# ── Chat memory (server-side, last 7 exchanges) ────────────────
_chat_memory = []

@app.route("/chat", methods=["POST"])
def chat():
    global _chat_memory
    user_msg      = request.json.get("message", "").strip()
    selected_lang = request.json.get("language", "English")
    last_result   = session.get("last_result")

    if not user_msg:
        return jsonify({"reply": "I did not receive your message. Please try again. 🤖"})
    if not GROQ_API_KEY:
        return jsonify({"reply": "My AI brain (Groq) is not configured. Please set GROQ_API_KEY. 🧠"})

    if len(_chat_memory) > 7:
        _chat_memory.pop(0)

    try:
        client  = Groq(api_key=GROQ_API_KEY)
        context = (
            "The user just scanned their eye. "
            f"Result: {json.dumps(last_result['final'] if last_result else 'No scan yet')}."
        )
        system_prompt = textwrap.dedent(f"""
            You are a friendly AI Eye Assistant who can give hope in the deepest dark times.
            Speak like a real person in plain everyday language.
            {context}
            Conversation memory (use for context): {_chat_memory}
            Keep responses VERY BRIEF — 2 to 4 sentences max.
            RESPOND ONLY IN {selected_lang.upper()} LANGUAGE.
            IF TELUGU: use only Telugu script. NO English letters (no Tenglish).
            IF HINDI: use only Devanagari script. NO English letters (no Hinglish).
            NO asterisks (*) or square brackets ([]).
            Be professional but friendly — like a trusted friend.
            If the user asks about their scan, refer to the data provided.
        """).strip()

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.5, max_tokens=500,
        )
        reply = completion.choices[0].message.content
        _chat_memory.append(f"User: {user_msg}")
        _chat_memory.append(f"AI: {reply}")
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Oops! I encountered an error: {str(e)} 🤖"})


if __name__ == "__main__":
    app.run(debug=True)