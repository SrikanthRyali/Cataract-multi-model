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
# ONLY haarcascade_eye.xml is used.
# haarcascade_righteye_2splits + haarcascade_lefteye_2splits are EXCLUDED:
# they are face-ROI-only cascades that fire on eyelashes and slit-lamp arcs.
_casc_dir   = cv2.data.haarcascades
eye_cascade = cv2.CascadeClassifier(_casc_dir + "haarcascade_eye.xml")

# ── Global thresholds ──────────────────────────────────────────
MIN_CONFIDENCE   = 30
MAX_ENTROPY      = 0.67
MAX_LAP_VARIANCE = 8000    # above → screenshot / noise

# Illustration-rejection (Layer 2b) — Rule A only
ILLUS_HI_SAT_THRESH = 0.75
ILLUS_SKIN_THRESH   = 0.08

# Non-eye / bright-background rejection (Layer 2c)
# ─────────────────────────────────────────────────────────────────────────────
# Real eye close-ups (slit-lamp, phone camera, fundus) NEVER have a large
# white/bright background — the image is dominated by iris + sclera tissue.
#
# Non-eye studio photos (hands, pills, charts, test cards) commonly have
# >35% pure-white background pixels because the subject was photographed
# on a light-box or white sheet.
#
# Measured values:
#   hand.jpg  white_bg=0.707   hand.webp white_bg=0.648
#   All eye images tested      white_bg < 0.05
#
# Threshold 0.35 gives a comfortable margin between the two populations.
WHITE_BG_THRESHOLD = 0.35   # fraction of (R>235, G>235, B>235) pixels

# Hough-anatomy gate (Layer 5b)
# ─────────────────────────────────────────────────────────────────────────────
# When Haar finds nothing and we rely on Hough circles, we add an additional
# check: for every candidate circle, the image region within that circle must
# NOT be uniformly very bright.  A real iris is always darker than 200 on
# average (even a white cataract has a dark iris ring).  A palm/finger joint
# is almost always > 180 average brightness.
HOUGH_CIRCLE_MAX_MEAN = 185  # mean brightness inside circle must be <= this

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
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════

def _crop_solid_borders(img, gray, std_thresh=18):
    """
    Remove solid-colour dataset border strips row-by-row / col-by-col.
    Never removes more than 35% from any side.
    """
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
    """
    Merge detection tuples (cx, cy, size) within `prox` pixels of each other.
    """
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
    """
    True if (cx, cy) lies within the outer 17% margin.
    17% covers slit-lamp corner beam artifacts (land at ~15-16% from edge).
    """
    return (cx < w * frac or cx > w * (1 - frac) or
            cy < h * frac or cy > h * (1 - frac))


def _filter_small_dets(dets, min_size_ratio=0.40):
    """
    Drop detections whose size < 40% of the largest.
    Removes eyelash rows, eyelid creases, and slit-lamp arc artifacts.
    """
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
    """
    Return the mean pixel brightness inside a circular region.
    Used to verify that a Hough circle contains dark iris tissue.
    """
    h, w = gray.shape
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    pixels = gray[mask]
    return float(np.mean(pixels)) if pixels.size > 0 else 255.0


# ══════════════════════════════════════════════════════════════
#  MAIN VALIDATOR  — is_eye_image()
#
#  Layer summary
#  ─────────────
#  LAYER 1  — Minimum size + blank/solid check
#  LAYER 2  — Laplacian upper bound (screenshots / pure noise)
#  LAYER 2b — Cartoon/illustration rejection (extreme HSV saturation)
#  LAYER 2c — ★ NEW: White-background / non-eye object rejection
#              Hands, pills, charts, test cards etc. photographed on
#              a light-box or white sheet have white_bg_frac > 0.35.
#              All real eye images tested have white_bg_frac < 0.05.
#  LAYER 3  — Strip solid-colour dataset border padding
#  LAYER 4  — Multi-pass Haar eye cascade with:
#               • Size filter (drops eyelash / arc hits < 40% of max)
#               • max_dim-based grouping proximity
#               • near_border frac = 0.17 (slit-lamp corner exclusion)
#               • Dominance + separation checks for multi-group cases
#  LAYER 5  — Hough-circle iris fallback (tight clinical crops only)
#               • Anatomy gate: every candidate circle must have mean
#                 brightness <= HOUGH_CIRCLE_MAX_MEAN (185)
#                 ★ NEW: rejects hands whose palm/knuckle circles are
#                 uniformly bright (mean 200-250) unlike iris (mean 60-150)
#
#  Validated acceptance:
#    ✅  Slit-lamp images (blurry, with illumination arc)
#    ✅  Wide close-up photos (900×368 type)
#    ✅  Fundus / ophthalmoscope crops
#    ✅  Phone camera close-ups
#    ✅  All 9 test dataset images
#
#  Validated rejection:
#    ❌  Hand photos (white background, bright Hough circles)
#    ❌  Both-eyes / full-face selfies
#    ❌  Screenshots / digital graphics
#    ❌  Cartoons / illustrations
#    ❌  Blank / solid / extreme-noise images
# ══════════════════════════════════════════════════════════════
def is_eye_image(image_path: str) -> tuple:
    """
    Validate that the uploaded image is a close-up photograph of ONE eye.
    Returns (is_valid: bool, error_message: str).
    error_message is "" when is_valid is True.
    """
    try:
        # ── READ ────────────────────────────────────────────────
        img = cv2.imread(image_path)
        if img is None:
            return False, (
                "Unable to read this file. "
                "Please upload a valid JPG, PNG, or similar image."
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h0, w0 = gray.shape
        print(f"\n[is_eye_image] {image_path}  {w0}x{h0}")

        # ── LAYER 1A: Minimum size ───────────────────────────────
        if h0 < 50 or w0 < 50:
            return False, (
                "Image is too small to analyse. "
                "Please upload a photo that is at least 50x50 pixels."
            )

        # ── LAYER 1B: Blank / solid colour ──────────────────────
        if np.std(gray) < 5:
            return False, (
                "The image appears to be blank or a solid colour. "
                "Please upload a real photograph of an eye."
            )

        # ── LAYER 2: Screenshot / digital noise guard ────────────
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"  [Layer2] lap_var={lap_var:.0f}")
        if lap_var > MAX_LAP_VARIANCE:
            return False, (
                "This appears to be a screenshot or computer-generated image. "
                "Please upload a real photograph taken by a camera or phone."
            )

        # ── LAYER 2b: Cartoon / illustration rejection ───────────
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

        # ── LAYER 2c: White-background / non-eye object rejection ─
        #
        # WHY THIS LAYER EXISTS
        # ─────────────────────
        # Photographs of hands, pills, ear charts, and other objects are
        # commonly taken on a white or bright background (light-box, paper).
        # These images pass all previous layers because:
        #   • Their Laplacian variance is normal (not a screenshot)
        #   • They have real skin tones (rule_b was already removed)
        #   • Haar finds no eyes (no false positive here — correct!)
        #   • Hough finds circles in palm lines / knuckle creases and passes
        #
        # A real eye close-up NEVER has a large white background because:
        #   • Slit-lamp images: dark background, bright iris ring
        #   • Phone close-ups: skin fills the frame around the eye
        #   • Fundus / ophthalmoscope: uniformly dark background
        #
        # MEASURED VALUES (R>235, G>235, B>235 pixel fraction):
        #   hand.jpg   = 0.707    hand.webp = 0.648
        #   All 9 eye test images < 0.05
        #   Threshold  = 0.35  (comfortable midpoint)
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

        # ── LAYER 3: Strip solid-colour dataset borders ─────────
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

        # ── LAYER 4: Haar eye detection ───────────────────────────
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

        # ── LAYER 5: Hough-circle iris fallback ─────────────────
        # Reached only when Haar finds nothing (very tight clinical crop).
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

        # ── LAYER 5b: Hough anatomy gate ★ NEW ──────────────────
        #
        # WHY THIS CHECK EXISTS
        # ──────────────────────
        # If Layer 2c (white background) somehow passes a non-eye image,
        # this is the final safety net before we accept a Hough result.
        #
        # A real iris is always dark tissue:
        #   • Normal iris: mean brightness 60-140
        #   • White cataract: mean brightness 120-170 (still has dark ring)
        #   • Average across all eye images: < 185
        #
        # A palm / knuckle / finger joint:
        #   • Always bright skin + white background
        #   • Mean brightness inside Hough circle: 190-250
        #
        # We compute the mean brightness inside EACH candidate circle.
        # If ALL circles are too bright → not an eye → reject.
        # If at least ONE circle has a sufficiently dark interior → accept.
        #
        # Threshold: HOUGH_CIRCLE_MAX_MEAN = 185
        # Measured:  hand.jpg circles → 170, 221, 229, 250
        #            hand.webp circles → 198, 207, 212, 226
        #            eye images  → 56–175 (at least one circle always qualifies)

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

        # Continue with only the valid dark-interior circles
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

        # ── Validate ───────────────────────────────────────────
        is_valid, err_msg = is_eye_image(img_path)
        if not is_valid:
            try:
                os.remove(img_path)
            except Exception:
                pass
            return render_template("index.html", error_message=err_msg, image_url=None)

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

        # ── Generate LLM report ────────────────────────────────
        summary = get_groq_summary(final_result, model_results)
        prediction_data = {
            "final":      final_result,
            "individual": model_results,
            "summary":    summary,
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