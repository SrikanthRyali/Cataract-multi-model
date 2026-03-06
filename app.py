import os
import uuid
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template, jsonify, session
from PIL import Image
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
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
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_KEY='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'

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

# ── Haar eye cascades ──────────────────────────────────────────
_casc_dir     = cv2.data.haarcascades
eye_cascade   = cv2.CascadeClassifier(_casc_dir + "haarcascade_eye.xml")
right_cascade = cv2.CascadeClassifier(_casc_dir + "haarcascade_righteye_2splits.xml")
left_cascade  = cv2.CascadeClassifier(_casc_dir + "haarcascade_lefteye_2splits.xml")

# ── Thresholds ─────────────────────────────────────────────────
MIN_CONFIDENCE   = 30     # % — below this result is unreliable
MAX_ENTROPY      = 0.67   # above this model is too uncertain
MAX_LAP_VARIANCE = 9500   # Laplacian variance above this → random noise / screenshot
                          # (raised from 8000 → allows sharp clinical scans like cat_0_1897)

# ── Layer 2b: Digital illustration rejection ──────────────────
# Rule A: very high flat saturation + no skin → digital art (car, cartoon)
ILLUS_HI_SAT_THRESH = 0.60   # fraction of pixels with HSV-S > 200
ILLUS_SKIN_THRESH   = 0.15   # fraction of skin-toned pixels

# ── Layer 2c: Hand / body-part rejection ──────────────────────
# Rule B: no bio-tissue colours AND no eye structure at all
ILLUS_WARM_THRESH   = 0.10   # fraction of warm-toned pixels

# Rule C: Positive iris confirmation — applies when cascade finds nothing
# ring_e = fraction of Hough circle perimeter coinciding with Canny edges
#   Real irises (limbal ring):         ring_e ≥ 0.24
#   Hands / legs / body parts:         ring_e ≤ 0.20
# Threshold 0.22 sits in the middle of the gap observed across all test images.
IRIS_RING_EDGE_THRESH = 0.22   # ring edge fraction below this → no limbal ring = not iris
IRIS_CX_THRESH        = 0.25   # dominant circle x-offset > this → off-centre = not iris
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
        print("❌ Groq API Key not configured!")
        return "Groq API Key not configured. Please set the GROQ_API_KEY environment variable."
    try:
        print(f"🔄 Generating Groq summary for prediction: {final_result['prediction']} ({final_result['confidence']}%)")
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
- **Blinking:** Blink fast 10×, close eyes 20 sec. Repeat 5×.
- **Eye Rotation:** Look up → right → down → left slowly.
- **Palming:** Rub hands warm, place over closed eyes gently.

## 💰 Surgery Costs (India)
- **Basic (SICS):** ₹15,000–₹25,000 — small incision, very safe.
- **Advanced (Phaco):** ₹40,000–₹80,000 — no-stitch, fast recovery.
- **Laser/Robot:** ₹1,00,000+ — extreme precision.
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
- **Sleep:** 7–8 hours protects eye health.

## 📖 Stay Proactive
- **Yearly Scan:** Good habit even with normal results.
- **Vision Changes:** If things go blurry suddenly, see a doctor.
"""

        pred_label = "Normal (Healthy)" if final_result["prediction"] == "Normal" else "Cataract Detected"
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
        print(f"🔍 Groq API Response: {completion.choices[0].message.content[:200]}...")
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq API Error: {str(e)}")
        return f"Error generating summary: {str(e)}"


# ══════════════════════════════════════════════════════════════
#  is_eye_image() — 6-layer robust single-eye validation
#
#  Accepts:  slit-lamp, ophthalmoscope, phone camera, close-ups,
#            dataset images with border strips, cataract eyes,
#            animal eyes, extremely dark or pale irises
#
#  Rejects:  hands, legs, fruits, cars, cartoons, screenshots,
#            faces with two eyes, blank/solid, random noise
#
#  LAYER 1 — Size & blank check
#  LAYER 2 — Noise/screenshot rejection (Laplacian upper bound ≤ 9500)
#  LAYER 2b— Digital illustration rejection
#             Rule A: hi_sat > 0.60 AND skin < 0.15  →  car/cartoon
#  LAYER 2c— Hand / body-part rejection
#             Rule B: skin < 0.15 AND warm < 0.10
#                     AND no cascade detections AND no Hough circle
#             Rule C: cascade found nothing (n_casc == 0)
#                     AND (cx_offset > 0.25 OR palm-line texture
#                          OR ring_edge_frac < 0.22 — no limbal ring)
#                     Works for all skin tones (no skin-fraction gate)
#  LAYER 3 — Strip solid-colour dataset border artifacts
#  LAYER 4 — Haar cascade (6-pass: 3 cascades × raw + CLAHE)
#             Score-dominance + border-proximity + max_dim separation
#  LAYER 5 — Hough-circle iris fallback for extreme close-ups
# ══════════════════════════════════════════════════════════════

def _crop_solid_borders(img, gray, std_thresh=18):
    """Remove solid-colour dataset border strips (column/row by column/row)."""
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
    """Group (cx, cy, r) detections within `prox` pixels of each other."""
    groups = []
    for d in dets:
        placed = False
        for g in groups:
            gc = (sum(x[0] for x in g) / len(g), sum(x[1] for x in g) / len(g))
            if np.sqrt((d[0] - gc[0]) ** 2 + (d[1] - gc[1]) ** 2) < prox:
                g.append(d); placed = True; break
        if not placed:
            groups.append([d])
    return groups


def _group_score(g):
    """Score = detection count × max radius."""
    return len(g) * max(d[2] for d in g)


def _group_center(g):
    return (sum(d[0] for d in g) / len(g), sum(d[1] for d in g) / len(g))


def _near_border(cx, cy, w, h, frac=0.15):
    return cx < w * frac or cx > w * (1 - frac) or cy < h * frac or cy > h * (1 - frac)


def _run_cascade(gray, gray_eq):
    """Run all 6 Haar eye cascade passes and return raw detections."""
    dets = []
    for casc, gimg, nn in [
        (eye_cascade,   gray,    5), (eye_cascade,   gray_eq, 5),
        (right_cascade, gray,    3), (right_cascade, gray_eq, 3),
        (left_cascade,  gray,    3), (left_cascade,  gray_eq, 3),
    ]:
        for (x, y, ew, eh) in casc.detectMultiScale(
                gimg, scaleFactor=1.1, minNeighbors=nn, minSize=(20, 20)):
            dets.append((x + ew // 2, y + eh // 2, ew))
    return dets


def _hough_dominant(gray, h, w):
    """
    Run Hough detection and return properties of the dominant interior circle.
    Returns (contrast, cx_offset, inside_std, circles, lap_in, ring_e) or None.
      contrast   = outside_mean − inside_mean (positive → inside darker = iris-like)
      cx_offset  = |cx/w − 0.5|              (0=centred, high=near edge)
      inside_std = std-dev of pixels inside   (low → uniform = palm-like, not iris)
      lap_in     = Laplacian variance in ROI  (high = fine texture / palm lines)
      ring_e     = fraction of circle perimeter coinciding with Canny edges
                   (high = real limbal ring, low = body-part Hough artefact)
    """
    mn  = min(h, w)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    geq     = clahe.apply(gray)
    blurred = cv2.GaussianBlur(geq, (9, 9), 2)
    for p2 in [40, 30, 22, 18, 14]:
        cc = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=int(mn * 0.28), param1=50, param2=p2,
            minRadius=int(mn * 0.09), maxRadius=int(mn * 0.65),
        )
        if cc is not None:
            raw      = np.round(cc[0]).astype(int).tolist()
            interior = [c for c in raw
                        if w * 0.10 <= c[0] <= w * 0.90
                        and h * 0.10 <= c[1] <= h * 0.90
                        and c[2] >= mn * 0.10]
            if interior:
                dom    = max(interior, key=lambda c: c[2])
                mask   = np.zeros_like(gray)
                cv2.circle(mask, (dom[0], dom[1]), dom[2], 255, -1)
                ins    = gray[mask == 255]
                out    = gray[mask == 0]
                in_m   = float(np.mean(ins))
                out_m  = float(np.mean(out)) if len(out) > 100 else in_m
                roi     = gray[
                    max(0, dom[1]-dom[2]) : dom[1]+dom[2],
                    max(0, dom[0]-dom[2]) : dom[0]+dom[2],
                ]
                lap_in  = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size > 0 else 0.0
                # Ring edge fraction: how much of the circle's perimeter aligns with real edges.
                # A real iris has a strong limbal ring → high ring_e.
                # A body-part Hough circle is fit to random texture → low ring_e.
                edge_map   = cv2.Canny(geq, 30, 80)
                ring_mask  = np.zeros_like(gray)
                cv2.circle(ring_mask, (dom[0], dom[1]), dom[2], 255, 6)
                ring_e = (float(np.mean(edge_map[ring_mask > 0] > 0))
                          if np.sum(ring_mask > 0) > 0 else 0.0)
                return (out_m - in_m, abs(dom[0] / w - 0.5), float(np.std(ins)), interior, lap_in, ring_e)
    return None


def is_eye_image(image_path):
    try:
        # ── LAYER 1: Size & blank ──────────────────────────────
        img = cv2.imread(image_path)
        if img is None:
            return False, "❌ Invalid Image: Unable to read file. Please upload a valid image."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h0, w0 = gray.shape

        if h0 < 50 or w0 < 50:
            return False, "❌ Invalid Image: Too small (minimum 50×50 px)."
        if np.std(gray) < 5:
            return False, "❌ Invalid Image: Image appears blank or solid-colour."

        # ── LAYER 2: Reject random noise / screenshots ─────────
        # Real biological eye images (even very blurry slit-lamp shots)
        # have Laplacian variance well below 9,500.
        # Random noise and JPEG-compressed screenshots exceed 10,000–100,000.
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var > MAX_LAP_VARIANCE:
            return (
                False,
                "❌ Invalid Image: Image appears to be a screenshot or digital graphic. "
                "Please upload a real photograph of an eye.",
            )

        # ── LAYER 2b / 2c: Colour-based non-eye rejection ─────
        # Compute colour features once — shared by Rules A, B, and C.
        hsv_img     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_chan       = hsv_img[:, :, 1].ravel().astype(np.float32)
        b_ch         = img[:, :, 0].ravel().astype(np.int32)
        g_ch         = img[:, :, 1].ravel().astype(np.int32)
        r_ch         = img[:, :, 2].ravel().astype(np.int32)

        hi_sat_frac  = float(np.mean(s_chan > 200))
        skin_frac    = float(np.mean(
            (r_ch > 80) & (g_ch > 40) & (b_ch > 20) &
            (r_ch - g_ch > 5) & (r_ch - b_ch > 15) &
            (np.maximum(np.maximum(r_ch, g_ch), b_ch) -
             np.minimum(np.minimum(r_ch, g_ch), b_ch) > 15)
        ))
        warm_frac    = float(np.mean((r_ch - b_ch > 20) & (r_ch > 100)))

        # ── Rule A: Digital illustration (car, cartoon, graphic) ──
        # Extremely high flat saturation with almost no skin = digital art.
        if hi_sat_frac > ILLUS_HI_SAT_THRESH and skin_frac < ILLUS_SKIN_THRESH:
            return (
                False,
                "❌ Invalid Image: This appears to be a digital illustration, not an eye photo. "
                "Please upload a real close-up photograph of an eye.",
            )

        # ── Pre-resize: cap very large images before heavy OpenCV work ──
        # Images > 1200px on longest side are downscaled for cascade/Hough.
        # This prevents STATUS_ACCESS_VIOLATION crashes on large slit-lamp
        # scans and cuts detection time by 4–9×.  The aspect ratio is kept.
        MAX_PROC_DIM = 1200
        scale_factor = 1.0
        if max(h0, w0) > MAX_PROC_DIM:
            scale_factor = MAX_PROC_DIM / max(h0, w0)
            new_w = int(w0 * scale_factor)
            new_h = int(h0 * scale_factor)
            img_s  = cv2.resize(img,  (new_w, new_h), interpolation=cv2.INTER_AREA)
            gray_s = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_s, gray_s = img, gray

        hs, ws = gray_s.shape

        # Rules B & C: run cascade + Hough once on the (possibly scaled) image.
        # Results are CACHED and reused in Layers 4 & 5 — no duplicate work.
        clahe_pre   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq_pre = clahe_pre.apply(gray_s)
        early_dets  = _run_cascade(gray_s, gray_eq_pre)   # cached for Layer 4
        n_casc      = len(early_dets)
        hough_info  = _hough_dominant(gray_s, hs, ws)     # cached for Layer 5

        # ── Rule B: No biological tissue AND no eye structure ──
        if (skin_frac < ILLUS_SKIN_THRESH and warm_frac < ILLUS_WARM_THRESH
                and n_casc == 0 and hough_info is None):
            return (
                False,
                "❌ Invalid Image: This does not appear to be a photograph of an eye. "
                "Please upload a real close-up eye photo.",
            )

        # ── Rule C: Require positive iris evidence when cascade is silent ──
        # When the Haar cascade finds nothing (n_casc == 0), we demand the
        # Hough circle shows actual iris/limbal-ring structure.  Three checks:
        #   (a) Off-centre circle (cx_offset > 0.25) → background blob, not iris
        #   (b) Bright+noisy inside (contrast < -5 AND lap_in > 1000) → palm lines
        #   (c) Weak ring edge (ring_e < 0.22) → no limbal ring = not an eye
        #       ring_e = fraction of the circle's perimeter that is a Canny edge
        #       Real irises always score ≥ 0.24; hands/legs score ≤ 0.20.
        # This fires regardless of skin colour so dark-skin hands/legs are caught too.
        if n_casc == 0:
            if hough_info is not None:
                contrast, cx_off, _, _, lap_in, ring_e = hough_info
                no_real_iris = (
                    (cx_off > IRIS_CX_THRESH)              # off-centre blob
                    or (contrast < -5 and lap_in > 1000)   # palm-line texture
                    or (ring_e < IRIS_RING_EDGE_THRESH)    # no limbal ring
                )
            else:
                # No Hough circle at all — if there's any flesh or warm tone
                # it's almost certainly a body part (real dark eyes still produce circles)
                no_real_iris = (skin_frac > 0.20 or warm_frac > 0.20)
            if no_real_iris:
                return (
                    False,
                    "❌ Invalid Image: This does not appear to be a close-up eye photo. "
                    "Please upload a clear photograph of a single open eye.",
                )

        # ── LAYER 3: Strip solid-colour dataset borders ────────
        # Work on the scaled image from here on — fast and memory-safe.
        img_s, gray_s = _crop_solid_borders(img_s, gray_s)
        h, w = gray_s.shape
        if h < 50 or w < 50:
            return False, "❌ Invalid Image: No valid eye content found after removing borders."

        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray_s)
        min_dim = min(h, w)
        max_dim = max(h, w)

        # ── LAYER 4: Multi-cascade Haar detection ─────────────
        # Reuse cached detections if no border cropping changed the image size.
        # If crop changed dimensions, re-run on the cropped version.
        if (h, w) == (hs, ws):
            all_dets = early_dets   # reuse — no crop happened
        else:
            all_dets = _run_cascade(gray_s, gray_eq)   # re-run after crop

        if all_dets:
            groups = _group_dets(all_dets, min_dim * 0.55)
            strong = [g for g in groups if max(d[2] for d in g) >= min_dim * 0.05]

            if len(strong) == 1:
                return True, ""

            if len(strong) > 1:
                scored = sorted(strong, key=_group_score, reverse=True)
                dom      = scored[0]
                dom_s    = _group_score(dom)
                dom_c    = _group_center(dom)

                # Discard groups hugging the image border (eyelash/skin hits)
                # or scoring < 30% of the dominant group
                interior_sec = [
                    g for g in scored[1:]
                    if not _near_border(*_group_center(g), w, h, 0.15)
                    and _group_score(g) >= dom_s * 0.30
                ]

                if not interior_sec:
                    return True, ""

                sec_s = max(_group_score(g) for g in interior_sec)
                if dom_s >= sec_s * 3.0:
                    return True, ""

                sec   = max(interior_sec, key=_group_score)
                sec_c = _group_center(sec)
                sep   = np.sqrt((dom_c[0] - sec_c[0]) ** 2 + (dom_c[1] - sec_c[1]) ** 2)

                # Use max_dim (not min_dim) so tall/portrait images aren't
                # wrongly rejected when the iris appears in different quadrants
                if sep > max_dim * 0.65:
                    return (
                        False,
                        f"❌ Invalid Image: {len(strong)} eyes detected. "
                        "Please upload a close-up photo of ONE eye only.",
                    )
                return True, ""

        # ── LAYER 5: Hough-circle iris fallback ───────────────
        # Reuse cached Hough result if no border crop changed dimensions.
        if hough_info is not None and (h, w) == (hs, ws):
            interior = hough_info[3]   # index 3 = circles list (contrast,cx,std,circles,lap,ring_e)
        else:
            blurred = cv2.GaussianBlur(gray_eq, (9, 9), 2)
            raw_circles = []
            for param2 in [40, 30, 22, 18, 14]:
                cc = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                    minDist=int(min_dim * 0.28), param1=50, param2=param2,
                    minRadius=int(min_dim * 0.09), maxRadius=int(min_dim * 0.65),
                )
                if cc is not None:
                    raw_circles = np.round(cc[0]).astype(int).tolist()
                    if len(raw_circles) <= 12:
                        break
            interior = [
                c for c in raw_circles
                if w * 0.10 <= c[0] <= w * 0.90
                and h * 0.10 <= c[1] <= h * 0.90
                and c[2] >= min_dim * 0.10
            ]
        if not interior:
            return (
                False,
                "❌ Invalid Image: No eye detected. "
                "Please upload a clear, close-up photo of a single open eye.",
            )

        dom_r       = max(c[2] for c in interior)
        significant = [c for c in interior if c[2] >= dom_r * 0.60]
        groups      = _group_dets(significant, min_dim * 0.55)

        if len(groups) == 1:
            return True, ""

        group_max_r    = sorted([max(c[2] for c in g) for g in groups], reverse=True)
        primary_r      = group_max_r[0]
        real_secondary = [r for r in group_max_r[1:] if r >= primary_r * 0.75]

        if not real_secondary:
            return True, ""

        group_centers = [_group_center(g) for g in groups]
        sep = np.sqrt(
            (group_centers[0][0] - group_centers[1][0]) ** 2
            + (group_centers[0][1] - group_centers[1][1]) ** 2
        )

        if sep > max_dim * 0.55:
            return (
                False,
                "❌ Invalid Image: 2 eyes detected. "
                "Please upload a photo of ONE eye only.",
            )

        return True, ""

    except Exception as e:
        return False, f"❌ Invalid Image: Error processing file — {str(e)}"


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
            error_message = "❌ No file selected. Please choose an eye image."
            return render_template("index.html", error_message=error_message)

        if not allowed_file(file.filename):
            error_message = (
                "❌ Unsupported file type. "
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
            try: os.remove(img_path)
            except Exception: pass
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
                    f"⚠️ Low confidence ({final_result['confidence']:.1f}%). "
                    "Model is unsure — please try a clearer image."
                ),
            )
        if final_result["avg_entropy"] > MAX_ENTROPY:
            return render_template(
                "index.html", image_url=image_url,
                error_message=(
                    f"⚠️ Model uncertainty too high ({final_result['avg_entropy']:.3f}). "
                    "Please upload a clearer eye image."
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


@app.route("/chat", methods=["POST"])
def chat():
    user_msg      = request.json.get("message", "").strip()
    selected_lang = request.json.get("language", "English")
    last_result   = session.get("last_result")

    if not user_msg:
        print("❌ Chat: Empty message received")
        return jsonify({"reply": "I didn't receive your message. Please try again. 🤖"})
    if not GROQ_API_KEY:
        print("❌ Chat: Groq API Key not configured!")
        return jsonify({"reply": "My AI brain (Groq) is not configured. Please set GROQ_API_KEY! 🧠"})

    try:
        print(f"💬 Processing chat message in {selected_lang}: '{user_msg[:50]}...'")
        client  = Groq(api_key=GROQ_API_KEY)
        context = (
            "The user just scanned their eye. "
            f"Result: {json.dumps(last_result['final'] if last_result else 'No scan yet')}."
        )
        system_prompt = textwrap.dedent(f"""
            You are a friendly AI Eye Assistant. Speak like a real person in plain everyday language.
            {context}
            Keep responses VERY BRIEF — 2 to 4 sentences max.
            RESPOND ONLY IN {selected_lang.upper()} LANGUAGE.
            IF TELUGU: use only Telugu script (తెలుగు లిపి). NO English letters.
            IF HINDI: use only Devanagari script (देवनागरी). NO English letters.
            Include EXACTLY ONE emoji in your entire response.
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
        print(f"💬 Chat Response ({selected_lang}): {reply[:100]}...")
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"❌ Chat API Error: {str(e)}")
        return jsonify({"reply": f"Oops! I encountered an error: {str(e)} 🤖"})


if __name__ == "__main__":
    app.run(debug=True)