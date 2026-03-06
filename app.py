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

# ── Haar eye cascades ──────────────────────────────────────────
_casc_dir     = cv2.data.haarcascades
eye_cascade   = cv2.CascadeClassifier(_casc_dir + "haarcascade_eye.xml")
right_cascade = cv2.CascadeClassifier(_casc_dir + "haarcascade_righteye_2splits.xml")
left_cascade  = cv2.CascadeClassifier(_casc_dir + "haarcascade_lefteye_2splits.xml")

# ── Thresholds ─────────────────────────────────────────────────
MIN_CONFIDENCE   = 30     # % — below this result is unreliable
MAX_ENTROPY      = 0.67   # above this model is too uncertain
MAX_LAP_VARIANCE = 8000   # Laplacian variance above this → random noise / screenshot
# Illustration / non-photo rejection thresholds (Layer 2b)
# Real eye photos always have warm/skin-toned pixels (iris surround, sclera, eyelids).
# Digital illustrations (cars, cartoons, graphics) have almost none.
# Rule A: heavily saturated image with no skin → digital illustration
# Rule B: neither skin tones nor warm tones present → not a biological photo
ILLUS_HI_SAT_THRESH  = 0.60   # fraction of pixels with HSV-S > 200
ILLUS_SKIN_THRESH    = 0.15   # fraction of skin-toned pixels
ILLUS_WARM_THRESH    = 0.10   # fraction of warm-toned pixels (broader than skin)

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
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"


# ══════════════════════════════════════════════════════════════
#  is_eye_image() — 5-layer robust single-eye validation
#
#  Accepts:  clinical/dataset eye images of all types:
#            slit-lamp, ophthalmoscope, phone camera, close-ups,
#            images with solid-colour dataset border strips,
#            cataract eyes (low Laplacian — NOT used as lower bound)
#
#  Rejects:  non-eye uploads — screenshots, cars, random images,
#            faces with two eyes, blank/solid, random noise
#
#  LAYER 1 — Size & blank check
#  LAYER 2 — Noise/screenshot rejection via Laplacian upper bound
#             (real biological eyes always have lap < 8000;
#              random noise, JPEG screenshots hit 10,000–100,000)
#  LAYER 2b— Digital illustration / non-photo rejection
#             Real eye photos always have warm/skin-toned pixels around
#             the iris (eyelids, sclera, skin). Digital art (cars,
#             cartoons, graphics) has almost none.
#             Rule A: hi_sat_frac > 0.60 AND skin_frac < 0.15
#             Rule B: skin_frac < 0.10 AND warm_frac < 0.10
#  LAYER 3 — Strip solid-colour dataset border artifacts
#  LAYER 4 — Haar cascade (6-pass: 3 cascades × raw + CLAHE)
#             Groups detections; uses score-dominance and border-
#             proximity filtering to suppress eyelash/texture noise.
#  LAYER 5 — Hough-circle iris fallback for extreme close-ups
#             where cascades miss; filters edge circles, discards
#             sub-dominant circles, rejects only when two large
#             groups are genuinely far apart.
# ══════════════════════════════════════════════════════════════

def _crop_solid_borders(img, gray, std_thresh=18):
    """Remove solid-colour dataset border strips (column/row by column/row)."""
    h, w = gray.shape
    t, b, l, r = 0, h, 0, w
    max_frac = 0.35   # never remove more than 35% from any side

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
    """Score = detection count × max radius (larger groups score higher)."""
    return len(g) * max(d[2] for d in g)


def _group_center(g):
    return (sum(d[0] for d in g) / len(g), sum(d[1] for d in g) / len(g))


def _near_border(cx, cy, w, h, frac=0.15):
    return cx < w * frac or cx > w * (1 - frac) or cy < h * frac or cy > h * (1 - frac)


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
        # have Laplacian variance well below 8,000.
        # Random noise images, JPEG-compressed screenshots, and non-photo
        # content regularly exceed 10,000–100,000.
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var > MAX_LAP_VARIANCE:
            return (
                False,
                "❌ Invalid Image: Image appears to be a screenshot or digital graphic. "
                "Please upload a real photograph of an eye.",
            )

        # ── LAYER 2b: Reject digital illustrations / non-photos ──
        # Real eye photos always have warm/skin-toned pixels (iris surround,
        # sclera, eyelids). Cars, cartoons, and other digital art have almost
        # none. Two independent rules — either one alone is enough to reject.
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # reused below
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_chan   = hsv_img[:, :, 1].ravel().astype(np.float32)
        b_ch     = img[:, :, 0].ravel().astype(np.int32)
        g_ch     = img[:, :, 1].ravel().astype(np.int32)
        r_ch     = img[:, :, 2].ravel().astype(np.int32)

        hi_sat_frac = float(np.mean(s_chan > 200))
        skin_frac   = float(np.mean(
            (r_ch > 80) & (g_ch > 40) & (b_ch > 20) &
            (r_ch - g_ch > 5) & (r_ch - b_ch > 15) &
            (np.maximum(np.maximum(r_ch, g_ch), b_ch) -
             np.minimum(np.minimum(r_ch, g_ch), b_ch) > 15)
        ))
        warm_frac   = float(np.mean((r_ch - b_ch > 20) & (r_ch > 100)))

        # Rule A: very high flat saturation + almost no skin → digital art
        rule_a = (hi_sat_frac > ILLUS_HI_SAT_THRESH) and (skin_frac < ILLUS_SKIN_THRESH)
        # Rule B: neither skin tones nor warm tones → not a biological photo
        rule_b = (skin_frac < 0.10) and (warm_frac < ILLUS_WARM_THRESH)

        if rule_a or rule_b:
            return (
                False,
                "❌ Invalid Image: This does not appear to be a photograph of an eye. "
                "Please upload a real close-up eye photo.",
            )

        # ── LAYER 3: Strip solid-colour dataset borders ────────
        img, gray = _crop_solid_borders(img, gray)
        h, w = gray.shape
        if h < 50 or w < 50:
            return False, "❌ Invalid Image: No valid eye content found after removing borders."

        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        min_dim = min(h, w)
        max_dim = max(h, w)

        # ── LAYER 4: Multi-cascade Haar detection ─────────────
        configs = [
            (eye_cascade,   gray,    5),
            (eye_cascade,   gray_eq, 5),
            (right_cascade, gray,    3),
            (right_cascade, gray_eq, 3),
            (left_cascade,  gray,    3),
            (left_cascade,  gray_eq, 3),
        ]
        all_dets = []
        for casc, gimg, nn in configs:
            for (x, y, ew, eh) in casc.detectMultiScale(
                    gimg, scaleFactor=1.1, minNeighbors=nn, minSize=(20, 20)):
                all_dets.append((x + ew // 2, y + eh // 2, ew))

        if all_dets:
            groups = _group_dets(all_dets, min_dim * 0.55)
            strong = [g for g in groups if max(d[2] for d in g) >= min_dim * 0.05]

            if len(strong) == 1:
                return True, ""

            if len(strong) > 1:
                # Sort groups by score (count × max_radius) descending
                scored = sorted(strong, key=_group_score, reverse=True)
                dom      = scored[0]
                dom_s    = _group_score(dom)
                dom_c    = _group_center(dom)

                # Discard secondary groups that:
                #   (a) hug the image border (eyelashes / skin texture hits), OR
                #   (b) score < 30% of dominant (clearly weaker detections)
                interior_sec = [
                    g for g in scored[1:]
                    if not _near_border(*_group_center(g), w, h, 0.15)
                    and _group_score(g) >= dom_s * 0.30
                ]

                # No real secondary candidates → treat as single eye
                if not interior_sec:
                    return True, ""

                # Dominant group is 3× stronger than any secondary → single eye
                sec_s = max(_group_score(g) for g in interior_sec)
                if dom_s >= sec_s * 3.0:
                    return True, ""

                # Check angular separation between dominant and strongest secondary
                sec   = max(interior_sec, key=_group_score)
                sec_c = _group_center(sec)
                sep   = np.sqrt((dom_c[0] - sec_c[0]) ** 2 + (dom_c[1] - sec_c[1]) ** 2)

                if sep > min_dim * 0.65:
                    return (
                        False,
                        f"❌ Invalid Image: {len(strong)} eyes detected. "
                        "Please upload a close-up photo of ONE eye only.",
                    )
                # Groups are close together → same iris, multiple cascade hits
                return True, ""

        # ── LAYER 5: Hough-circle iris fallback ───────────────
        # For extreme close-ups where cascades find nothing.
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
                    break   # use first level that gives a manageable count

        # Keep only circles centred in the inner 80% of the image
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

        # Dominant iris = largest interior circle
        dom_r       = max(c[2] for c in interior)
        significant = [c for c in interior if c[2] >= dom_r * 0.60]

        groups = _group_dets(significant, min_dim * 0.55)

        if len(groups) == 1:
            return True, ""

        # Multiple groups: only reject if a second group is truly iris-sized
        group_max_r = sorted([max(c[2] for c in g) for g in groups], reverse=True)
        primary_r   = group_max_r[0]

        # Secondary must be ≥75% of primary's radius to count as a real second iris
        real_secondary = [r for r in group_max_r[1:] if r >= primary_r * 0.75]
        if not real_secondary:
            return True, ""   # secondary groups are reflections / noise

        # Measure separation between the two strongest group centres
        group_centers = [_group_center(g) for g in groups]
        sep = np.sqrt(
            (group_centers[0][0] - group_centers[1][0]) ** 2
            + (group_centers[0][1] - group_centers[1][1]) ** 2
        )

        # Two real irises are separated by more than 55% of the longer image dimension
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

        # Save with UUID to prevent filename collisions / special-char issues
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
        return jsonify({"reply": "I didn't receive your message. Please try again. 🤖"})
    if not GROQ_API_KEY:
        return jsonify({"reply": "My AI brain (Groq) is not configured. Please set GROQ_API_KEY! 🧠"})

    try:
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
        return jsonify({"reply": completion.choices[0].message.content})

    except Exception as e:
        return jsonify({"reply": f"Oops! I encountered an error: {str(e)} 🤖"})


if __name__ == "__main__":
    app.run(debug=True)