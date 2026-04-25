import streamlit as st
from PIL import Image
import numpy as np
import time
import os

# ── CONFIDENCE THRESHOLD ───────────────────────────────────────────────────────
# If the model's top-1 softmax probability is below this value, the image is
# rejected as unrelated to eggplant. Raise it (e.g. 0.75) to be stricter,
# lower it (e.g. 0.45) to be more permissive.
CONFIDENCE_THRESHOLD = 0.55

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Yieldy",
    page_icon="🌱",
    layout="centered",
)

# ── DISEASE DATABASE ───────────────────────────────────────────────────────────

DISEASE_INFO = {
    "Eggplant Healthy Leaf": {
        "cause": "None — plant is healthy",
        "symptom": "No disease detected; foliage appears normal",
        "actions": [
            "Continue regular monitoring every 3–5 days.",
            "Maintain a balanced fertilization schedule (N-P-K).",
            "Ensure consistent, even soil moisture.",
        ],
        "prevention": [
            "Keep up with preventive scouting routines.",
            "Monitor weather forecasts for sudden humidity changes.",
            "Maintain proper plant spacing for airflow.",
            "Remove weeds regularly to reduce pest harborage.",
        ],
        "severity": "None",
        "color": "#27ae60",
    },
    "Eggplant Insect Pest Disease": {
        "cause": "Various insect pests (aphids, thrips, mites, borers)",
        "symptom": "Visible pest damage, holes, frass, stippling, or distorted foliage",
        "actions": [
            "Identify the specific pest before applying any treatment.",
            "Apply appropriate insecticide (e.g., spinosad for borers, imidacloprid for sucking pests).",
            "Remove and destroy heavily infested plant parts.",
            "Install sticky yellow traps to monitor flying pest populations.",
        ],
        "prevention": [
            "Use fine-mesh nets over seedbeds to exclude early-stage pests.",
            "Introduce beneficial insects (e.g., ladybugs, Trichogramma wasps).",
            "Avoid excessive nitrogen fertilizer that produces soft, pest-attractive tissue.",
            "Practice field sanitation — remove crop debris promptly after harvest.",
        ],
        "severity": "High",
        "color": "#8e44ad",
    },
    "Eggplant Leaf Spot Disease": {
        "cause": "Cercospora melongenae / Alternaria spp. (fungal)",
        "symptom": "Circular brown or grey lesions with defined margins on leaf surfaces",
        "actions": [
            "Remove and dispose of all severely spotted leaves.",
            "Apply chlorothalonil or mancozeb-based fungicide.",
            "Irrigate at the base of the plant — avoid wetting foliage.",
            "Reduce canopy humidity by pruning excess leaves.",
        ],
        "prevention": [
            "Avoid dense planting; allow adequate airflow between plants.",
            "Apply organic mulch to prevent spore-laden soil splash onto leaves.",
            "Scout weekly and act at the first sign of lesions.",
            "Rotate crops each season to break fungal cycles.",
        ],
        "severity": "Medium",
        "color": "#e67e22",
    },
    "Eggplant Mosaic Virus Disease": {
        "cause": "Tobacco Mosaic Virus (TMV) or Cucumber Mosaic Virus (CMV); vectored by aphids",
        "symptom": "Mosaic-patterned yellowing, leaf puckering, stunted growth, and distorted fruit",
        "actions": [
            "Remove and destroy infected plants immediately — there is no cure.",
            "Control aphid vectors with insecticidal soap or neem oil.",
            "Disinfect all tools with 10% bleach solution between plants.",
            "Alert neighboring farms if spread is detected.",
        ],
        "prevention": [
            "Source certified virus-free planting material.",
            "Control aphid populations proactively before virus spread occurs.",
            "Plant resistant or tolerant varieties where available.",
            "Install reflective mulch to deter aphid landing.",
        ],
        "severity": "High",
        "color": "#e74c3c",
    },
    "Eggplant Small Leaf Disease": {
        "cause": "Phytoplasma infection; transmitted by leafhoppers",
        "symptom": "Abnormally small, yellowing leaves; shortened internodes; broom-like branching",
        "actions": [
            "Remove and destroy all symptomatic plants to prevent spread.",
            "Control leafhopper vectors using systemic insecticides (e.g., imidacloprid).",
            "Report severe outbreaks to your local agricultural extension office.",
            "Do not use plant material from infected fields for propagation.",
        ],
        "prevention": [
            "Use certified, pathogen-tested seedlings from reputable nurseries.",
            "Install yellow sticky traps to monitor and reduce leafhopper populations.",
            "Plant trap crops (e.g., sunflower) at field borders to divert leafhoppers.",
            "Avoid planting near known phytoplasma-infected fields.",
        ],
        "severity": "High",
        "color": "#c0392b",
    },
    "Eggplant Wilt Disease": {
        "cause": "Ralstonia solanacearum (bacterial) or Fusarium oxysporum (fungal)",
        "symptom": "Sudden, progressive wilting of shoots and leaves despite adequate moisture",
        "actions": [
            "Remove and destroy wilted plants immediately; bag before carrying out.",
            "Drench surrounding soil with copper-based bactericide or biocontrol agents.",
            "Switch to drip irrigation — avoid overhead watering.",
            "Do not replant solanaceous crops in the same soil for at least 2 seasons.",
        ],
        "prevention": [
            "Use certified disease-free seedlings and resistant varieties.",
            "Practice crop rotation with non-solanaceous crops (e.g., corn, legumes).",
            "Disinfect tools between plants using 70% isopropyl alcohol.",
            "Improve soil drainage and avoid waterlogging.",
        ],
        "severity": "High",
        "color": "#d35400",
    },
    # ── Fruit classes ───────────────────────────────────────────────────────────
    "Eggplant Healthy Fruit": {
        "cause": "None — fruit is healthy",
        "symptom": "No disease detected; fruit surface appears normal with uniform color",
        "actions": [
            "Harvest at the correct maturity stage to avoid over-ripening.",
            "Handle fruits gently during harvest to prevent bruising.",
            "Store in cool, dry conditions away from direct sunlight.",
        ],
        "prevention": [
            "Continue regular field scouting every 3–5 days.",
            "Maintain balanced fertilization to support healthy fruit development.",
            "Ensure consistent soil moisture to prevent physiological disorders.",
            "Control insect pests early before they reach the fruiting stage.",
        ],
        "severity": "None",
        "color": "#27ae60",
    },
    "Eggplant Fruit Creaking": {
        "cause": "Physiological stress — irregular watering, calcium deficiency, or rapid temperature fluctuation",
        "symptom": "Surface cracking or splitting on the fruit skin, often exposing inner flesh",
        "actions": [
            "Harvest affected fruits immediately to prevent secondary infections.",
            "Apply calcium foliar spray (calcium nitrate) to remaining fruits.",
            "Regulate irrigation to maintain consistent soil moisture levels.",
            "Avoid large gaps between watering cycles, especially during fruit swell.",
        ],
        "prevention": [
            "Use drip irrigation for uniform and steady moisture delivery.",
            "Apply mulch to buffer soil temperature and moisture fluctuations.",
            "Supplement with calcium and boron during fruit development stages.",
            "Avoid over-irrigation followed by drought stress.",
        ],
        "severity": "Medium",
        "color": "#f39c12",
    },
    "Eggplant Phomopsis Blight": {
        "cause": "Phomopsis vexans (fungal pathogen)",
        "symptom": "Dark brown to black sunken lesions on fruit surface; may show concentric rings with pycnidia",
        "actions": [
            "Remove and destroy all infected fruits immediately.",
            "Apply mancozeb or copper-based fungicide to remaining fruits and foliage.",
            "Avoid overhead irrigation to reduce humidity around fruit.",
            "Improve canopy airflow through pruning of excess shoots.",
        ],
        "prevention": [
            "Use certified disease-free seeds or treated planting material.",
            "Practice strict crop rotation — avoid eggplant in the same field for 2–3 seasons.",
            "Apply preventive fungicide sprays during humid weather.",
            "Sanitize all field equipment to prevent spore transfer.",
        ],
        "severity": "High",
        "color": "#8e44ad",
    },
    "Eggplant Shoot and Fruit Borer": {
        "cause": "Leucinodes orbonalis (insect pest — lepidopteran larva)",
        "symptom": "Small entry holes on fruit surface with frass; internal tunneling and rotting of flesh",
        "actions": [
            "Remove and destroy all bored fruits — do not leave them on the ground.",
            "Apply spinosad or emamectin benzoate targeting young larvae.",
            "Install pheromone traps to monitor adult moth populations.",
            "Prune bored shoots immediately and dispose of off-field.",
        ],
        "prevention": [
            "Use fine-mesh net bags over individual fruits during development.",
            "Plant at the start of the dry season to reduce borer pressure.",
            "Release egg parasitoids (Trichogramma spp.) as biocontrol agents.",
            "Maintain field sanitation — remove crop residues promptly after harvest.",
        ],
        "severity": "High",
        "color": "#c0392b",
    },
    "Eggplant Wet Rot": {
        "cause": "Pythium spp. or Erwinia carotovora (water mold / bacterial soft rot)",
        "symptom": "Water-soaked, mushy lesions on fruit; rapid tissue collapse with foul odor",
        "actions": [
            "Remove and destroy all rotting fruits immediately — bag and dispose off-site.",
            "Apply copper hydroxide or fosetyl-aluminum to affected plants.",
            "Reduce irrigation frequency and improve field drainage.",
            "Avoid wounding fruits during cultural practices.",
        ],
        "prevention": [
            "Ensure well-drained soil and raised beds to prevent waterlogging.",
            "Avoid overhead irrigation — use drip systems instead.",
            "Space plants adequately to reduce humidity within the canopy.",
            "Apply biocontrol agents (Trichoderma spp.) to the soil before planting.",
        ],
        "severity": "High",
        "color": "#2c3e50",
    },
}

CLASS_NAMES = list(DISEASE_INFO.keys())

# ── MODEL LOADER ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """
    Load the EfficientNet-B0 model from 'yieldy_model.pth'.

    The model must be saved with:
        torch.save(model.state_dict(), 'yieldy_model.pth')
    and must have been trained with num_classes matching CLASS_NAMES above.

    Falls back to demo/mock mode if the weight file is not found.
    """
    model_path = "yieldy_model.pth"
    if not os.path.exists(model_path):
        st.warning(
            f"Weight file `{model_path}` not found in the working directory. "
            "Running in **Demo Mode** with mock inference.",
            icon="⚠️",
        )
        return None

    try:
        import torch
        import timm

        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=len(CLASS_NAMES),  # matches train.py
        )
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    except Exception as exc:
        st.error(
            f"Model file found but could not be loaded: {exc}\n\n"
            "Check that the checkpoint was saved with the correct num_classes and the "
            "correct EfficientNet-B0 architecture. Falling back to Demo Mode.",
            icon="🚨",
        )
        return None


# ── INFERENCE ──────────────────────────────────────────────────────────────────

def predict(image: Image.Image, model):
    """
    Run inference on a PIL image.

    Returns:
        predicted_class (str): top-1 class name
        confidence (float):    top-1 softmax probability
        all_scores (dict):     {class_name: probability} for all 6 classes
    """
    if model is None:
        # ── MOCK MODE (deterministic from image content) ──────────────────────
        img_array = np.array(image.resize((224, 224))).astype(np.float32)
        seed = int(img_array.mean() * 100) % 2_147_483_647
        rng = np.random.default_rng(seed)
        raw_scores = rng.dirichlet(np.ones(len(CLASS_NAMES)) * 0.5)
        top_idx = int(np.argmax(raw_scores))
        raw_scores[top_idx] *= 3
        raw_scores /= raw_scores.sum()
        # ─────────────────────────────────────────────────────────────────────
    else:
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        tensor = transform(image.convert("RGB")).unsqueeze(0)  # (1, 3, 224, 224)
        with torch.no_grad():
            logits = model(tensor)                             # (1, 6)
            raw_scores = (
                torch.softmax(logits, dim=1).squeeze().numpy()
            )                                                  # (6,)

    predicted_idx   = int(np.argmax(raw_scores))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(raw_scores[predicted_idx])
    all_scores      = {
        CLASS_NAMES[i]: float(raw_scores[i]) for i in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, all_scores


# ── UI HELPERS ─────────────────────────────────────────────────────────────────

def severity_badge(severity: str, color: str) -> str:
    return f"""<span style="
        background:{color}22;
        color:{color};
        border:1px solid {color};
        padding:2px 10px;
        border-radius:20px;
        font-size:0.8rem;
        font-weight:600;
    ">⚠ Severity: {severity}</span>"""


def confidence_color(conf: float) -> str:
    if conf >= 0.80:
        return "#27ae60"
    elif conf >= 0.55:
        return "#e67e22"
    return "#e74c3c"


# ── MAIN APP ───────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <h1 style="font-size:2.4rem; margin-bottom:0;">🌱 Yieldy</h1>
        <p style="color:gray; font-size:1rem; margin-top:4px;">
            Early crop disease detection for Filipino eggplant farmers
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Load model — shows warning/error banners internally if needed
    model = load_model()
    if model is None:
        st.info(
            "**Demo Mode** — Upload any eggplant photo to see a simulated diagnosis. "
            "Place `yieldy_model.pth` in the same directory as `app.py` to enable real inference.",
            icon="ℹ️",
        )
    else:
        st.success("✅ EfficientNet-B0 model loaded — running live inference.", icon="✅")

    st.markdown("Upload an image of your eggplant (or an affected plant part) to begin. 🍆")
    st.caption(
        "Supports JPG, JPEG, PNG · "
        "Best results with clear, close-up photos of the affected area."
    )

    uploaded_file = st.file_uploader(
        label="Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown("Running Analysis…")
            progress = st.progress(0)
            for i in range(1, 101):
                time.sleep(0.008)
                progress.progress(i)

            predicted_class, confidence, all_scores = predict(image, model)

            # ── Rejection gate ─────────────────────────────────────────────
            if confidence < CONFIDENCE_THRESHOLD:
                progress.empty()
                st.warning(
                    "⚠️ **Unrecognized Image**\n\n"
                    "This image doesn't appear to be an eggplant leaf or fruit. "
                    "Please upload a clear, close-up photo of an eggplant plant part "
                    "(leaf, fruit, or shoot) for an accurate diagnosis.",
                    icon="🚫",
                )
                st.caption(
                    f"Model confidence: {confidence * 100:.1f}% "
                    f"(minimum required: {CONFIDENCE_THRESHOLD * 100:.0f}%)"
                )
                return
            # ──────────────────────────────────────────────────────────────
            info       = DISEASE_INFO[predicted_class]
            conf_color = confidence_color(confidence)

            progress.empty()

            # ── Result card ────────────────────────────────────────────────
            st.markdown(f"""
            <div style="
                border: 1px solid {info['color']};
                border-radius: 12px;
                padding: 16px 20px;
                background: {info['color']}11;
            ">
                <div style="font-size:1.4rem; font-weight:700; color:{info['color']};">
                    {predicted_class}
                </div>
                <div style="font-size:0.85rem; color:gray; margin: 4px 0 8px 0;">
                    <i>{info['cause']}</i>
                </div>
                {severity_badge(info['severity'], info['color'])}
                <div style="margin-top:12px; font-size:0.9rem;">
                    <b>Primary Symptom:</b> {info['symptom']}
                </div>
                <div style="margin-top:8px;">
                    <span style="font-size:1.5rem; font-weight:700; color:{conf_color};">
                        {confidence * 100:.1f}%
                    </span>
                    <span style="font-size:1rem; color:gray;"> Confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Action steps + Prevention ──────────────────────────────────────
        col3, col4 = st.columns(2, gap="large")

        with col3:
            st.markdown("🚨 Immediate Action Steps")
            for i, action in enumerate(info["actions"], 1):
                st.markdown(f"**{i}.** {action}")

        with col4:
            st.markdown("🛡️ Prevention Tips")
            for tip in info["prevention"]:
                st.markdown(f"- {tip}")

        st.divider()

        # ── Confidence breakdown ───────────────────────────────────────────
        st.markdown("#### 📊 Confidence Breakdown (All Classes)")
        scores_sorted = dict(
            sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        )

        for cls, score in scores_sorted.items():
            bar_color    = DISEASE_INFO[cls]["color"]
            bar_pct      = score * 100
            is_predicted = cls == predicted_class
            label_style  = "font-weight:700;" if is_predicted else "color:gray;"
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                    <span style="font-size:0.85rem; {label_style}">{cls}</span>
                    <span style="font-size:0.85rem; {label_style}">{bar_pct:.1f}%</span>
                </div>
                <div style="background:#eee; border-radius:8px; height:10px; overflow:hidden;">
                    <div style="
                        width:{bar_pct}%;
                        background:{bar_color};
                        height:100%;
                        border-radius:8px;
                        transition:width 0.5s;
                    "></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.caption(
            "⚠️ Yieldy is currently under development. "
            "Always confirm diagnosis through a qualified agricultural extension officer."
        )

    else:
        # ── Empty state ────────────────────────────────────────────────────
        st.markdown("""
        <div style="
            border: 2px dashed #ccc;
            border-radius: 12px;
            padding: 48px 24px;
            text-align: center;
            color: #aaa;
        ">
            <div style="font-size: 3rem;">📷</div>
            <div style="font-size:1rem; margin-top: 8px;">
                Upload a photo of your eggplant crop to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🌿 Yieldy")
        st.markdown("A crop disease detection tool for Philippine eggplant farmers.")
        st.divider()
        st.markdown("**Detectable Classes:**")
        for cls in CLASS_NAMES:
            color = DISEASE_INFO[cls]["color"]
            st.markdown(f"<span style='color:{color}'>●</span> {cls}", unsafe_allow_html=True)
        st.divider()
        st.markdown("**Model:** EfficientNet-B0 (Transfer Learning)")
        st.markdown(f"**Classes:** {len(CLASS_NAMES)}")
        st.markdown("**Target Crop:** Eggplant *(Solanum melongena)*")
        st.divider()
        st.caption("Yieldy v0.2")


if __name__ == "__main__":
    main()