import streamlit as st
from PIL import Image
import numpy as np
import time
import os

# PAGE CONFIG

st.set_page_config(
    page_title="Yieldy",
    page_icon="🌱",
    layout="centered",
)

# DISEASE DATABASE SECTION

DISEASE_INFO = {
    "Bacterial Wilt": {
        "cause": "Ralstonia solanacearum",
        "symptom": "Sudden wilting of shoots and leaves",
        "actions": [
            "Remove and destroy infected plants immediately.",
            "Avoid overhead irrigation — use drip irrigation instead.",
            "Apply copper-based bactericides to surrounding plants.",
            "Do not replant eggplant in the same soil for at least 2 seasons.",
        ],
        "prevention": [
            "Use certified disease-free seedlings.",
            "Practice crop rotation with non-solanaceous crops.",
            "Disinfect tools between plants with 70% alcohol.",
            "Improve soil drainage to reduce moisture buildup.",
        ],
        "severity": "High",
        "color": "#e74c3c",
    },
    "Phomopsis Blight": {
        "cause": "Phomopsis vexans",
        "symptom": "Fruit rot and stem lesions with dark margins",
        "actions": [
            "Remove and bag all infected fruits and stems.",
            "Apply mancozeb or copper oxychloride based fungicide.",
            "Improve canopy ventilation by pruning excess foliage.",
            "Avoid wetting foliage during irrigation.",
        ],
        "prevention": [
            "Use resistant varieties when available.",
            "Apply preventive fungicide sprays during wet seasons.",
            "Space plants adequately to improve air circulation.",
            "Avoid wounding plants during field operations.",
        ],
        "severity": "Medium",
        "color": "#e67e22",
    },
    "Cercospora Leaf Spot": {
        "cause": "Cercospora melongenae",
        "symptom": "Circular brown lesions with grey centers on leaves",
        "actions": [
            "Remove severely affected leaves and dispose of them.",
            "Apply chlorothalonil or mancozeb fungicide.",
            "Reduce humidity around plants.",
            "Irrigate at the base of the plant, not overhead.",
        ],
        "prevention": [
            "Avoid dense planting to allow airflow.",
            "Apply mulch to prevent soil splash onto leaves.",
            "Scout weekly and act at first sign of lesions.",
            "Rotate crops each season.",
        ],
        "severity": "Low",
        "color": "#f1c40f",
    },
    "Fruit and Shoot Borer": {
        "cause": "Leucinodes orbonalis",
        "symptom": "Bored holes in shoots and fruits with larval frass",
        "actions": [
            "Cut and destroy all wilted shoots immediately.",
            "Collect and bury infested fruits at least 30 cm deep.",
            "Apply spinosad or cypermethrin at dusk when moths are active.",
            "Install pheromone traps to monitor adult moth population.",
        ],
        "prevention": [
            "Use fine mesh nets over seedbeds.",
            "Practice clean cultivation — remove crop debris after harvest.",
            "Introduce natural enemies like Trichogramma wasps.",
            "Avoid excessive nitrogen fertilizer that attracts borers.",
        ],
        "severity": "High",
        "color": "#8e44ad",
    },
    "Healthy Plant": {
        "cause": "None",
        "symptom": "No disease detected",
        "actions": [
            "Continue regular monitoring every 3–5 days.",
            "Maintain balanced fertilization schedule.",
            "Ensure consistent soil moisture.",
        ],
        "prevention": [
            "Keep up with preventive scouting routines.",
            "Monitor weather forecasts for humidity changes.",
            "Maintain proper plant spacing.",
        ],
        "severity": "None",
        "color": "#27ae60",
    },
}

CLASS_NAMES = list(DISEASE_INFO.keys())

# MODEL LOADER
@st.cache_resource
def load_model():
    """
    Load the EfficientNet-B0 model.

    TO USE YOUR REAL MODEL:
    1. Replace the block below with:
           import torch
           import timm
           model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=5)
           model.load_state_dict(torch.load('yieldy_model.pth', map_location='cpu'))
           model.eval()
           return model
    2. Remove the `return None` line.
    """
    model_path = "yieldy_model.pth"
    if os.path.exists(model_path):
        try:
            import torch
            import timm
            model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            return model
        except Exception as e:
            st.warning(f"Model file found but could not be loaded: {e}. Running in demo mode.")
            return None
    return None  # Demo/mock mode


def predict(image: Image.Image, model):
    """
    Run inference on a PIL image.
    Returns: (class_name: str, confidence: float, all_scores: dict)
    """
    if model is None:
        # MOCK MODE 
        img_array = np.array(image.resize((224, 224))).astype(np.float32)
        seed = int(img_array.mean() * 100) % 2147483647
        rng = np.random.default_rng(seed)
        raw_scores = rng.dirichlet(np.ones(5) * 0.5)
        top_idx = int(np.argmax(raw_scores))
        raw_scores[top_idx] = raw_scores[top_idx] * 3
        raw_scores = raw_scores / raw_scores.sum()
        # End of Mock
    else:
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            raw_scores = torch.softmax(logits, dim=1).squeeze().numpy()

    predicted_idx = int(np.argmax(raw_scores))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(raw_scores[predicted_idx])
    all_scores = {CLASS_NAMES[i]: float(raw_scores[i]) for i in range(len(CLASS_NAMES))}

    return predicted_class, confidence, all_scores


# UI HELPERS
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


# MAIN APP
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

    # Model status banner
    model = load_model()
    if model is None:
        st.info("**Demo Mode** — Project in Progress. "
                "Project is still under development.", icon="ℹ️")
    else:
        st.success("✅ Model loaded successfully.", icon="✅")

    st.markdown("Upload an Image of your Eggplant (or a part of it to be assessed.) 🍆")
    st.caption("Supports JPG, JPEG, PNG. Best results with clear, close-up photos of the affected plant part.")

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
            st.markdown("Running Analysis...")
            progress = st.progress(0)
            for i in range(1, 101):
                time.sleep(0.008)
                progress.progress(i)

            predicted_class, confidence, all_scores = predict(image, model)
            info = DISEASE_INFO[predicted_class]
            conf_color = confidence_color(confidence)

            progress.empty()

            # Result card
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
                    <span style="
                        font-size:1.5rem; font-weight:700; color:{conf_color};
                    ">{confidence*100:.1f}%</span>
                    <span style="font-size:0.8rem; color:gray;"> confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Action steps + prevention
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

        # Confidence breakdown chart
        st.markdown("#### 📊 Confidence Breakdown (All Classes)")
        scores_sorted = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))

        for cls, score in scores_sorted.items():
            bar_color = DISEASE_INFO[cls]["color"]
            bar_pct = score * 100
            is_predicted = cls == predicted_class
            label_style = "font-weight:700;" if is_predicted else "color:gray;"
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
        st.caption("⚠️ Yieldy is currently still under development. Always re-confirm diagnosis through trusted sources.")

    else:
        # Empty state
        st.markdown("""
        <div style="
            border: 2px dashed #ccc;
            border-radius: 12px;
            padding: 48px 24px;
            text-align: center;
            color: #aaa;
        ">
            <div style="font-size: 3rem;">📷</div>
            <div style="font-size:1rem; margin-top: 8px;">Upload a photo of your eggplant crop to begin.</div>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🌿 Yieldy")
        st.markdown("A crop disease detection tool for Philippine eggplant farmers.")
        st.divider()
        st.markdown("**Detectable Diseases:**")
        for cls in CLASS_NAMES:
            color = DISEASE_INFO[cls]["color"]
            st.markdown(f"<span style='color:{color}'>●</span> {cls}", unsafe_allow_html=True)
        st.divider()
        st.markdown("**Model:** EfficientNet-B0 (Transfer Learning)")
        st.markdown("**Target Crop:** Eggplant *(Solanum melongena)*")
        st.divider()
        st.caption("Yieldy v0.1 ")


if __name__ == "__main__":
    main()
