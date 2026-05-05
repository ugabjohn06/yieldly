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

# DISEASE DATABASE — matched to dataset folder names
DISEASE_INFO = {
    "Eggplant Healthy Leaf": {
        "cause": "None",
        "symptom": "No disease detected — plant appears healthy",
        "actions": [
            "Continue regular monitoring every 3–5 days.",
            "Maintain balanced fertilization schedule.",
            "Ensure consistent soil moisture.",
            "Keep weeds cleared around the base of plants.",
        ],
        "prevention": [
            "Keep up with preventive scouting routines.",
            "Monitor weather forecasts for humidity changes.",
            "Maintain proper plant spacing for air circulation.",
            "Rotate crops each season to prevent soil-borne diseases.",
        ],
        "severity": "None",
        "color": "#27ae60",
    },
    "Eggplant Wilt Disease": {
        "cause": "Ralstonia solanacearum (Bacterial Wilt) or Fusarium oxysporum (Fusarium Wilt)",
        "symptom": "Sudden wilting of shoots, yellowing, and collapse of the entire plant",
        "actions": [
            "Remove and destroy infected plants immediately — do not compost.",
            "Avoid overhead irrigation; switch to drip irrigation.",
            "Apply copper-based bactericides to surrounding healthy plants.",
            "Do not replant eggplant in the same soil for at least 2 seasons.",
            "Disinfect all tools with 70% alcohol after contact with infected plants.",
        ],
        "prevention": [
            "Use certified disease-free seedlings.",
            "Practice crop rotation with non-solanaceous crops (e.g., corn, beans).",
            "Improve soil drainage to reduce moisture buildup.",
            "Choose wilt-resistant varieties when available.",
        ],
        "severity": "High",
        "color": "#e74c3c",
    },
    "Eggplant Leaf Spot Disease": {
        "cause": "Cercospora melongenae or other fungal/bacterial pathogens",
        "symptom": "Circular brown or grey lesions with defined margins on leaves; premature leaf drop",
        "actions": [
            "Remove and dispose of heavily affected leaves immediately.",
            "Apply chlorothalonil or mancozeb-based fungicide.",
            "Reduce canopy humidity by pruning excess foliage.",
            "Irrigate at the base of the plant — avoid wetting the leaves.",
        ],
        "prevention": [
            "Avoid dense planting to allow airflow between plants.",
            "Apply mulch to prevent soil splash onto lower leaves.",
            "Scout weekly and act at first sign of lesions.",
            "Rotate crops each season.",
        ],
        "severity": "Medium",
        "color": "#e67e22",
    },
    "Eggplant Insect Pest Disease": {
        "cause": "Leucinodes orbonalis (Fruit & Shoot Borer), flea beetles, aphids, or other insect pests",
        "symptom": "Bore holes in shoots/fruits, wilted shoot tips, frass (insect droppings), distorted leaves",
        "actions": [
            "Cut and destroy all wilted shoots immediately.",
            "Collect and bury infested fruits at least 30 cm deep.",
            "Apply spinosad or cypermethrin at dusk when moths are most active.",
            "Install pheromone traps to monitor and reduce adult moth population.",
            "Introduce natural predators like Trichogramma wasps if available.",
        ],
        "prevention": [
            "Use fine mesh nets over seedbeds to block adult insects.",
            "Practice clean cultivation — remove all crop debris after harvest.",
            "Avoid excessive nitrogen fertilizer, which attracts borers.",
            "Scout plants twice weekly during fruiting stage.",
        ],
        "severity": "High",
        "color": "#8e44ad",
    },
    "Eggplant Mosaic Virus Disease": {
        "cause": "Tobacco Mosaic Virus (TMV) or Cucumber Mosaic Virus (CMV), spread by aphids",
        "symptom": "Mosaic-patterned yellowing on leaves, leaf curling, stunted growth, distorted fruits",
        "actions": [
            "Remove and destroy infected plants — there is no cure for viral infections.",
            "Control aphid populations immediately using insecticidal soap or neem oil.",
            "Disinfect hands and tools after handling infected plants.",
            "Isolate affected plants to prevent spread to healthy ones.",
        ],
        "prevention": [
            "Use virus-free certified seedlings.",
            "Control aphid vectors with reflective mulches or yellow sticky traps.",
            "Avoid smoking near plants (TMV can be carried on hands from tobacco).",
            "Plant resistant varieties whenever possible.",
        ],
        "severity": "High",
        "color": "#c0392b",
    },
    "Eggplant Small Leaf Disease": {
        "cause": "Phytoplasma infection, transmitted by leafhoppers",
        "symptom": "Abnormally small, yellowed leaves; shortened internodes; stunted bushy appearance; little to no fruiting",
        "actions": [
            "Remove and destroy all infected plants — phytoplasma has no cure.",
            "Apply insecticide to control leafhopper vectors in surrounding plants.",
            "Avoid replanting in the same area for at least one full season.",
            "Report severe outbreaks to your local DA extension office.",
        ],
        "prevention": [
            "Control leafhopper populations using yellow sticky traps.",
            "Use insect-proof screens on seedbeds.",
            "Remove weeds around the farm that may harbor leafhoppers.",
            "Inspect new seedlings carefully before transplanting.",
        ],
        "severity": "High",
        "color": "#d35400",
    },
    "Eggplant White Mold Disease": {
        "cause": "Sclerotinia sclerotiorum or Phomopsis vexans (fungal)",
        "symptom": "White cottony mold growth on stems, leaves, or fruit; soft rot; water-soaked lesions",
        "actions": [
            "Remove and bag all infected plant parts — do not compost.",
            "Apply mancozeb or copper oxychloride fungicide to affected areas.",
            "Improve canopy ventilation by pruning excess foliage.",
            "Reduce irrigation frequency temporarily to lower humidity.",
        ],
        "prevention": [
            "Space plants adequately to improve air circulation.",
            "Avoid wetting foliage during irrigation.",
            "Apply preventive fungicide sprays during wet and humid seasons.",
            "Avoid wounding plants during field operations.",
        ],
        "severity": "Medium",
        "color": "#7f8c8d",
    },
}

# Class names must match dataset folder names EXACTLY
CLASS_NAMES = list(DISEASE_INFO.keys())


# MODEL LOADER
@st.cache_resource
def load_model():
    """
    Load the EfficientNet-B0 model (7 classes).

    The model was trained with a custom Sequential classifier head:
        Dropout → Linear(1280, 512) → ReLU → Dropout → Linear(512, 7)
    which is saved under the key 'classifier' in the state dict.

    After training your model with the Colab notebook, place
    'yieldly2_model.pth' in the same directory as this file.
    """
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yieldly2_model.pth")
    if os.path.exists(model_path):
        try:
            import torch
            import torch.nn as nn
            import timm

            # Build EfficientNet-B0 backbone (no top classifier)
            model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
            num_features = model.num_features  # 1280 for EfficientNet-B0

            # Recreate the exact same classifier head used during training:
            # Sequential(Dropout, Linear(1280,512), ReLU, Dropout, Linear(512,7))
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 7),
            )

            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            st.warning(f"Model file found but could not be loaded: {e}. Running in demo mode.")
            return None
    return None  # Demo/mock mode


def predict(image: Image.Image, model):
    """
    Run inference on a PIL image using Test-Time Augmentation (TTA).

    TTA runs the model on several augmented versions of the same image
    and averages the predictions. This makes the model more robust to
    real-world photos (different angles, lighting, phone cameras).

    Returns: (class_name: str, confidence: float, all_scores: dict)
    """
    if model is None:
        # MOCK MODE — deterministic random based on image content
        img_array = np.array(image.resize((224, 224))).astype(np.float32)
        seed = int(img_array.mean() * 100) % 2147483647
        rng = np.random.default_rng(seed)
        raw_scores = rng.dirichlet(np.ones(7) * 0.5)
        top_idx = int(np.argmax(raw_scores))
        raw_scores[top_idx] = raw_scores[top_idx] * 3
        raw_scores = raw_scores / raw_scores.sum()
    else:
        import torch
        from torchvision import transforms

        img = image.convert("RGB")

        # TTA: define multiple augmentation views of the same image
        tta_transforms = [
            # Original
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            # Slightly brighter
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            # Center crop (zoomed in)
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            # Vertical flip
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        ]

        all_probs = []
        with torch.no_grad():
            for tf in tta_transforms:
                tensor = tf(img).unsqueeze(0)
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1).squeeze().numpy()
                all_probs.append(probs)

        # Average predictions across all TTA views
        raw_scores = np.mean(all_probs, axis=0)

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


# Display name — strips "Eggplant " prefix for cleaner UI
def display_name(cls: str) -> str:
    return cls.replace("Eggplant ", "")


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
        st.info("**Demo Mode** — No trained model found. "
                "Train your model using the provided Colab notebook and place `yieldy_model.pth` here.", icon="ℹ️")
    else:
        st.success("✅ Model loaded successfully.", icon="✅")

    st.markdown("Upload an image of your eggplant (or a part of it) to be assessed. 🍆")
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
                    {display_name(predicted_class)}
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
                    <span style="font-size:1rem; color:gray;"> Confidence</span>
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
                    <span style="font-size:0.85rem; {label_style}">{display_name(cls)}</span>
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
        st.caption("⚠️ Yieldy is currently still under development. Always confirm diagnosis through a trusted agricultural expert.")

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
        st.markdown("**Detectable Conditions:**")
        for cls in CLASS_NAMES:
            color = DISEASE_INFO[cls]["color"]
            st.markdown(f"<span style='color:{color}'>●</span> {display_name(cls)}", unsafe_allow_html=True)
        st.divider()
        st.markdown("**Model:** EfficientNet-B0 (Transfer Learning)")
        st.markdown("**Target Crop:** Eggplant *(Solanum melongena)*")
        st.markdown("**Classes:** 7")
        st.divider()
        st.caption("Yieldy v0.2")


if __name__ == "__main__":
    main()
