
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.weather_advisory import get_weather
from src.disease_detection import predict_disease
from src.market_analysis import get_sample_market_data, analyze_market_data


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriSmart AI | Smart Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "crop_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "crop_metrics.json"

DATA_DIR = BASE_DIR / "data"
PLANT_VILLAGE_DIR = DATA_DIR / "plant_village"

TASK_FILE = DATA_DIR / "project_tasks.json"

HERO_IMAGE = (
    "https://images.unsplash.com/photo-1500382017468-9049fed747ef"
    "?auto=format&fit=crop&w=1800&q=85"
)

CROP_IMAGE = (
    "https://images.unsplash.com/photo-1499529112087-3cb3b73cec95"
    "?auto=format&fit=crop&w=900&q=80"
)

WEATHER_IMAGE = (
    "https://images.unsplash.com/photo-1534088568595-a066f410bcda"
    "?auto=format&fit=crop&w=900&q=80"
)

DISEASE_IMAGE = (
    "https://images.unsplash.com/photo-1518977676601-b53f82aba655"
    "?auto=format&fit=crop&w=900&q=80"
)

MARKET_IMAGE = (
    "https://images.unsplash.com/photo-1542838132-92c53300491e"
    "?auto=format&fit=crop&w=900&q=80"
)


# ============================================================
# 2. DISEASE INFORMATION DATABASE
# ============================================================

DISEASE_INFO = {

    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "crop": "Tomato",
        "symptoms": [
            "Brown circular spots on older leaves.",
            "Spots may develop ring-like patterns.",
            "Leaves may turn yellow and dry."
        ],
        "management": [
            "Remove badly infected leaves and plant debris.",
            "Avoid watering directly over leaves.",
            "Maintain proper spacing and air circulation.",
            "Use locally recommended disease-management products "
            "only after expert advice."
        ]
    },

    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "crop": "Tomato",
        "symptoms": [
            "Dark, water-soaked patches on leaves.",
            "Brown or black lesions may spread quickly.",
            "White growth may appear on leaf undersides in humid conditions."
        ],
        "management": [
            "Remove and safely dispose of infected plant material.",
            "Avoid prolonged leaf wetness.",
            "Improve field ventilation and drainage.",
            "Seek local agricultural advice quickly because this disease "
            "can spread rapidly."
        ]
    },

    "Tomato___Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "crop": "Tomato",
        "symptoms": [
            "Small dark spots on leaves.",
            "Spots may have yellow surrounding areas.",
            "Fruit may develop raised or scabby spots."
        ],
        "management": [
            "Avoid handling plants when they are wet.",
            "Remove severely affected plant material.",
            "Use clean seeds and healthy seedlings.",
            "Ask an agricultural expert about locally approved management."
        ]
    },

    "Tomato___Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "crop": "Tomato",
        "symptoms": [
            "Pale yellow patches on upper leaf surfaces.",
            "Olive-green or grey growth underneath leaves.",
            "Leaves may curl and dry."
        ],
        "management": [
            "Improve ventilation between plants.",
            "Avoid excessive moisture on leaves.",
            "Remove infected leaves where practical.",
            "Monitor crops regularly in humid conditions."
        ]
    },

    "Tomato___Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "crop": "Tomato",
        "symptoms": [
            "Numerous small circular spots.",
            "Spots may have grey centres and dark edges.",
            "Lower leaves often show symptoms first."
        ],
        "management": [
            "Remove infected lower leaves.",
            "Clear crop debris after harvest.",
            "Avoid overhead irrigation.",
            "Maintain adequate plant spacing."
        ]
    },

    "Tomato___Target_Spot": {
        "name": "Tomato Target Spot",
        "crop": "Tomato",
        "symptoms": [
            "Brown circular leaf spots.",
            "Spots may show concentric rings.",
            "Severely affected leaves can fall."
        ],
        "management": [
            "Remove infected plant debris.",
            "Improve airflow within the crop.",
            "Avoid prolonged leaf wetness.",
            "Get expert advice if symptoms continue spreading."
        ]
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "crop": "Tomato",
        "symptoms": [
            "Leaves curl upward or become distorted.",
            "Leaves may become yellow.",
            "Plants may show stunted growth."
        ],
        "management": [
            "Remove severely affected plants after local expert confirmation.",
            "Monitor whiteflies, which can spread the virus.",
            "Use healthy seedlings.",
            "Follow integrated pest management recommendations."
        ]
    },

    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "crop": "Tomato",
        "symptoms": [
            "Light and dark green mosaic patterns on leaves.",
            "Leaves may become distorted.",
            "Plant growth may be reduced."
        ],
        "management": [
            "Remove infected plants where advised.",
            "Clean tools and hands after handling plants.",
            "Use disease-free seeds and seedlings.",
            "Avoid moving sap between healthy and infected plants."
        ]
    },

    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "crop": "Potato",
        "symptoms": [
            "Brown spots on older leaves.",
            "Spots may have concentric ring patterns.",
            "Yellowing can develop around affected areas."
        ],
        "management": [
            "Remove infected leaves and crop debris.",
            "Maintain balanced crop nutrition.",
            "Avoid unnecessary leaf wetness.",
            "Follow locally recommended disease management."
        ]
    },

    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "crop": "Potato",
        "symptoms": [
            "Dark, water-soaked leaf lesions.",
            "White growth may appear on leaf undersides.",
            "Infection may spread rapidly in cool, wet weather."
        ],
        "management": [
            "Remove infected plant material as advised.",
            "Avoid moving infected tubers or plants between fields.",
            "Monitor weather and crop conditions frequently.",
            "Contact an agricultural expert promptly."
        ]
    },

    "Pepper,_bell___Bacterial_spot": {
        "name": "Bell Pepper Bacterial Spot",
        "crop": "Bell Pepper",
        "symptoms": [
            "Small dark spots on leaves.",
            "Yellowing around leaf lesions.",
            "Raised spots may develop on fruits."
        ],
        "management": [
            "Use healthy seedlings and disease-free seeds.",
            "Avoid working in wet crops.",
            "Remove infected plant debris.",
            "Maintain good spacing and field hygiene."
        ]
    }
}


# ============================================================
# 3. PROJECT TASK TRACKER
# ============================================================

INITIAL_TASKS = [

    {
        "task": "Crop recommendation model",
        "module": "Crop Recommendation",
        "status": "Done"
    },

    {
        "task": "Weather advisory module",
        "module": "Weather Advisory",
        "status": "Done"
    },

    {
        "task": "Plant disease detection model",
        "module": "Disease Detection",
        "status": "Done"
    },

    {
        "task": "Market analysis module",
        "module": "Market Analysis",
        "status": "In Progress"
    },

    {
        "task": "Disease examples and guidance gallery",
        "module": "Disease Detection",
        "status": "In Progress"
    },

    {
        "task": "Live mandi data integration",
        "module": "Market Analysis",
        "status": "Pending"
    },

    {
        "task": "Project radar and task tracker",
        "module": "Project Dashboard",
        "status": "In Progress"
    },

    {
        "task": "Final system testing",
        "module": "Project Dashboard",
        "status": "In Progress"
    },

    {
        "task": "README and documentation",
        "module": "Project Dashboard",
        "status": "Done"
    }
]


def load_tasks():
    """Load saved tasks or create the initial task list."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not TASK_FILE.exists():
        save_tasks(INITIAL_TASKS)
        return INITIAL_TASKS.copy()

    try:
        with open(TASK_FILE, "r", encoding="utf-8") as file:
            tasks = json.load(file)

        if not isinstance(tasks, list):
            return INITIAL_TASKS.copy()

        return tasks

    except (json.JSONDecodeError, OSError):
        return INITIAL_TASKS.copy()


def save_tasks(tasks):
    """Save project tasks to a local JSON file."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(TASK_FILE, "w", encoding="utf-8") as file:
        json.dump(tasks, file, indent=4)


# ============================================================
# 4. DISEASE DATASET IMAGE FINDER
# ============================================================

@st.cache_data
def find_disease_images():
    """
    Find representative images from the downloaded PlantVillage dataset.
    Uses the original dataset images, not generated images.
    """

    found = {}

    if not PLANT_VILLAGE_DIR.exists():
        return found

    image_extensions = {".jpg", ".jpeg", ".png"}

    for folder in PLANT_VILLAGE_DIR.rglob("*"):

        if not folder.is_dir():
            continue

        class_name = folder.name

        if class_name not in DISEASE_INFO:
            continue

        images = [
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in image_extensions
        ]

        if images:
            found[class_name] = str(images[0])

    return found


# ============================================================
# 5. MODEL LOADING
# ============================================================

@st.cache_resource
def load_crop_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_crop_metrics():
    if not METRICS_PATH.exists():
        return {}

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 6. PROFESSIONAL FRONTEND CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --green-dark: #103b2b;
    --green-main: #18794e;
    --green-light: #e9f5ed;
    --text-dark: #183c2b;
    --text-muted: #63796b;
    --background: #f3f7f3;
    --white: #ffffff;
    --border: #dce9df;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text-dark);
}

.block-container {
    max-width: 1450px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--green-dark) !important;
    font-weight: 800 !important;
}

p, li {
    color: var(--text-dark);
}


/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0a2d20 0%,
        #103b2b 55%,
        #15543a 100%
    );
    border-right: 1px solid #245b40;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.3rem;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.17);
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 9px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 11px 12px;
    border-radius: 11px;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.045);
    transition: all 0.25s ease;
    cursor: pointer;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #245f41;
    border-color: #7dc99a;
    transform: translateX(4px);
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #23764b;
    border: 1px solid #91d9aa;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #e1f0e5 !important;
}


/* BUTTONS */

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(145deg, #258d5b, #12623d) !important;
    color: #ffffff !important;
    border: 1px solid #2b9866 !important;
    border-radius: 11px !important;
    font-weight: 800 !important;
    min-height: 45px;
    padding: 10px 20px !important;
    box-shadow: 0 5px 0 #0a492c, 0 8px 17px rgba(15,80,48,0.16) !important;
    transition: all 0.18s ease !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(145deg, #31a76c, #18794e) !important;
    color: white !important;
    transform: translateY(-3px);
    box-shadow: 0 8px 0 #0a492c, 0 13px 23px rgba(15,80,48,0.22) !important;
}

.stButton > button:active,
.stFormSubmitButton > button:active {
    transform: translateY(3px) scale(0.98) !important;
}


/* INPUTS */

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #ffffff !important;
    color: #183c2b !important;
    border: 1px solid #cbdccf !important;
    border-radius: 9px !important;
}

.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stFileUploader label {
    color: #315b40 !important;
    font-weight: 700 !important;
}

div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #183c2b !important;
    border-radius: 9px !important;
}


/* FORMS */

div[data-testid="stForm"] {
    background: #ffffff !important;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 5px 18px rgba(20,60,35,0.04);
}


/* HERO */

.hero {
    background:
        linear-gradient(90deg, rgba(5,31,19,0.94), rgba(5,31,19,0.30)),
        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85");
    background-size: cover;
    background-position: center;
    min-height: 275px;
    border-radius: 22px;
    padding: 40px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 12px 30px rgba(20,60,35,0.13);
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 39px;
    font-weight: 800;
    max-width: 700px;
    line-height: 1.2;
}

.hero p {
    color: #edf8ef !important;
    font-size: 15px;
    max-width: 650px;
}

.eyebrow {
    color: #a9edbd !important;
    font-weight: 800;
    letter-spacing: 2px;
    font-size: 12px;
}


/* HEADINGS */

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: #17452e;
    margin-top: 20px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #6c8375;
    font-size: 14px;
    margin-bottom: 20px;
}


/* CARDS */

.stat-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 19px;
    min-height: 115px;
    box-shadow: 0 5px 18px rgba(20,60,35,0.045);
    transition: all 0.25s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(20,60,35,0.11);
}

.stat-label {
    color: #647c6c;
    font-size: 13px;
    font-weight: 700;
}

.stat-value {
    color: #17653e;
    font-size: 29px;
    font-weight: 800;
    margin-top: 9px;
}

.stat-note {
    color: #708577;
    font-size: 12px;
}

.feature-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 5px 18px rgba(20,60,35,0.05);
    transition: all 0.25s ease;
    margin-bottom: 18px;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(20,60,35,0.12);
}

.feature-card img {
    width: 100%;
    height: 150px;
    object-fit: cover;
}

.feature-content {
    padding: 15px;
}

.feature-content h3 {
    color: #17452e !important;
    font-size: 17px;
    margin-bottom: 6px;
}

.feature-content p {
    color: #718477 !important;
    font-size: 13px;
}


/* IMAGES */

div[data-testid="stImage"] {
    overflow: hidden;
    border-radius: 14px;
    margin-bottom: 8px;
}

div[data-testid="stImage"] img {
    border-radius: 14px;
}


/* INFORMATION PANELS */

.info-panel {
    background: #ffffff !important;
    color: #183c2b !important;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 5px 17px rgba(20,60,35,0.045);
}

.info-panel h3 {
    color: #17653e !important;
}

.info-panel p,
.info-panel li {
    color: #315b40 !important;
}


/* METRICS */

div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dce9df;
    border-radius: 14px;
    padding: 17px;
    box-shadow: 0 5px 15px rgba(20,60,35,0.045);
}

div[data-testid="stMetric"] label {
    color: #526b5a !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #145b37 !important;
    font-weight: 800 !important;
}


/* ALERTS */

div[data-testid="stAlert"] {
    background: #ffffff !important;
    color: #183c2b !important;
    border: 1px solid #d8e8dc !important;
    border-radius: 13px !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] li {
    color: #183c2b !important;
}


/* TABLES AND PROGRESS */

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stProgress"] > div > div {
    background: #198653 !important;
}


/* FILE UPLOAD */

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #a8cbb1;
    border-radius: 13px;
    padding: 12px;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #718779;
    padding: 25px;
    font-size: 12px;
}


/* MOBILE */

@media (max-width: 768px) {
    .hero {
        min-height: 220px;
        padding: 24px;
    }

    .hero h1 {
        font-size: 28px;
    }

    .section-title {
        font-size: 21px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 7. REUSABLE UI FUNCTIONS
# ============================================================

def section_heading(title, subtitle):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True
    )


def feature_card(image_url, title, description):
    st.markdown(
        f"""
        <div class="feature-card">
            <img src="{image_url}" alt="Agriculture feature">
            <div class="feature-content">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def information_panel(title, content):
    st.markdown(
        f"""
        <div class="info-panel">
            <h3>{title}</h3>
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )


def get_module_progress(tasks):
    """Calculate module progress from saved task statuses."""

    status_values = {
        "Pending": 0,
        "In Progress": 50,
        "Done": 100
    }

    modules = [
        "Crop Recommendation",
        "Weather Advisory",
        "Disease Detection",
        "Market Analysis"
    ]

    progress = {}

    for module in modules:

        module_tasks = [
            task for task in tasks
            if task.get("module") == module
        ]

        if module_tasks:

            scores = [
                status_values.get(task.get("status"), 0)
                for task in module_tasks
            ]

            progress[module] = sum(scores) / len(scores)

        else:
            progress[module] = 0

    return progress


# ============================================================
# 8. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:15px 5px 22px 5px;">
            <div style="font-size:38px;">🌱</div>
            <h2 style="color:white !important;margin:0;">
                AgriSmart AI
            </h2>
            <p style="font-size:11px;color:#c1ddc9 !important;
            letter-spacing:1.5px;">
                INTELLIGENT FARMING PLATFORM
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### NAVIGATION")

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "📡 Project Radar & Tasks",
            "🌾 Crop Recommendation",
            "🌦️ Weather Advisory",
            "🍃 Disease Detection",
            "📈 Market Analysis",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### SYSTEM INFORMATION")

    st.markdown(
        """
        <div style="background:rgba(255,255,255,0.08);
        padding:13px;border-radius:12px;margin-bottom:8px;">
            <span style="color:#a7e6b7;">●</span>
            <b>Crop model</b><br>
            <small>Random Forest</small>
        </div>

        <div style="background:rgba(255,255,255,0.08);
        padding:13px;border-radius:12px;margin-bottom:8px;">
            <span style="color:#a7e6b7;">●</span>
            <b>Disease model</b><br>
            <small>MobileNetV2</small>
        </div>

        <div style="background:rgba(255,255,255,0.08);
        padding:13px;border-radius:12px;margin-bottom:8px;">
            <span style="color:#a7e6b7;">●</span>
            <b>Weather service</b><br>
            <small>Open-Meteo</small>
        </div>

        <div style="background:rgba(255,255,255,0.08);
        padding:13px;border-radius:12px;">
            <span style="color:#a7e6b7;">●</span>
            <b>Market analysis</b><br>
            <small>CSV and sample data</small>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("AI Crop Advisory System")
    st.caption("Smart India Hackathon Project")


# ============================================================
# 9. LOAD PROJECT TASKS
# ============================================================

tasks = load_tasks()

done_count = sum(
    1 for task in tasks
    if task.get("status") == "Done"
)

pending_count = sum(
    1 for task in tasks
    if task.get("status") == "Pending"
)

progress_count = sum(
    1 for task in tasks
    if task.get("status") == "In Progress"
)

overall_progress = (
    done_count / len(tasks) * 100
    if tasks else 0
)

module_progress = get_module_progress(tasks)


# ============================================================
# 10. MAIN DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                SMART FARMING • ARTIFICIAL INTELLIGENCE
            </div>

            <h1>
                Grow Smarter.<br>
                Farm Better.
            </h1>

            <p>
                Empowering farmers with intelligent crop recommendations,
                weather insights, plant disease detection and agricultural
                market analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    section_heading(
        "Agriculture Intelligence Centre",
        "Your complete AI-powered farming decision support platform."
    )

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        ("04", "AI Modules", "Integrated project modules"),
        ("22", "Crop Classes", "Crop recommendation categories"),
        ("39", "Disease Classes", "Disease classification categories"),
        (f"{overall_progress:.0f}%", "Project Progress", "Based on task statuses")
    ]

    for col, (value, label, note) in zip(
        [c1, c2, c3, c4], cards
    ):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                    <div class="stat-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    section_heading(
        "Project Activity Overview",
        "See your current project completion status."
    )

    p1, p2, p3 = st.columns(3)

    p1.metric("✅ Completed Tasks", done_count)
    p2.metric("🔄 In Progress", progress_count)
    p3.metric("⏳ Pending Tasks", pending_count)

    st.progress(overall_progress / 100)

    st.caption(
        f"{done_count} of {len(tasks)} tasks marked as completed."
    )

    section_heading(
        "Explore AI Solutions",
        "Choose an intelligent tool from the left navigation panel."
    )

    left, right = st.columns(2, gap="large")

    with left:

        feature_card(
            CROP_IMAGE,
            "🌾 Crop Recommendation",
            "Analyze NPK values, soil pH, rainfall, temperature and humidity "
            "to predict a suitable crop."
        )

        feature_card(
            WEATHER_IMAGE,
            "🌦️ Weather Advisory",
            "Check weather conditions and receive general farming alerts."
        )

    with right:

        feature_card(
            DISEASE_IMAGE,
            "🍃 Plant Disease Detection",
            "Explore common disease examples or upload a leaf photograph "
            "for AI classification."
        )

        feature_card(
            MARKET_IMAGE,
            "📈 Agricultural Market Analysis",
            "Compare mandi prices and analyze the supplied market data."
        )

    information_panel(
        "🌍 Our Mission",
        "To make agricultural decision-making more accessible "
        "through machine learning, deep learning and useful "
        "data-driven insights."
    )


# ============================================================
# 11. PROJECT RADAR AND TASK TRACKER
# ============================================================

elif page == "📡 Project Radar & Tasks":

    section_heading(
        "Project Radar & Task Management",
        "Monitor the progress of your AI Crop Advisory project."
    )

    st.markdown(
        """
        <div class="info-panel">
            <h3>📡 Project Monitoring Centre</h3>
            Track the development of each project module, identify
            pending work and update your task status as the project progresses.
        </div>
        """,
        unsafe_allow_html=True
    )

    # TOP PROJECT STATISTICS

    a, b, c, d = st.columns(4)

    a.metric("Total Tasks", len(tasks))
    b.metric("✅ Done", done_count)
    c.metric("🔄 In Progress", progress_count)
    d.metric("⏳ Pending", pending_count)

    st.markdown("---")

    # RADAR CHART

    section_heading(
        "Module Progress Radar",
        "The radar chart updates automatically when you change task statuses."
    )

    radar_labels = list(module_progress.keys())

    radar_values = [
        module_progress[label]
        for label in radar_labels
    ]

    radar_values_closed = radar_values + [radar_values[0]]
    angles = np.linspace(
        0,
        2 * np.pi,
        len(radar_labels),
        endpoint=False
    ).tolist()

    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(
        figsize=(8, 7),
        subplot_kw={"polar": True}
    )

    ax.plot(
        angles_closed,
        radar_values_closed,
        linewidth=2.5,
        marker="o"
    )

    ax.fill(
        angles_closed,
        radar_values_closed,
        alpha=0.20
    )

    ax.set_xticks(angles)
    ax.set_xticklabels(radar_labels, fontsize=9)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(
        ["20%", "40%", "60%", "80%", "100%"],
        fontsize=8
    )

    ax.set_title(
        "AGRI SMART PROJECT PROGRESS",
        pad=25,
        fontsize=13,
        fontweight="bold"
    )

    ax.grid(True)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # MODULE PROGRESS DETAILS

    section_heading(
        "Individual Module Progress",
        "Progress is calculated from the tasks assigned to each module."
    )

    for module, value in module_progress.items():

        st.write(f"**{module} — {value:.0f}%**")
        st.progress(value / 100)

    st.markdown("---")

    # TASK MANAGEMENT

    section_heading(
        "Project Task Board",
        "Change any task between Pending, In Progress and Done."
    )

    status_options = [
        "Pending",
        "In Progress",
        "Done"
    ]

    for index, task in enumerate(tasks):

        with st.container(border=True):

            left, right = st.columns([3, 1])

            with left:

                st.markdown(f"**{task.get('task', 'Unnamed task')}**")

                st.caption(
                    f"Module: {task.get('module', 'General')}"
                )

            with right:

                current_status = task.get("status", "Pending")

                if current_status not in status_options:
                    current_status = "Pending"

                st.selectbox(
                    "Task Status",
                    status_options,
                    index=status_options.index(current_status),
                    key=f"task_status_{index}"
                )

    if st.button(
        "💾 Save Task Updates",
        use_container_width=True
    ):

        updated_tasks = []

        for index, task in enumerate(tasks):

            updated_task = task.copy()

            updated_task["status"] = st.session_state.get(
                f"task_status_{index}",
                task.get("status", "Pending")
            )

            updated_tasks.append(updated_task)

        save_tasks(updated_tasks)

        st.success("Project task statuses saved successfully.")
        st.rerun()

    st.markdown("---")

    # ADD NEW TASK

    section_heading(
        "Add a New Project Task",
        "Create additional tasks when new project work is identified."
    )

    with st.form("add_task_form"):

        new_task_name = st.text_input(
            "Task Name",
            placeholder="Example: Test disease detection with new images"
        )

        new_task_module = st.selectbox(
            "Select Module",
            [
                "Crop Recommendation",
                "Weather Advisory",
                "Disease Detection",
                "Market Analysis",
                "Project Dashboard"
            ]
        )

        new_task_status = st.selectbox(
            "Initial Status",
            status_options
        )

        add_task = st.form_submit_button(
            "➕ Add Project Task",
            use_container_width=True
        )

    if add_task:

        if not new_task_name.strip():

            st.warning("Please enter a task name.")

        else:

            tasks.append({
                "task": new_task_name.strip(),
                "module": new_task_module,
                "status": new_task_status
            })

            save_tasks(tasks)

            st.success("New task added successfully.")
            st.rerun()

    st.caption(
        "Task information is stored locally in data/project_tasks.json."
    )


# ============================================================
# 12. CROP RECOMMENDATION
# ============================================================

elif page == "🌾 Crop Recommendation":

    section_heading(
        "Crop Recommendation Intelligence",
        "Find a suitable crop using soil and climate parameters."
    )

    st.image(CROP_IMAGE, use_container_width=True)

    information_panel(
        "🌱 Smart Crop Prediction",
        "Enter soil nutrient values and environmental conditions. "
        "The trained Random Forest model analyzes your inputs "
        "and recommends a suitable crop."
    )

    with st.form("crop_form"):

        st.subheader("01 — Soil Parameters")

        c1, c2, c3 = st.columns(3)

        with c1:
            nitrogen = st.number_input(
                "Nitrogen (N)",
                min_value=0.0,
                max_value=200.0,
                value=90.0
            )

        with c2:
            phosphorus = st.number_input(
                "Phosphorus (P)",
                min_value=0.0,
                max_value=200.0,
                value=42.0
            )

        with c3:
            potassium = st.number_input(
                "Potassium (K)",
                min_value=0.0,
                max_value=250.0,
                value=43.0
            )

        st.subheader("02 — Environmental Parameters")

        c4, c5, c6, c7 = st.columns(4)

        with c4:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=20.87
            )

        with c5:
            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=82.0
            )

        with c6:
            ph = st.number_input(
                "Soil pH",
                min_value=0.0,
                max_value=14.0,
                value=6.5
            )

        with c7:
            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                max_value=5000.0,
                value=203.0
            )

        submitted = st.form_submit_button(
            "🌱 Generate Crop Recommendation",
            use_container_width=True
        )

    if submitted:

        try:

            model = load_crop_model()

            data = pd.DataFrame(
                [[
                    nitrogen,
                    phosphorus,
                    potassium,
                    temperature,
                    humidity,
                    ph,
                    rainfall
                ]],
                columns=[
                    "N",
                    "P",
                    "K",
                    "temperature",
                    "humidity",
                    "ph",
                    "rainfall"
                ]
            )

            prediction = model.predict(data)[0]
            probabilities = model.predict_proba(data)[0]

            results = pd.DataFrame({
                "Crop": model.classes_,
                "Confidence": probabilities
            }).sort_values(
                "Confidence",
                ascending=False
            ).head(3)

            st.markdown("---")
            st.subheader("🎯 AI Prediction Results")

            st.markdown(
                f"""
                <div class="info-panel"
                     style="border-left:6px solid #198653;">
                    <div style="color:#63796b;font-size:13px;">
                        RECOMMENDED CROP
                    </div>
                    <div style="color:#17653e;font-size:32px;
                    font-weight:800;margin-top:8px;">
                        🌾 {str(prediction).upper()}
                    </div>
                    <div style="color:#63796b;font-size:12px;">
                        Prediction generated by Random Forest model
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Top 3 Crop Predictions")

            for _, row in results.iterrows():

                st.write(
                    f"**{row['Crop']}** — "
                    f"{row['Confidence'] * 100:.2f}%"
                )

                st.progress(float(row["Confidence"]))

            st.caption(
                "Confidence values are model outputs, not guarantees "
                "of actual crop yield or suitability."
            )

        except Exception as e:

            st.error(f"Crop recommendation error: {e}")


# ============================================================
# 13. WEATHER ADVISORY
# ============================================================

elif page == "🌦️ Weather Advisory":

    section_heading(
        "Weather Intelligence Centre",
        "Get weather information and general farming alerts for your city."
    )

    st.image(WEATHER_IMAGE, use_container_width=True)

    information_panel(
        "🌤️ Weather Monitoring",
        "Enter a city name to retrieve weather conditions and "
        "configured weather-based farming alerts."
    )

    with st.form("weather_form"):

        city = st.text_input(
            "Enter City Name",
            placeholder="Example: Phagwara, Ludhiana, Amritsar"
        )

        weather_submit = st.form_submit_button(
            "🌦️ Get Weather Report",
            use_container_width=True
        )

    if weather_submit:

        if not city.strip():

            st.warning("Please enter a city name.")

        else:

            with st.spinner("Fetching weather information..."):

                try:

                    weather = get_weather(city.strip())

                    if not isinstance(weather, dict):

                        st.error("Unexpected weather service response.")

                    elif "error" in weather:

                        st.error(weather["error"])

                    else:

                        st.markdown("---")

                        st.subheader(
                            f"🌤️ Weather Report: {weather.get('city', city)}"
                        )

                        st.caption(
                            f"{weather.get('region', '')}, "
                            f"{weather.get('country', '')}"
                        )

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "🌡️ Temperature",
                            f"{weather.get('temperature', 'N/A')} °C"
                        )

                        c2.metric(
                            "💧 Humidity",
                            f"{weather.get('humidity', 'N/A')}%"
                        )

                        c3.metric(
                            "🌧️ Rainfall",
                            f"{weather.get('precipitation', 'N/A')} mm"
                        )

                        c4, c5 = st.columns(2)

                        c4.metric(
                            "💨 Wind Speed",
                            f"{weather.get('wind_speed', 'N/A')} km/h"
                        )

                        c5.metric(
                            "☔ Rain Probability",
                            f"{weather.get('rain_probability', 'N/A')}%"
                        )

                        st.markdown("---")
                        st.subheader("🌾 Agricultural Weather Alerts")

                        for alert in weather.get("alerts", []):
                            st.warning(f"🌿 {alert}")

                        st.caption(
                            "Weather alerts are general guidance and "
                            "should not replace local agricultural advice."
                        )

                except Exception as e:

                    st.error(f"Weather service error: {e}")


# ============================================================
# 14. DISEASE DETECTION AND EXAMPLE GALLERY
# ============================================================

elif page == "🍃 Disease Detection":

    section_heading(
        "Plant Health & Disease Intelligence",
        "Learn about common plant diseases and analyze your own leaf images."
    )

    st.image(DISEASE_IMAGE, use_container_width=True)

    information_panel(
        "🔬 AI-Powered Plant Diagnosis",
        "Explore common disease examples below. When you are ready, "
        "upload a leaf photograph and the MobileNetV2 model will "
        "analyze its possible condition."
    )

    # -----------------------------------------------
    # DISEASE EXAMPLE GALLERY
    # -----------------------------------------------

    section_heading(
        "🌿 Common Plant Disease Examples",
        "View example leaf images, symptoms and basic management guidance."
    )

    disease_images = find_disease_images()

    example_classes = [
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Bacterial_spot",
        "Tomato___Leaf_Mold",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Pepper,_bell___Bacterial_spot"
    ]

    available_examples = [
        class_name for class_name in example_classes
        if class_name in DISEASE_INFO
    ]

    gallery_columns = st.columns(2)

    for index, class_name in enumerate(available_examples):

        info = DISEASE_INFO[class_name]

        with gallery_columns[index % 2]:

            with st.container(border=True):

                st.markdown(f"### 🍃 {info['name']}")

                if class_name in disease_images:

                    st.image(
                        disease_images[class_name],
                        caption=f"PlantVillage example: {info['name']}",
                        use_container_width=True
                    )

                else:

                    st.info(
                        "A dataset image for this example was not found "
                        "in the expected PlantVillage folder."
                    )

                st.caption(f"Crop: {info['crop']}")

                with st.expander("View symptoms and management advice"):

                    st.markdown("**Common symptoms**")

                    for symptom in info["symptoms"]:
                        st.write(f"• {symptom}")

                    st.markdown("**Basic management guidance**")

                    for advice in info["management"]:
                        st.write(f"• {advice}")

    st.info(
        "The gallery uses examples from the PlantVillage dataset where "
        "available. Images are reference examples, not a diagnosis of "
        "your own plant."
    )

    st.markdown("---")

    # -----------------------------------------------
    # UPLOAD AND ANALYZE
    # -----------------------------------------------

    section_heading(
        "🔍 Analyze Your Plant",
        "Upload a clear photograph of a single plant leaf."
    )

    uploaded_image = st.file_uploader(
        "Upload Plant Leaf Photograph",
        type=["jpg", "jpeg", "png"],
        key="disease_upload"
    )

    if uploaded_image is not None:

        try:

            image = Image.open(uploaded_image).convert("RGB")

            st.image(
                image,
                caption="Your Uploaded Plant Image",
                width=350
            )

            if st.button(
                "🔍 Analyze Plant Image",
                use_container_width=True
            ):

                with st.spinner("AI model is analyzing the image..."):

                    result = predict_disease(image)

                if not isinstance(result, dict):

                    st.error("Unexpected disease detection response.")

                elif "error" in result:

                    st.error(result["error"])

                else:

                    st.markdown("---")
                    st.subheader("🍃 Disease Detection Results")

                    if result.get("is_background", False):

                        st.warning(
                            "The image may not contain a recognizable "
                            "plant leaf. Please upload a clearer leaf image."
                        )

                    else:

                        condition = result.get("condition", "Unknown")
                        crop = result.get("crop", "Unknown")
                        confidence = float(result.get("confidence", 0))

                        st.markdown(
                            f"""
                            <div class="info-panel"
                                 style="border-left:6px solid #198653;">
                                <div style="font-size:13px;color:#63796b;">
                                    DETECTED CONDITION
                                </div>
                                <div style="font-size:25px;
                                color:#17653e;font-weight:800;margin-top:8px;">
                                    🍃 {condition}
                                </div>
                                <div style="color:#63796b;">
                                    Crop: {crop}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        c1, c2 = st.columns(2)

                        c1.metric("Crop", crop)

                        c2.metric(
                            "Prediction Confidence",
                            f"{confidence:.2f}%"
                        )

                        # Find matching disease guidance
                        normalized_condition = re.sub(
                            r"[^a-z0-9]+",
                            "",
                            condition.lower()
                        )

                        matched_info = None

                        for key, info in DISEASE_INFO.items():

                            key_condition = key.split("___", 1)[-1]

                            normalized_key = re.sub(
                                r"[^a-z0-9]+",
                                "",
                                key_condition.lower()
                            )

                            if (
                                normalized_key in normalized_condition
                                or normalized_condition in normalized_key
                            ):

                                matched_info = info
                                break

                        if matched_info:

                            st.markdown("---")
                            st.subheader("📋 Disease Information")

                            st.markdown(
                                f"**Disease:** {matched_info['name']}"
                            )

                            st.markdown("**Common symptoms**")

                            for symptom in matched_info["symptoms"]:
                                st.write(f"• {symptom}")

                            st.markdown("**Suggested management steps**")

                            for advice in matched_info["management"]:
                                st.write(f"• {advice}")

                        elif "healthy" in condition.lower():

                            st.success(
                                "The model classified this image as healthy. "
                                "Continue regular crop monitoring."
                            )

                        else:

                            st.info(
                                "Detailed guidance for this particular "
                                "classification is not yet available in "
                                "the example database."
                            )

                    # TOP THREE PREDICTIONS

                    st.markdown("---")
                    st.subheader("Top AI Predictions")

                    for item in result.get("top_predictions", []):

                        label = item.get("class", "Unknown")
                        score = float(item.get("confidence", 0))

                        st.write(
                            f"**{label}** — {score:.2f}%"
                        )

                        st.progress(
                            max(0.0, min(1.0, score / 100))
                        )

                    st.warning(
                        "Important: This is an AI-based preliminary "
                        "classification, not a confirmed diagnosis. "
                        "Image quality, lighting and real-field conditions "
                        "can affect predictions. Confirm serious symptoms "
                        "with an agricultural expert before treatment."
                    )

        except Exception as e:

            st.error(f"Disease detection error: {e}")


# ============================================================
# 15. MARKET ANALYSIS
# ============================================================

elif page == "📈 Market Analysis":

    section_heading(
        "Agricultural Market Intelligence",
        "Compare mandi prices using demonstration data or your own CSV."
    )

    st.image(MARKET_IMAGE, use_container_width=True)

    st.info(
        "Built-in prices are illustrative demonstration data, not "
        "verified live mandi prices. Upload verified market data "
        "to analyze actual prices."
    )

    market_data = get_sample_market_data()

    uploaded_csv = st.file_uploader(
        "Upload Mandi Price CSV",
        type=["csv"],
        key="market_upload"
    )

    if uploaded_csv is not None:

        try:

            market_data = pd.read_csv(uploaded_csv)

            required = [
                "Crop",
                "Mandi",
                "Min Price",
                "Max Price",
                "Modal Price"
            ]

            missing = [
                col for col in required
                if col not in market_data.columns
            ]

            if missing:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing)
                )

                market_data = None

            else:

                price_columns = [
                    "Min Price",
                    "Max Price",
                    "Modal Price"
                ]

                for col in price_columns:

                    market_data[col] = pd.to_numeric(
                        market_data[col],
                        errors="coerce"
                    )

                if market_data[required].isnull().any().any():

                    st.error(
                        "CSV contains missing or invalid values."
                    )

                    market_data = None

                elif (market_data[price_columns] < 0).any().any():

                    st.error("Market prices cannot be negative.")

                    market_data = None

                elif (
                    (market_data["Min Price"] > market_data["Max Price"])
                    | (market_data["Modal Price"] < market_data["Min Price"])
                    | (market_data["Modal Price"] > market_data["Max Price"])
                ).any():

                    st.error(
                        "Please verify the market price ranges."
                    )

                    market_data = None

        except Exception as e:

            st.error(f"CSV upload error: {e}")
            market_data = None

    if market_data is not None:

        st.subheader("📋 Available Market Data")

        st.dataframe(
            market_data,
            use_container_width=True,
            hide_index=True
        )

        crops = sorted(
            market_data["Crop"].dropna().astype(str).unique()
        )

        if crops:

            selected_crop = st.selectbox(
                "Select Crop for Market Analysis",
                crops
            )

            try:

                analysis = analyze_market_data(
                    market_data,
                    selected_crop
                )

                st.markdown("---")

                st.subheader(
                    f"📈 Market Analysis: {selected_crop}"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Average Modal Price",
                    f"₹{analysis['average_price']:.2f}"
                )

                c2.metric(
                    "Highest Modal Price",
                    f"₹{analysis['highest_price']:.2f}"
                )

                c3.metric(
                    "Lowest Modal Price",
                    f"₹{analysis['lowest_price']:.2f}"
                )

                st.success(
                    f"🏆 Highest Modal Price Mandi: "
                    f"{analysis['best_mandi']}"
                )

                st.subheader("Mandi Price Comparison")

                chart_data = analysis["data"]

                st.bar_chart(
                    chart_data.set_index("Mandi")["Modal Price"]
                )

                st.caption(
                    "Prices are assumed to be in INR per quintal."
                )

            except Exception as e:

                st.error(f"Market analysis error: {e}")

        else:

            st.warning("No crop records found in the market data.")


# ============================================================
# 16. ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    section_heading(
        "About AgriSmart AI",
        "Technology supporting smarter agricultural decisions."
    )

    st.image(HERO_IMAGE, use_container_width=True)

    information_panel(
        "🎯 Project Objective",
        "The AI Crop Advisory & Farmer Decision Support System "
        "aims to provide farmers with data-driven agricultural "
        "information through machine learning and deep learning."
    )

    information_panel(
        "🧠 Technologies Used",
        "Python • Streamlit • Pandas • Scikit-learn • "
        "TensorFlow • MobileNetV2 • Open-Meteo • Matplotlib"
    )

    information_panel(
        "⚙️ Main Components",
        """
        <ul>
            <li>Machine Learning-based crop recommendation.</li>
            <li>Weather information and general farming alerts.</li>
            <li>Deep Learning-based plant disease classification.</li>
            <li>Agricultural mandi price analysis.</li>
            <li>Project progress radar and task management.</li>
        </ul>
        """
    )

    information_panel(
        "📊 Model Performance",
        """
        Crop model held-out dataset accuracy: <b>99.32%</b>.<br>
        Disease model validation accuracy: <b>94.49%</b>.<br><br>
        These figures describe performance on the respective
        evaluation datasets, not guaranteed real-world accuracy.
        """
    )

    information_panel(
        "⚠️ Project Limitations",
        """
        Model performance on training datasets does not guarantee
        equivalent performance in real farms. Weather alerts use
        configured thresholds, and built-in market prices are
        demonstration data rather than verified live prices.
        Disease gallery guidance is general information and should
        not replace expert diagnosis.
        """
    )


# ============================================================
# 17. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <h3 style="color:#17452e !important;">🌱 AgriSmart AI</h3>

        AI Crop Advisory & Farmer Decision Support System

        <br><br>

        Built with Python • Machine Learning • Deep Learning • Streamlit

        <br>

        Smart Agriculture | Intelligent Farming
    </div>
    """,
    unsafe_allow_html=True
)
