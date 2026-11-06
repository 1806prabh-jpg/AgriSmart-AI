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
TEAM_FILE = DATA_DIR / "team_members.json"
SETTINGS_FILE = DATA_DIR / "app_settings.json"

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
# 3. LOCAL DATA STORAGE HELPERS
# ============================================================

def load_json_file(file_path, default_value):
    """Read a JSON file safely and return a default if unavailable."""
    try:
        if not file_path.exists():
            return default_value

        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except (OSError, json.JSONDecodeError, TypeError):
        return default_value


def save_json_file(file_path, data):
    """Save JSON safely using a temporary file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    temporary_path.replace(file_path)


# ============================================================
# 4. PROJECT TASK TRACKER
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
    save_json_file(TASK_FILE, tasks)


# ============================================================
# 5. DISEASE DATASET IMAGE FINDER
# ============================================================

@st.cache_data
def find_disease_images():
    """
    Find representative images from the downloaded PlantVillage dataset.
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
# 6. MODEL LOADING
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
# 7. PROFESSIONAL FRONTEND CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root{
    --forest:#183B2B;
    --green:#64845A;
    --cream:#F5F4EC;
    --ink:#26352B;
    --muted:#758075;
    --line:#E3E7DD;
    --white:#FFFFFF;
}

html,body,[class*="css"]{
    font-family:'DM Sans',sans-serif;
}

.stApp{
    background:var(--cream);
    color:var(--ink);
}

.block-container{
    max-width:1480px;
    padding-top:1.7rem;
    padding-bottom:3rem;
}

header[data-testid="stHeader"]{
    background:transparent;
}

h1,h2,h3,h4{
    font-family:'Manrope',sans-serif!important;
    color:var(--forest)!important;
    font-weight:800!important;
}

p,li{
    color:var(--ink);
}

[data-testid="stSidebar"]{
    background:var(--forest);
    border-right:1px solid #31523D;
}

[data-testid="stSidebar"] *{
    color:#F4F7F1!important;
}

[data-testid="stSidebar"] hr{
    border-color:rgba(255,255,255,.16);
}

[data-testid="stSidebar"] .stButton{
    margin:0 0 4px 0;
}

[data-testid="stSidebar"] .stButton>button{
    background:transparent!important;
    border:1px solid transparent!important;
    color:#EAF1E8!important;
    text-align:left!important;
    justify-content:flex-start!important;
    padding:9px 12px!important;
    min-height:39px!important;
    font-weight:600!important;
}

[data-testid="stSidebar"] .stButton>button:hover{
    background:rgba(255,255,255,.10)!important;
    border-color:rgba(255,255,255,.14)!important;
    transform:none!important;
    box-shadow:none!important;
}

[data-testid="stSidebar"] .stButton>button[kind="primary"]{
    background:#315D40!important;
    border-color:#739477!important;
    color:#FFFFFF!important;
}

.nav-group-label{
    font-size:10px;
    letter-spacing:1.4px;
    font-weight:800;
    color:#AFC8B2;
    padding:15px 10px 7px;
    text-transform:uppercase;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{
    color:#DCE8DC!important;
}

.stButton>button,.stFormSubmitButton>button{
    background:var(--forest)!important;
    color:#fff!important;
    border:1px solid #315D40!important;
    border-radius:9px!important;
    font-weight:700!important;
    min-height:42px;
    transition:all .18s ease!important;
}

.stButton>button:hover,.stFormSubmitButton>button:hover{
    background:#315D40!important;
    transform:translateY(-1px);
    box-shadow:0 5px 12px rgba(24,59,43,.14)!important;
}

.stTextInput input,.stNumberInput input,.stTextArea textarea{
    background:#fff!important;
    color:var(--ink)!important;
    border:1px solid #D6DED3!important;
    border-radius:8px!important;
}

.stTextInput label,.stNumberInput label,.stSelectbox label,.stFileUploader label{
    color:#405B46!important;
    font-weight:700!important;
}

div[data-baseweb="select"]>div{
    background:#fff!important;
    color:var(--ink)!important;
    border-radius:8px!important;
}

div[data-testid="stForm"]{
    background:#fff!important;
    border:1px solid var(--line);
    border-radius:14px;
    padding:20px;
    box-shadow:0 4px 14px rgba(24,59,43,.035);
}

.hero{
    background:linear-gradient(90deg,rgba(14,43,29,.90),rgba(14,43,29,.30)),
    url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85");
    background-size:cover;
    background-position:center;
    min-height:260px;
    border-radius:18px;
    padding:36px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    margin-bottom:24px;
}

.hero h1{
    color:#fff!important;
    font-size:38px;
    line-height:1.2;
    max-width:700px;
}

.hero p{
    color:#F0F4EC!important;
    font-size:15px;
    max-width:660px;
}

.eyebrow{
    color:#D4E2C9!important;
    font-weight:700;
    letter-spacing:1.5px;
    font-size:11px;
}

.section-title{
    font-family:'Manrope',sans-serif;
    font-size:24px;
    font-weight:800;
    color:var(--forest);
    margin-top:18px;
    margin-bottom:5px;
}

.section-subtitle{
    color:var(--muted);
    font-size:14px;
    margin-bottom:18px;
}

.stat-card,.feature-card,.info-panel{
    background:#fff;
    border:1px solid var(--line);
    border-radius:14px;
    box-shadow:0 3px 12px rgba(24,59,43,.035);
    transition:transform .18s ease,box-shadow .18s ease;
}

.stat-card{
    padding:18px;
    min-height:112px;
}

.stat-card:hover,.feature-card:hover{
    transform:translateY(-3px);
    box-shadow:0 9px 20px rgba(24,59,43,.08);
}

.stat-label{
    color:#758075;
    font-size:12px;
    font-weight:700;
}

.stat-value{
    color:var(--forest);
    font-size:28px;
    font-weight:800;
    margin-top:8px;
}

.stat-note{
    color:var(--muted);
    font-size:11px;
}

.feature-card{
    overflow:hidden;
    margin-bottom:16px;
}

.feature-card img{
    width:100%;
    height:145px;
    object-fit:cover;
}

.feature-content{
    padding:14px;
}

.feature-content h3{
    font-size:16px;
    margin:0 0 6px;
}

.feature-content p{
    color:var(--muted)!important;
    font-size:13px;
    margin:0;
}

.info-panel{
    padding:18px;
    margin-bottom:14px;
}

.info-panel h3{
    color:var(--forest)!important;
    font-size:17px;
}

.info-panel p,.info-panel li{
    color:#405B46!important;
}

div[data-testid="stImage"]{
    overflow:hidden;
    border-radius:12px;
    margin-bottom:8px;
}

div[data-testid="stImage"] img{
    border-radius:12px;
}

div[data-testid="stMetric"]{
    background:#fff;
    border:1px solid var(--line);
    border-radius:12px;
    padding:15px;
}

div[data-testid="stMetric"] label{
    color:#667568!important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"]{
    color:var(--forest)!important;
    font-weight:800!important;
}

div[data-testid="stAlert"]{
    background:#fff!important;
    color:var(--ink)!important;
    border:1px solid var(--line)!important;
    border-radius:11px!important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] li{
    color:var(--ink)!important;
}

div[data-testid="stDataFrame"]{
    border:1px solid var(--line);
    border-radius:10px;
    overflow:hidden;
}

div[data-testid="stProgress"]>div>div{
    background:#64845A!important;
}

[data-testid="stFileUploader"]{
    background:#fff;
    border:1px dashed #B9C8B5;
    border-radius:11px;
    padding:10px;
}

.footer{
    text-align:center;
    color:#7A857A;
    padding:24px;
    font-size:12px;
    border-top:1px solid var(--line);
    margin-top:28px;
}

@media(max-width:768px){
    .hero{
        min-height:210px;
        padding:22px;
    }

    .hero h1{
        font-size:28px;
    }

    .section-title{
        font-size:21px;
    }
}

</style>
""", unsafe_allow_html=True)


# Load saved preferences once per Streamlit session.

DEFAULT_SETTINGS = {
    "preferred_language": "English",
    "display_density": "Comfortable"
}

if "app_settings" not in st.session_state:

    saved_settings = load_json_file(
        SETTINGS_FILE,
        DEFAULT_SETTINGS.copy()
    )

    if not isinstance(saved_settings, dict):
        saved_settings = DEFAULT_SETTINGS.copy()

    st.session_state.app_settings = {
        **DEFAULT_SETTINGS,
        **saved_settings
    }


if st.session_state.app_settings.get("display_density") == "Compact":

    st.markdown("""
    <style>
    .block-container{
        padding-top:1rem!important;
        padding-bottom:1.5rem!important;
    }

    div[data-testid="stMetric"]{
        padding:10px!important;
    }

    .stat-card{
        padding:13px!important;
        min-height:96px!important;
    }

    div[data-testid="stVerticalBlock"]{
        gap:.75rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 8. REUSABLE UI FUNCTIONS
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
# 9. SIDEBAR NAVIGATION
# ============================================================

NAVIGATION_GROUPS = {
    "OVERVIEW": ["🏠 Dashboard"],

    "FARM INTELLIGENCE": [
        "🌾 Crop Recommendation",
        "🌦️ Weather Advisory",
        "🍃 Disease Detection"
    ],

    "MARKET & ANALYTICS": [
        "📈 Market Analysis",
        "📡 Project Tasks & Radar",
        "📄 Reports"
    ],

    "WORKSPACE": [
        "👥 Team",
        "❔ Help & Support"
    ],

    "SYSTEM": [
        "⚙️ Settings",
        "ℹ️ About Project"
    ]
}


def set_active_page(page_name):
    st.session_state.active_page = page_name


if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Dashboard"


with st.sidebar:

    st.markdown(
        """
        <div style="padding:14px 7px 18px 7px;">
            <div style="font-size:34px;">🌱</div>
            <div style="font-size:23px;font-weight:800;color:#FFFFFF;">
                AgriSmart AI
            </div>
            <div style="font-size:10px;color:#C1DDC9;
                        letter-spacing:1.35px;margin-top:3px;">
                INTELLIGENT FARMING PLATFORM
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    for group_name, pages in NAVIGATION_GROUPS.items():

        st.markdown(
            f'<div class="nav-group-label">{group_name}</div>',
            unsafe_allow_html=True
        )

        for page_name in pages:

            st.button(
                page_name,
                key=f"navigation_{page_name}",
                type=(
                    "primary"
                    if st.session_state.active_page == page_name
                    else "secondary"
                ),
                use_container_width=True,
                on_click=set_active_page,
                args=(page_name,)
            )

    st.markdown("---")

    st.markdown(
        "<div class='nav-group-label'>SYSTEM STATUS</div>",
        unsafe_allow_html=True
    )

    status_items = [
        ("Crop model", "Random Forest"),
        ("Disease model", "MobileNetV2"),
        ("Weather service", "Open-Meteo"),
        ("Market analysis", "CSV + sample data")
    ]

    for status_title, status_detail in status_items:

        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,.08);
                        padding:10px 12px;border-radius:9px;
                        margin-bottom:7px;">
                <div style="font-size:12px;font-weight:700;color:#FFFFFF;">
                    ● {status_title}
                </div>
                <div style="font-size:11px;color:#C1DDC9;margin-left:14px;">
                    {status_detail}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption("AI Crop Advisory & Farmer Decision Support System")
    st.caption("Smart India Hackathon Project")


page = st.session_state.active_page


# ============================================================
# 10. LOAD PROJECT TASKS
# ============================================================

tasks = load_tasks()

if "team_members" not in st.session_state:

    saved_team = load_json_file(TEAM_FILE, [])

    st.session_state.team_members = (
        saved_team if isinstance(saved_team, list) else []
    )


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
    (done_count + progress_count * 0.5) / len(tasks) * 100
    if tasks else 0
)

module_progress = get_module_progress(tasks)


# ============================================================
# 11. MAIN DASHBOARD
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
        ("04", "Advisory Modules", "2 trained models + 2 data services"),
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
        f"Weighted project progress: {overall_progress:.0f}% "
        "(Done = 100%, In Progress = 50%, Pending = 0%)."
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
# 12. PROJECT RADAR AND TASK TRACKER
# ============================================================

elif page == "📡 Project Tasks & Radar":

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
# 16. TEAM, REPORTS, SETTINGS AND SUPPORT
# ============================================================

elif page == "👥 Team":

    section_heading(
        "Team Workspace",
        "Keep your project contributors, responsibilities and contributions organised."
    )

    st.info(
        "Add your actual project members below. "
        "The application does not create fictional team profiles."
    )

    with st.form("team_member_form", clear_on_submit=True):

        st.subheader("Add a team member")

        member_name = st.text_input(
            "Full name",
            placeholder="Enter the member's name"
        )

        member_role = st.selectbox(
            "Project role",
            [
                "Project Lead",
                "Machine Learning Developer",
                "Backend Developer",
                "Frontend Developer",
                "Data & Research",
                "Documentation & Testing",
                "Other"
            ]
        )

        member_contribution = st.text_area(
            "Contribution / responsibility",
            placeholder="Describe the work handled by this member",
            height=90
        )

        add_member = st.form_submit_button(
            "＋ Add team member",
            use_container_width=True
        )

    if add_member:

        cleaned_name = member_name.strip()

        if not cleaned_name:

            st.warning("Please enter the member's name.")

        elif any(
            m.get("name", "").casefold() == cleaned_name.casefold()
            for m in st.session_state.team_members
        ):

            st.warning("A team member with this name already exists.")

        else:

            st.session_state.team_members.append({
                "name": cleaned_name,
                "role": member_role,
                "contribution": member_contribution.strip() or "Not specified"
            })

            save_json_file(
                TEAM_FILE,
                st.session_state.team_members
            )

            st.success("Team member saved successfully.")
            st.rerun()

    st.markdown("---")
    st.subheader("Current project team")

    members = st.session_state.team_members

    if members:

        st.metric("Team members", len(members))

        st.dataframe(
            pd.DataFrame(members),
            use_container_width=True,
            hide_index=True
        )

        team_csv = (
            pd.DataFrame(members)
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download team directory (CSV)",
            team_csv,
            "agrismart_team.csv",
            "text/csv",
            use_container_width=True
        )

        st.subheader("Remove a member")

        member_names = [
            m.get("name", "Unnamed")
            for m in members
        ]

        selected_member = st.selectbox(
            "Select member",
            member_names,
            key="remove_team_member"
        )

        if st.button("Remove selected member", type="secondary"):

            remove_index = member_names.index(selected_member)

            st.session_state.team_members.pop(remove_index)

            save_json_file(
                TEAM_FILE,
                st.session_state.team_members
            )

            st.success("Team member removed.")
            st.rerun()

    else:

        st.markdown(
            """
            <div class='info-panel'>
                <h3>No team members added yet</h3>
                Use the form above to add your real project contributors.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption(
        "Team information is saved locally in data/team_members.json "
        "and remains available after restarting the app on the same installation."
    )


elif page == "📄 Reports":

    section_heading(
        "Reports & Exports",
        "Download project progress and model evaluation information."
    )

    report_tasks = pd.DataFrame(tasks)

    st.subheader("Project task report")

    st.dataframe(
        report_tasks,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download task report (CSV)",
        report_tasks.to_csv(index=False).encode("utf-8"),
        "agrismart_project_tasks.csv",
        "text/csv",
        use_container_width=True
    )

    st.subheader("Model evaluation summary")

    crop_metrics = load_crop_metrics()

    metrics_payload = {
        "crop_model": crop_metrics,
        "disease_model": {},
        "project_task_summary": {
            "total": len(tasks),
            "done": done_count,
            "in_progress": progress_count,
            "pending": pending_count,
            "weighted_progress_percent": round(
                overall_progress,
                2
            )
        }
    }

    disease_metrics_path = (
        BASE_DIR / "models" / "disease_metrics.json"
    )

    if disease_metrics_path.exists():

        try:

            with open(
                disease_metrics_path,
                "r",
                encoding="utf-8"
            ) as metric_file:

                metrics_payload["disease_model"] = json.load(
                    metric_file
                )

        except (OSError, json.JSONDecodeError):

            st.warning("Disease metrics file could not be read.")

    st.json(metrics_payload)

    st.download_button(
        "Download project report (JSON)",
        json.dumps(
            metrics_payload,
            indent=4
        ).encode("utf-8"),
        "agrismart_project_report.json",
        "application/json",
        use_container_width=True
    )

    st.caption(
        "Reports contain the metrics and task data currently available "
        "in the project; they do not represent live market or farm outcomes."
    )
elif page == "⚙️ Settings":

    section_heading(
        "Application Settings",
        "Manage display preferences for your AgriSmart AI workspace."
    )

    current_settings = st.session_state.app_settings

    language_options = [
        "English",
        "Punjabi (planned)",
        "Hindi (planned)"
    ]

    density_options = [
        "Comfortable",
        "Compact"
    ]

    with st.form("settings_form"):

        st.subheader("Display preferences")

        preferred_language = st.selectbox(
            "Preferred language",
            language_options,
            index=(
                language_options.index(
                    current_settings.get(
                        "preferred_language",
                        "English"
                    )
                )
                if current_settings.get(
                    "preferred_language",
                    "English"
                ) in language_options
                else 0
            )
        )

        display_density = st.selectbox(
            "Display density",
            density_options,
            index=(
                density_options.index(
                    current_settings.get(
                        "display_density",
                        "Comfortable"
                    )
                )
                if current_settings.get(
                    "display_density",
                    "Comfortable"
                ) in density_options
                else 0
            )
        )

        save_settings = st.form_submit_button(
            "Save preferences",
            use_container_width=True
        )

    if save_settings:

        updated_settings = {
            "preferred_language": preferred_language,
            "display_density": display_density
        }

        save_json_file(
            SETTINGS_FILE,
            updated_settings
        )

        st.session_state.app_settings = updated_settings

        st.success(
            "Preferences saved. Display density will apply immediately after this refresh."
        )

        st.rerun()

    st.markdown("---")
    st.subheader("Application information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.write("**Application:** AgriSmart AI")
        st.write("**Interface:** Streamlit")
        st.write("**Crop model:** Random Forest Classifier")

    with info_col2:

        st.write("**Disease model:** MobileNetV2")
        st.write("**Weather provider:** Open-Meteo")
        st.write("**Market data:** Sample data / uploaded CSV")

    st.caption(
        "Punjabi and Hindi are listed as future language options; "
        "the current interface is available in English. Preferences "
        "are stored locally in data/app_settings.json."
    )


elif page == "❔ Help & Support":

    section_heading(
        "Help & Support",
        "Quick guidance for using the agricultural decision-support tools."
    )

    with st.expander(
        "How do I use Crop Recommendation?",
        expanded=True
    ):

        st.write(
            "Enter soil NPK values, temperature, humidity, soil pH "
            "and rainfall. Submit the form to view the model's crop "
            "prediction and top three class scores."
        )

    with st.expander("How do I use Disease Detection?"):

        st.write(
            "Upload a clear photograph of one leaf. The model predicts "
            "one of its trained classes. Results are preliminary and "
            "should be confirmed by an agricultural expert when "
            "symptoms are serious."
        )

    with st.expander("How do I use Weather Advisory?"):

        st.write(
            "Enter a city name. The application retrieves weather "
            "information through Open-Meteo and displays general "
            "threshold-based alerts."
        )

    with st.expander("How do I use Market Analysis?"):

        st.write(
            "Use the demonstration dataset or upload a CSV containing "
            "Crop, Mandi, Min Price, Max Price and Modal Price. "
            "Built-in sample prices are illustrative, not live verified prices."
        )

    with st.expander("Are predictions guaranteed?"):

        st.write(
            "No. Models are decision-support tools. Dataset performance "
            "does not guarantee results under real farm conditions. "
            "Validate important decisions with local agricultural experts."
        )


# ============================================================
# 17. ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    section_heading(
        "About AgriSmart AI",
        "Technology supporting smarter agricultural decisions."
    )

    st.image(
        HERO_IMAGE,
        use_container_width=True
    )

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
# 18. FOOTER
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