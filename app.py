import inspect
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime
import os
import urllib.request

# ═══════════════════════════════════════════════════════════
#                DATA & CONFIGURATION
# ═══════════════════════════════════════════════════════════

# MST 50-tone skin scale (RGB)
skin_tones = {
    1: (255, 235, 230), 2: (255, 228, 225), 3: (255, 224, 220), 4: (255, 220, 210), 5: (255, 215, 205),
    6: (255, 210, 200), 7: (255, 205, 195), 8: (255, 205, 190), 9: (250, 200, 185), 10: (245, 195, 180),
    11: (240, 190, 175), 12: (240, 185, 170), 13: (240, 180, 165), 14: (240, 180, 150), 15: (235, 175, 145),
    16: (230, 170, 140), 17: (225, 165, 135), 18: (220, 160, 130), 19: (215, 155, 125), 20: (210, 150, 120),
    21: (205, 145, 115), 22: (200, 140, 110), 23: (195, 135, 105), 24: (190, 130, 100), 25: (185, 125, 95),
    26: (180, 120, 90), 27: (175, 115, 85), 28: (170, 110, 80), 29: (165, 105, 75), 30: (160, 100, 70),
    31: (155, 95, 65), 32: (150, 90, 60), 33: (145, 85, 58), 34: (140, 82, 56), 35: (135, 80, 54),
    36: (130, 75, 52), 37: (125, 72, 50), 38: (120, 70, 50), 39: (115, 67, 48), 40: (110, 65, 46),
    41: (105, 60, 44), 42: (100, 58, 42), 43: (95, 55, 40), 44: (90, 50, 40), 45: (85, 48, 38),
    46: (80, 45, 36), 47: (70, 40, 32), 48: (65, 35, 28), 49: (60, 30, 25), 50: (50, 25, 20)
}

# Recommendations based on 50-tone scale
mst_recommendations = {
    1: ["Soft Pink", "Powder Blue", "Lavender", "Mint Green", "Peach"],
    2: ["Baby Blue", "Rose Pink", "Light Lavender", "Cream", "Soft Coral"],
    3: ["Lavender", "Baby Blue", "Mint", "Soft Pink", "Light Grey"],
    4: ["Periwinkle", "Blush Pink", "Sage Green", "Ivory", "Light Peach"],
    5: ["Sky Blue", "Rose", "Lilac", "Champagne", "Soft Yellow"],
    6: ["Powder Blue", "Dusty Pink", "Seafoam", "Vanilla", "Light Coral"],
    7: ["Cornflower Blue", "Mauve", "Mint", "Cream", "Apricot"],
    8: ["Cerulean", "Pink", "Aqua", "Beige", "Peach"],
    9: ["Azure", "Salmon", "Turquoise", "Taupe", "Coral"],
    10: ["Bright Blue", "Coral Pink", "Teal", "Sand", "Melon"],
    11: ["Teal", "Coral", "Periwinkle", "Camel", "Rose Gold"],
    12: ["Emerald", "Peach", "Royal Blue", "Tan", "Copper"],
    13: ["Coral", "Teal", "Aqua", "Dusty Rose", "Olive Green"],
    14: ["Turquoise", "Salmon", "Navy", "Khaki", "Rust"],
    15: ["Jade", "Apricot", "Cobalt", "Caramel", "Brick Red"],
    16: ["Seafoam", "Tangerine", "Sapphire", "Mocha", "Terracotta"],
    17: ["Mint", "Coral", "Deep Blue", "Bronze", "Burnt Sienna"],
    18: ["Aquamarine", "Peach", "Indigo", "Cognac", "Cinnamon"],
    19: ["Turquoise", "Mango", "Navy Blue", "Chestnut", "Paprika"],
    20: ["Teal", "Cantaloupe", "Royal Blue", "Walnut", "Clay"],
    21: ["Emerald", "Tangerine", "Cobalt", "Camel", "Crimson"],
    22: ["Bright Blue", "Emerald", "Turquoise", "Deep Purple", "Coral"],
    23: ["Peacock Blue", "Orange", "Violet", "Chocolate", "Red"],
    24: ["Sapphire", "Mango", "Plum", "Espresso", "Scarlet"],
    25: ["Electric Blue", "Papaya", "Eggplant", "Coffee", "Ruby"],
    26: ["White", "Black", "Red", "Cobalt Blue", "Magenta"],
    27: ["Bright Turquoise", "Coral Red", "Purple", "Dark Brown", "Hot Pink"],
    28: ["Cyan", "Flame Orange", "Deep Purple", "Mahogany", "Fuchsia"],
    29: ["Azure", "Burnt Orange", "Royal Purple", "Umber", "Cerise"],
    30: ["Cerulean", "Rust", "Violet", "Sepia", "Rose Red"],
    31: ["Cobalt", "Mustard", "Magenta", "Chocolate", "Lime"],
    32: ["Gold", "Bronze", "Mustard", "Bright Green", "Royal Blue"],
    33: ["Canary Yellow", "Copper", "Hot Pink", "Forest Green", "Ultramarine"],
    34: ["Sunflower", "Rust", "Fuchsia", "Hunter Green", "Sapphire"],
    35: ["Golden Yellow", "Terracotta", "Magenta", "Emerald", "Navy"],
    36: ["Marigold", "Clay", "Pink", "Teal", "Indigo"],
    37: ["Amber", "Brick", "Rose", "Jade", "Deep Blue"],
    38: ["Yellow", "Electric Blue", "Crimson", "Orange", "White"],
    39: ["Lemon", "Azure", "Scarlet", "Tangerine", "Ivory"],
    40: ["Bright Yellow", "Cerulean", "Ruby", "Burnt Orange", "Cream"],
    41: ["Gold", "Emerald", "Magenta", "Orange", "White"],
    42: ["Emerald", "Sapphire", "Ruby", "Amethyst", "Silver"],
    43: ["Jade", "Turquoise", "Garnet", "Topaz", "Pearl"],
    44: ["Forest Green", "Royal Blue", "Crimson", "Amber", "Platinum"],
    45: ["Kelly Green", "Cobalt", "Scarlet", "Gold", "Diamond White"],
    46: ["Electric Blue", "Hot Pink", "Lime", "Gold", "Pure White"],
    47: ["Neon Green", "Fuchsia", "Yellow", "Rose Gold", "Bright White"],
    48: ["Bright Turquoise", "Magenta", "Canary", "Copper", "Snow White"],
    49: ["White", "Neon Pink", "Gold", "Fuchsia", "Turquoise"],
    50: ["Pure White", "Electric Pink", "Bright Gold", "Neon Green", "Silver"]
}

color_hex_map = {
    "Lavender": "#E6E6FA", "Baby Blue": "#89CFF0", "Mint": "#98FF98", "Soft Pink": "#FFB6C1", "Light Grey": "#D3D3D3",
    "Emerald": "#50C878", "Sapphire": "#0F52BA", "Ruby": "#E0115F", "Navy Blue": "#000080", "Burgundy": "#800020",
    "Coral": "#FF7F50", "Teal": "#008080", "Aqua": "#00FFFF", "Dusty Rose": "#DCAE96", "Olive Green": "#808000",
    "Terracotta": "#E2725B", "Burnt Orange": "#CC5500", "Mustard": "#FFDB58", "Copper": "#B87333", "Bright Blue": "#0096FF",
    "Turquoise": "#40E0D0", "Deep Purple": "#9B30FF", "White": "#FFFFFF", "Black": "#000000", "Red": "#FF0000",
    "Cobalt Blue": "#0047AB", "Magenta": "#FF00FF", "Gold": "#FFD700", "Bronze": "#CD7F32", "Bright Green": "#66FF00",
    "Royal Blue": "#4169E1", "Yellow": "#FFFF00", "Electric Blue": "#7DF9FF", "Crimson": "#DC143C", "Orange": "#FFA500",
    "Amethyst": "#9966CC", "Silver": "#C0C0C0", "Bright Red": "#FF0000", "Neon Pink": "#FF6EC7", "Fuchsia": "#FF00FF",
    "Powder Blue": "#B0E0E6", "Peach": "#FFE5B4", "Rose Pink": "#FF66CC", "Light Lavender": "#E6E6FA", "Cream": "#FFFDD0",
    "Soft Coral": "#F88379", "Periwinkle": "#CCCCFF", "Blush Pink": "#FE828C", "Sage Green": "#9DC183", "Ivory": "#FFFFF0",
    "Light Peach": "#FFE5CC", "Sky Blue": "#87CEEB", "Rose": "#FF007F", "Lilac": "#C8A2C8", "Champagne": "#F7E7CE",
    "Soft Yellow": "#FFFF99", "Seafoam": "#93E9BE", "Vanilla": "#F3E5AB", "Light Coral": "#F08080", "Cornflower Blue": "#6495ED",
    "Mauve": "#E0B0FF", "Apricot": "#FBCEB1", "Cerulean": "#007BA7", "Pink": "#FFC0CB", "Beige": "#F5F5DC",
    "Azure": "#007FFF", "Salmon": "#FA8072", "Taupe": "#483C32", "Sand": "#C2B280", "Melon": "#FEBAAD",
    "Camel": "#C19A6B", "Rose Gold": "#B76E79", "Tan": "#D2B48C", "Rust": "#B7410E", "Jade": "#00A86B",
    "Cobalt": "#0047AB", "Caramel": "#C68E17", "Brick Red": "#CB4154", "Mocha": "#967969", "Burnt Sienna": "#E97451",
    "Aquamarine": "#7FFFD4", "Indigo": "#4B0082", "Cognac": "#9A463D", "Cinnamon": "#D2691E", "Mango": "#FDBE02",
    "Chestnut": "#954535", "Paprika": "#8B2500", "Walnut": "#773F1A", "Clay": "#B66A50", "Tangerine": "#F28500",
    "Cantaloupe": "#FFA500", "Peacock Blue": "#005F73", "Violet": "#8F00FF", "Chocolate": "#7B3F00", "Scarlet": "#FF2400",
    "Plum": "#8E4585", "Espresso": "#4E312D", "Eggplant": "#614051", "Coffee": "#6F4E37", "Hot Pink": "#FF69B4",
    "Cyan": "#00FFFF", "Flame Orange": "#FFA500", "Mahogany": "#C04000", "Cerise": "#DE3163", "Umber": "#635147",
    "Sepia": "#704214", "Lime": "#00FF00", "Canary Yellow": "#FFEF00", "Ultramarine": "#120A8F", "Sunflower": "#FFDA03",
    "Hunter Green": "#355E3B", "Forest Green": "#228B22", "Marigold": "#EAA221", "Kelly Green": "#4CBB17", "Garnet": "#733635",
    "Topaz": "#FFC87C", "Pearl": "#EAE0C8", "Platinum": "#E5E4E2", "Diamond White": "#F0EFF4", "Neon Green": "#39FF14",
    "Electric Pink": "#F535AA", "Lemon": "#FFF700", "Amber": "#FFBF00", "Pure White": "#FFFFFF", "Bright White": "#FAFAFA",
    "Snow White": "#FFFAFA", "Bright Gold": "#FFD700", "Bright Turquoise": "#08E8DE"
}

# Professional SVG Icons
SVG_ICONS = {
    "upload": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>',
    "swatch": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
    "sparkles": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 1.912 5.813a2 2 0 0 0 1.275 1.275L21 12l-5.813 1.912a2 2 0 0 0-1.275 1.275L12 21l-1.912-5.813a2 2 0 0 0-1.275-1.275L3 12l5.813-1.912a2 2 0 0 0 1.275-1.275L12 3z"></path></svg>',
    "info": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
}

# ═══════════════════════════════════════════════════════════
#                CORE LOGIC FUNCTIONS
# ═══════════════════════════════════════════════════════════

def detect_face_and_skin_tone(image):
    try:
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cascade_path = 'haarcascade_frontalface_default.xml'

        if not os.path.exists(cascade_path):
            url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
            urllib.request.urlretrieve(url, cascade_path)

        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

        if len(faces) == 0:
            return None, None, False, "No face detected! Please upload a clear photo."

        x, y, w, h = faces[0]
        face_region = img_bgr[y:y+h, x:x+w]
        face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        avg_color = cv2.resize(face_region_rgb, (1, 1)).flatten()

        mst_score = min(skin_tones.keys(), key=lambda k: np.linalg.norm(avg_color - np.array(skin_tones[k])))
        return mst_score, skin_tones[mst_score], True, None
    except Exception as e:
        return None, None, False, f"Error: {str(e)}"

def create_color_swatch(rgb_tuple, size=(100, 100)):
    swatch = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    swatch[:, :] = rgb_tuple
    return swatch

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def generate_report(uploaded_file, mst_score, skin_rgb, recommendations):
    report = f"SKIN TONE ANALYSIS REPORT\n{'='*30}\n"
    report += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    report += f"MST Score: {mst_score}/50\n"
    report += f"Hex Code: {rgb_to_hex(skin_rgb)}\n\n"
    report += "RECOMMENDED PALETTE:\n"
    for color in recommendations:
        report += f"- {color} ({color_hex_map.get(color)})\n"
    return report

# ═══════════════════════════════════════════════════════════
#                STREAMLIT UI LAYOUT
# ═══════════════════════════════════════════════════════════

st.set_page_config(page_title="Skin Tone Analyzer Pro", page_icon="🎭", layout="wide")

# Custom CSS for SaaS-style UI
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    * {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
    
    .st_sk_hero_title {{
        text-align: center; font-size: 3.5rem; font-weight: 800;
        background: linear-gradient(135deg, #FF007F, #7000FF, #00EAFF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px; letter-spacing: -2px;
    }}
    .st_sk_hero_subtitle {{
        text-align: center; font-size: 1.1rem; color: #64748b; margin-bottom: 3rem;
    }}
    .st_sk_glass_card {{
        background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 24px;
        padding: 2rem; box-shadow: 0 15px 35px rgba(0,0,0,0.05); margin-bottom: 1.5rem;
    }}
    .st_sk_icon_header {{
        display: flex; align-items: center; gap: 12px; font-size: 1.4rem; font-weight: 700; color: #1e293b; margin-bottom: 1.2rem;
    }}
    .st_sk_icon_svg {{ color: #7000FF; display: flex; align-items: center; justify-content: center; }}
    
    .st_sk_palette_card {{
        background: white; border-radius: 16px; overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #f1f5f9; transition: 0.3s;
    }}
    .st_sk_palette_card:hover {{ transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,0,0,0.1); }}
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {{ color: #FF007F; font-weight: 800; }}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="st_sk_hero_title">Skin Tone Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="st_sk_hero_subtitle">Professional-grade color mapping via Monk Skin Tone Baseline.</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f'<div class="st_sk_glass_card"><div class="st_sk_icon_header"><span class="st_sk_icon_svg">{SVG_ICONS["sparkles"]}</span> Protocol</div>Precision analysis for inclusive design and personal styling.</div>', unsafe_allow_html=True)
    st.markdown("### 💡 Guidance")
    st.info("Use natural light and avoid harsh shadows for optimal metric sampling.")

# Main Layout
col_main, col_preview = st.columns([3, 2], gap="large")

with col_main:
    st.markdown(f'<div class="st_sk_icon_header"><span class="st_sk_icon_svg">{SVG_ICONS["upload"]}</span> Image Dropzone</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Profile", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        with st.spinner("Processing facial geometry..."):
            mst_score, skin_rgb, face_detected, error_message = detect_face_and_skin_tone(image)

        if face_detected:
            st.markdown('<div class="st_sk_glass_card">', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Calculated MST Score", f"{mst_score} / 50")
            m2.image(create_color_swatch(skin_rgb, (100, 100)), caption=f"Sampled: {rgb_to_hex(skin_rgb)}", width=100)
            st.markdown('</div>', unsafe_allow_html=True)

            # Palette Results
            st.markdown(f'<div class="st_sk_icon_header"><span class="st_sk_icon_svg">{SVG_ICONS["swatch"]}</span> Recommended Palette</div>', unsafe_allow_html=True)
            recs = mst_recommendations[mst_score]
            p_cols = st.columns(5)
            for idx, color_name in enumerate(recs):
                hex_c = color_hex_map.get(color_name, "#CCCCCC")
                with p_cols[idx]:
                    st.markdown(f"""
                        <div class="st_sk_palette_card">
                            <div style="background:{hex_c}; height:60px;"></div>
                            <div style="padding:8px; text-align:center; font-size:0.8rem; font-weight:600;">{color_name}</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.error(error_message)

with col_preview:
    st.markdown(f'<div class="st_sk_icon_header"><span class="st_sk_icon_svg">{SVG_ICONS["info"]}</span> Frame Output</div>', unsafe_allow_html=True)
    if uploaded_file:
        st.image(image, use_container_width=True, caption="Source Scan")
        report = generate_report(uploaded_file, mst_score, skin_rgb, recs)
        st.download_button("Download Diagnostic Report", report, f"mst_report_{mst_score}.txt", use_container_width=True)
    else:
        st.info("Upload an image to start the analysis pipeline.")

# Full Scale Preview
if not uploaded_file:
    st.markdown("---")
    st.markdown(f'<div class="st_sk_icon_header"><span class="st_sk_icon_svg">{SVG_ICONS["swatch"]}</span> MST Baseline (1-50)</div>', unsafe_allow_html=True)
    grid_cols = st.columns(10)
    for i in range(50):
        t_id = i + 1
        with grid_cols[i % 10]:
            h_code = rgb_to_hex(skin_tones[t_id])
            st.markdown(f'<div style="background:{h_code}; height:30px; border-radius:4px; margin-bottom:2px;"></div><div style="font-size:0.6rem; text-align:center;">P-{t_id}</div>', unsafe_allow_html=True)

# Footer
st.markdown(f"""
    <div style="text-align: center; margin-top: 5rem; padding-bottom: 2rem; border-top: 1px solid #f1f5f9;">
        <p style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; padding-top: 2rem;">
            SYSTEM ARCHITECT: MAHFUZA LASKAR | APPBRIDGE LAB INTEGRATION
        </p>
    </div>
""", unsafe_allow_html=True)