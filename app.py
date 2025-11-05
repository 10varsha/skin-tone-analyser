import inspect
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime

# MST skin tones (RGB tuples)
skin_tones = {
    1: (255, 224, 220),
    2: (255, 205, 190),
    3: (240, 180, 150),      
    4: (220, 160, 130),
    5: (200, 140, 110),
    6: (180, 120, 90),
    7: (150, 90, 60),
    8: (120, 70, 50),
    9: (90, 50, 40),
    10: (60, 30, 20)
}

mst_recommendations = {
    1: ["Lavender", "Baby Blue", "Mint", "Soft Pink", "Light Grey"],
    2: ["Emerald", "Sapphire", "Ruby", "Navy Blue", "Burgundy"],
    3: ["Coral", "Teal", "Aqua", "Dusty Rose", "Olive Green"],
    4: ["Terracotta", "Burnt Orange", "Mustard", "Copper"],
    5: ["Bright Blue", "Emerald", "Turquoise", "Deep Purple", "Coral"],
    6: ["White", "Black", "Red", "Cobalt Blue", "Magenta"],
    7: ["Gold", "Bronze", "Mustard", "Bright Green", "Royal Blue"],
    8: ["Yellow", "Electric Blue", "Crimson", "Orange", "White"],
    9: ["Emerald", "Sapphire", "Ruby", "Amethyst", "Silver", "Bright Red"],
    10: ["White", "Neon Pink", "Gold", "Fuchsia", "Turquoise"]
}

# Color hex codes for visualization
color_hex_map = {
    "Lavender": "#E6E6FA", "Baby Blue": "#89CFF0", "Mint": "#98FF98",
    "Soft Pink": "#FFB6C1", "Light Grey": "#D3D3D3", "Emerald": "#50C878",
    "Sapphire": "#0F52BA", "Ruby": "#E0115F", "Navy Blue": "#000080",
    "Burgundy": "#800020", "Coral": "#FF7F50", "Teal": "#008080",
    "Aqua": "#00FFFF", "Dusty Rose": "#DCAE96", "Olive Green": "#808000",
    "Terracotta": "#E2725B", "Burnt Orange": "#CC5500", "Mustard": "#FFDB58",
    "Copper": "#B87333", "Bright Blue": "#0096FF", "Turquoise": "#40E0D0",
    "Deep Purple": "#9B30FF", "White": "#FFFFFF", "Black": "#000000",
    "Red": "#FF0000", "Cobalt Blue": "#0047AB", "Magenta": "#FF00FF",
    "Gold": "#FFD700", "Bronze": "#CD7F32", "Bright Green": "#66FF00",
    "Royal Blue": "#4169E1", "Yellow": "#FFFF00", "Electric Blue": "#7DF9FF",
    "Crimson": "#DC143C", "Orange": "#FFA500", "Amethyst": "#9966CC",
    "Silver": "#C0C0C0", "Bright Red": "#FF0000", "Neon Pink": "#FF6EC7",
    "Fuchsia": "#FF00FF"
}

# Determine the compatible Streamlit keyword for full-width images
_image_width_kwarg = (
    "use_container_width"
    if "use_container_width" in inspect.signature(st.image).parameters
    else "use_column_width"
)


def render_full_width_image(image, caption=None):
    """Render an image that adapts to the container width across Streamlit versions."""
    st.image(image, caption=caption, **{_image_width_kwarg: True})

def detect_face_and_skin_tone(image):
    """
    Detect face in image and calculate skin tone using Haar Cascade
    Returns: mst_score, skin_rgb, face_detected, error_message
    """
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Load Haar Cascade face detector
        import os
        cascade_path = 'haarcascade_frontalface_default.xml'

        if not os.path.exists(cascade_path):
            import urllib.request
            url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
            urllib.request.urlretrieve(url, cascade_path)

        face_cascade = cv2.CascadeClassifier(cascade_path)

        # Verify cascade loaded correctly
        if face_cascade.empty():
            raise Exception("Error loading Haar Cascade classifier")

        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )
        
        # Error handling: No face detected
        if len(faces) == 0:
            return None, None, False, "No face detected! Please upload a clear photo with your face visible."
        
        # Use the first (largest) face detected
        x, y, w, h = faces[0]
        
        # Extract face region from the image
        face_region = img_bgr[y:y+h, x:x+w]
        face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        
        # Resize face for easier processing
        resized = cv2.resize(face_region_rgb, (200, 200))
        
        # Calculate average color of the face
        avg_color = resized.mean(axis=0).mean(axis=0)
        
        # Find closest MST using Euclidean distance
        mst_score = min(
            skin_tones.keys(),
            key=lambda k: np.linalg.norm(avg_color - np.array(skin_tones[k]))
        )
        
        return mst_score, skin_tones[mst_score], True, None
        
    except Exception as e:
        return None, None, False, f"Error processing image: {str(e)}"

def create_color_swatch(rgb_tuple, size=(100, 100)):
    """Create a color swatch image from RGB values"""
    swatch = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    swatch[:, :] = rgb_tuple
    return swatch

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color code"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def generate_report(uploaded_file, mst_score, skin_rgb, recommendations):
    """Generate downloadable text report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║          SKIN TONE ANALYSIS REPORT                       ║
╚══════════════════════════════════════════════════════════╝

Image: {uploaded_file.name}
Analysis Date: {datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")}

┌─ SKIN TONE ANALYSIS ────────────────────────────────────┐
│ Monk Skin Tone (MST) Score: {mst_score}/10
│ Detected RGB Values: {skin_rgb}
│ Hex Color Code: {rgb_to_hex(skin_rgb)}
└─────────────────────────────────────────────────────────┘

┌─ YOUR RECOMMENDED COLOR PALETTE ────────────────────────┐
"""
    for i, color in enumerate(recommendations, 1):
        hex_code = color_hex_map.get(color, "#000000")
        report += f"│ {i}. {color:<20} {hex_code}\n"
    
    report += """└─────────────────────────────────────────────────────────┘

💡 STYLE TIPS:
• These colors complement your natural skin tone beautifully
• Mix and match within your color palette for best results
• Consider the occasion and lighting when choosing colors
• Don't be afraid to experiment with different shades!

✨ ABOUT MST SCALE:
The Monk Skin Tone (MST) scale is a 10-shade scale designed to
represent a diverse range of skin tones for better inclusivity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated by Skin Tone Analyzer
Visit: https://github.com/yourusername/skin-tone-analyzer
"""
    return report

# ═══════════════════════════════════════════════════════════
#                    STREAMLIT UI
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Skin Tone Analyzer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for candy theme
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF69B4, #87CEEB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #FF69B4;
        margin-bottom: 2rem;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #FFE5F0 0%, #E0F4FF 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(255, 105, 180, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Results section */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(135, 206, 235, 0.1);
        border: 2px solid #FFE5F0;
    }
    
    /* Color swatch container */
    .color-swatch {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 2px solid #FFE5F0;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #FF69B4;
        font-size: 2rem;
    }
    
    /* Buttons */
    .stDownloadButton button {
        background: linear-gradient(90deg, #FF69B4, #87CEEB);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #E0F4FF;
        border-left: 4px solid #87CEEB;
    }
    
    .stError {
        background-color: #FFE5F0;
        border-left: 4px solid #FF69B4;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🎨 Skin Tone Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover your perfect color palette in seconds ✨</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📖 About")
    st.info("""
    **Skin Tone Analyzer** uses the Monk Skin Tone (MST) scale to determine 
    your skin tone and recommend colors that complement your natural complexion.
    """)
    
    st.markdown("### 🎯 How It Works")
    st.markdown("""
    1. **Upload** a clear photo of your face
    2. **Face Detection** automatically finds your face
    3. **Analysis** calculates your MST score (1-10)
    4. **Results** shows your perfect color palette
    5. **Download** your personalized report
    """)
    
    st.markdown("### 💡 Tips for Best Results")
    st.success("""
    ✓ Use natural lighting  
    ✓ Face clearly visible  
    ✓ Avoid heavy makeup  
    ✓ No filters or effects  
    ✓ Formats: JPG, PNG, JPEG
    """)
    
    st.markdown("### 📊 MST Scale")
    st.markdown("""
    The scale ranges from **1 (lightest)** to **10 (darkest)**, 
    designed to represent diverse skin tones inclusively.
    """)

# Main content area
st.markdown("---")

# Image upload section
st.markdown("### 📸 Upload Your Photo")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of your face",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        # Load and validate image
        image = Image.open(uploaded_file)
        image = image.convert('RGB')
        
        # Create two columns for layout
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("#### 📷 Your Photo")
            st.image(image, caption="Uploaded Image")
        
        # Process image in real-time
        with st.spinner("🔍 Analyzing your skin tone..."):
            mst_score, skin_rgb, face_detected, error_message = detect_face_and_skin_tone(image)
        
        # Handle errors
        if not face_detected:
            st.error(f"❌ {error_message}")
            st.info("💡 **Tip:** Make sure your face is clearly visible and well-lit. Try a different photo!")
        
        else:
            # Display results in second column
            with col2:
                st.markdown("#### 🎯 Your Results")
                
                # MST Score with custom styling
                st.metric(
                    label="Monk Skin Tone Score",
                    value=f"{mst_score} / 10",
                    delta="Detected"
                )
                
                # Detected skin tone swatch
                st.markdown("**Your Skin Tone:**")
                swatch = create_color_swatch(skin_rgb, (120, 120))
                st.image(swatch, width=120)
                
                # RGB and Hex values
                hex_color = rgb_to_hex(skin_rgb)
                st.caption(f"🎨 RGB: {skin_rgb}")
                st.caption(f"🎨 Hex: {hex_color}")
            
            # Color recommendations section
            st.markdown("---")
            st.markdown("### ✨ Your Perfect Color Palette")
            
            recommendations = mst_recommendations[mst_score]
            
            # Display recommended colors as swatches
            cols = st.columns(5)
            for idx, color in enumerate(recommendations):
                with cols[idx % 5]:
                    hex_color = color_hex_map.get(color, "#CCCCCC")
                    
                    # Convert hex to RGB
                    hex_color_clean = hex_color.lstrip('#')
                    rgb = tuple(int(hex_color_clean[i:i+2], 16) for i in (0, 2, 4))
                    
                    # Create swatch
                    color_swatch = create_color_swatch(rgb, (100, 100))
                    render_full_width_image(color_swatch)
                    st.markdown(f"**{color}**")
                    st.caption(hex_color)
            
            # Download section
            st.markdown("---")
            st.markdown("### 📥 Download Your Report")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            
            with col_btn1:
                report_text = generate_report(uploaded_file, mst_score, skin_rgb, recommendations)
                st.download_button(
                    label="📄 Download Report",
                    data=report_text,
                    file_name=f"skin_tone_report_MST{mst_score}.txt",
                    mime="text/plain"
                )
            
            # Success message
            st.success("✅ **Analysis Complete!** Your personalized color recommendations are ready.")
            
            # Additional info
            with st.expander("💡 Learn More About Your Results"):
                st.markdown(f"""
                **Your MST Score: {mst_score}/10**
                
                This score represents where your skin tone falls on the Monk Skin Tone scale, 
                a diverse and inclusive scale created to better represent all skin tones.
                
                **Why These Colors?**
                
                The recommended colors are scientifically chosen to complement your specific 
                skin tone based on color theory and contrast principles. These colors will:
                - Make your skin glow naturally
                - Enhance your overall appearance
                - Work well in various settings and lighting conditions
                
                **How to Use:**
                - Try incorporating these colors into your wardrobe
                - Use them as reference for makeup selection
                - Consider them for home decor or personal branding
                """)
    
    except Exception as e:
        st.error(f"❌ **Error loading image:** {str(e)}")
        st.info("Please upload a valid image file (PNG, JPG, or JPEG format).")

else:
    # Landing state - show example
    st.info("👆 **Upload a photo above to get started!**")
    
    st.markdown("---")
    st.markdown("### 📊 The MST Scale")
    st.caption("Here's what the 10 skin tones look like:")
    
    cols = st.columns(5)
    for i in range(10):
        with cols[i % 5]:
            mst_num = i + 1
            swatch = create_color_swatch(skin_tones[mst_num], (80, 80))
            render_full_width_image(swatch)
            st.caption(f"MST {mst_num}", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <p style='color: #FF69B4; font-size: 0.9rem;'>
            Made with 💖 by Your Name | Powered by OpenCV & Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)
