import inspect
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime
import streamlit.components.v1 as components  # Add this at the top

# MST skin tones (RGB tuples)
# 50-tone skin scale (RGB tuples) - Fair to Deep Dark
skin_tones = {
    # Very Fair (1-10)
    1: (255, 235, 230),
    2: (255, 228, 225),
    3: (255, 224, 220),
    4: (255, 220, 210),
    5: (255, 215, 205),
    6: (255, 210, 200),
    7: (255, 205, 195),
    8: (255, 205, 190),
    9: (250, 200, 185),
    10: (245, 195, 180),
    
    # Fair (11-20)
    11: (240, 190, 175),
    12: (240, 185, 170),
    13: (240, 180, 165),
    14: (240, 180, 150),
    15: (235, 175, 145),
    16: (230, 170, 140),
    17: (225, 165, 135),
    18: (220, 160, 130),
    19: (215, 155, 125),
    20: (210, 150, 120),
    
    # Light Medium (21-30)
    21: (205, 145, 115),
    22: (200, 140, 110),
    23: (195, 135, 105),
    24: (190, 130, 100),
    25: (185, 125, 95),
    26: (180, 120, 90),
    27: (175, 115, 85),
    28: (170, 110, 80),
    29: (165, 105, 75),
    30: (160, 100, 70),
    
    # Medium (31-40)
    31: (155, 95, 65),
    32: (150, 90, 60),
    33: (145, 85, 58),
    34: (140, 82, 56),
    35: (135, 80, 54),
    36: (130, 75, 52),
    37: (125, 72, 50),
    38: (120, 70, 50),
    39: (115, 67, 48),
    40: (110, 65, 46),
    
    # Deep Medium (41-45)
    41: (105, 60, 44),
    42: (100, 58, 42),
    43: (95, 55, 40),
    44: (90, 50, 40),
    45: (85, 48, 38),
    
    # Deep Dark (46-50)
    46: (80, 45, 36),
    47: (70, 40, 32),
    48: (65, 35, 28),
    49: (60, 30, 25),
    50: (50, 25, 20)
}

# Color recommendations based on 50-tone scale
mst_recommendations = {
    # Very Fair (1-10) - Cool undertones work best
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
    
    # Fair (11-20) - Warm and jewel tones
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
    
    # Light Medium (21-30) - Rich, vibrant colors
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
    
    # Medium (31-40) - Bold, saturated colors
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
    
    # Deep Medium (41-45) - Jewel tones and metallics
    41: ["Gold", "Emerald", "Magenta", "Orange", "White"],
    42: ["Emerald", "Sapphire", "Ruby", "Amethyst", "Silver"],
    43: ["Jade", "Turquoise", "Garnet", "Topaz", "Pearl"],
    44: ["Forest Green", "Royal Blue", "Crimson", "Amber", "Platinum"],
    45: ["Kelly Green", "Cobalt", "Scarlet", "Gold", "Diamond White"],
    
    # Deep Dark (46-50) - Bright, electric colors and metallics
    46: ["Electric Blue", "Hot Pink", "Lime", "Gold", "Pure White"],
    47: ["Neon Green", "Fuchsia", "Yellow", "Rose Gold", "Bright White"],
    48: ["Bright Turquoise", "Magenta", "Canary", "Copper", "Snow White"],
    49: ["White", "Neon Pink", "Gold", "Fuchsia", "Turquoise"],
    50: ["Pure White", "Electric Pink", "Bright Gold", "Neon Green", "Silver"]
}

# Colors to avoid based on 50-tone scale
colors_to_avoid = {
    # Very Fair (1-10) – Avoid harsh, overly warm, and very dark tones
    1: ["Black", "Charcoal", "Mustard", "Rust", "Neon Pink"],
    2: ["Jet Black", "Burnt Orange", "Olive", "Hot Pink", "Dark Brown"],
    3: ["Deep Red", "Mustard Yellow", "Chocolate", "Neon Green", "Rust"],
    4: ["Black", "Copper", "Dark Olive", "Bright Orange", "Neon Yellow"],
    5: ["Charcoal", "Rust", "Deep Mustard", "Hot Pink", "Dark Green"],
    6: ["Black", "Burnt Sienna", "Neon Orange", "Olive Brown", "Deep Purple"],
    7: ["Dark Brown", "Rust", "Mustard", "Neon Pink", "Black"],
    8: ["Charcoal", "Olive", "Burnt Orange", "Hot Red", "Neon Green"],
    9: ["Black", "Deep Mustard", "Rust", "Dark Olive", "Electric Pink"],
    10: ["Charcoal", "Burnt Orange", "Mustard", "Deep Brown", "Neon Yellow"],
    
    # Fair (11-20) – Avoid washed-out, overly cool, and muddy tones
    11: ["Baby Pink", "Ice Blue", "Pale Yellow", "Grey Beige", "Ash Grey"],
    12: ["Powder Pink", "Frost Blue", "Light Lavender", "Dull Beige", "Cool Grey"],
    13: ["Pale Mint", "Icy Blue", "Dusty Grey", "Light Taupe", "Faded Pink"],
    14: ["Baby Blue", "Soft Lilac", "Pale Peach", "Ash Brown", "Muted Grey"],
    15: ["Light Yellow", "Powder Blue", "Cool Lavender", "Grey", "Washed Beige"],
    16: ["Pastel Pink", "Ice Mint", "Faded Blue", "Dull Brown", "Soft Grey"],
    17: ["Pale Coral", "Baby Blue", "Light Grey", "Dusty Lavender", "Ash Beige"],
    18: ["Powder Yellow", "Icy Pink", "Faded Purple", "Cool Grey", "Muted Beige"],
    19: ["Light Mint", "Baby Pink", "Ash Brown", "Pale Grey", "Frost Blue"],
    20: ["Soft Yellow", "Ice Blue", "Light Taupe", "Dusty Grey", "Faded Peach"],
    
    # Light Medium (21-30) – Avoid faded, dusty, and weak tones
    21: ["Light Beige", "Dusty Pink", "Pale Yellow", "Soft Grey", "Faded Blue"],
    22: ["Muted Lavender", "Ash Grey", "Light Cream", "Dusty Peach", "Faded Mint"],
    23: ["Pale Blue", "Washed Pink", "Light Taupe", "Soft Beige", "Dusty Lilac"],
    24: ["Faded Coral", "Muted Purple", "Light Grey", "Ash Brown", "Pale Mint"],
    25: ["Soft Yellow", "Dusty Rose", "Light Beige", "Faded Blue", "Grey"],
    26: ["Off White", "Cream", "Light Grey", "Beige", "Dusty Pink"],
    27: ["Muted Coral", "Faded Purple", "Soft Brown", "Light Grey", "Dusty Blue"],
    28: ["Pale Orange", "Ash Grey", "Light Lavender", "Faded Pink", "Soft Beige"],
    29: ["Muted Red", "Dusty Blue", "Light Brown", "Soft Grey", "Faded Purple"],
    30: ["Pale Peach", "Light Taupe", "Dusty Violet", "Grey Beige", "Faded Coral"],
    
    # Medium (31-40) – Avoid pastels and overly cool/light tones
    31: ["Baby Pink", "Powder Blue", "Mint", "Light Lavender", "Pale Yellow"],
    32: ["Soft Peach", "Ice Blue", "Pastel Green", "Light Lilac", "Cream"],
    33: ["Powder Pink", "Baby Blue", "Soft Grey", "Light Mint", "Pale Beige"],
    34: ["Pastel Yellow", "Ice Mint", "Light Blue", "Soft Pink", "Ash Grey"],
    35: ["Baby Pink", "Powder Blue", "Light Lavender", "Cream", "Soft Yellow"],
    36: ["Soft Coral", "Ice Blue", "Light Mint", "Pastel Purple", "Beige"],
    37: ["Light Peach", "Powder Blue", "Soft Pink", "Pale Yellow", "Grey"],
    38: ["Pastel Blue", "Light Lavender", "Soft Coral", "Mint", "Cream"],
    39: ["Baby Yellow", "Ice Blue", "Soft Pink", "Light Grey", "Beige"],
    40: ["Powder Pink", "Light Mint", "Pastel Blue", "Soft Yellow", "Ash Grey"],
    
    # Deep Medium (41-45) – Avoid muted, dusty, and pale tones
    41: ["Dusty Pink", "Muted Peach", "Light Beige", "Soft Yellow", "Pale Grey"],
    42: ["Pastel Blue", "Light Lavender", "Soft Peach", "Beige", "Ash Grey"],
    43: ["Dusty Rose", "Muted Coral", "Light Taupe", "Soft Yellow", "Pale Mint"],
    44: ["Pastel Pink", "Light Blue", "Soft Grey", "Beige", "Faded Yellow"],
    45: ["Muted Peach", "Dusty Lavender", "Light Brown", "Soft Beige", "Pale Grey"],
    
    # Deep Dark (46-50) – Avoid dull, muted, and low-contrast tones
    46: ["Olive Brown", "Mud Grey", "Dusty Blue", "Muted Pink", "Faded Green"],
    47: ["Dull Beige", "Muted Lavender", "Ash Grey", "Soft Brown", "Faded Yellow"],
    48: ["Olive Drab", "Dusty Coral", "Muted Blue", "Grey Brown", "Soft Peach"],
    49: ["Mud Brown", "Dusty Pink", "Muted Grey", "Faded Blue", "Soft Beige"],
    50: ["Dull Olive", "Ash Brown", "Muted Purple", "Dusty Yellow", "Grey Beige"]
}

# Expanded color hex codes for visualization
color_hex_map = {
    # Existing colors
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
    "Fuchsia": "#FF00FF",
    
    # New colors for 50-tone scale
    "Powder Blue": "#B0E0E6", "Peach": "#FFE5B4", "Rose Pink": "#FF66CC",
    "Light Lavender": "#E6E6FA", "Cream": "#FFFDD0", "Soft Coral": "#F88379",
    "Periwinkle": "#CCCCFF", "Blush Pink": "#FE828C", "Sage Green": "#9DC183",
    "Ivory": "#FFFFF0", "Light Peach": "#FFE5CC", "Sky Blue": "#87CEEB",
    "Rose": "#FF007F", "Lilac": "#C8A2C8", "Champagne": "#F7E7CE",
    "Soft Yellow": "#FFFF99", "Seafoam": "#93E9BE", "Vanilla": "#F3E5AB",
    "Light Coral": "#F08080", "Cornflower Blue": "#6495ED", "Mauve": "#E0B0FF",
    "Apricot": "#FBCEB1", "Cerulean": "#007BA7", "Pink": "#FFC0CB",
    "Beige": "#F5F5DC", "Azure": "#007FFF", "Salmon": "#FA8072",
    "Taupe": "#483C32", "Sand": "#C2B280", "Melon": "#FEBAAD",
    "Camel": "#C19A6B", "Rose Gold": "#B76E79", "Tan": "#D2B48C",
    "Rust": "#B7410E", "Jade": "#00A86B", "Cobalt": "#0047AB",
    "Caramel": "#C68E17", "Brick Red": "#CB4154", "Mocha": "#967969",
    "Burnt Sienna": "#E97451", "Aquamarine": "#7FFFD4", "Indigo": "#4B0082",
    "Cognac": "#9A463D", "Cinnamon": "#D2691E", "Mango": "#FDBE02",
    "Chestnut": "#954535", "Paprika": "#8B2500", "Walnut": "#773F1A",
    "Clay": "#B66A50", "Tangerine": "#F28500", "Cantaloupe": "#FFA500",
    "Peacock Blue": "#005F73", "Violet": "#8F00FF", "Chocolate": "#7B3F00",
    "Scarlet": "#FF2400", "Plum": "#8E4585", "Espresso": "#4E312D",
    "Eggplant": "#614051", "Coffee": "#6F4E37", "Hot Pink": "#FF69B4",
    "Cyan": "#00FFFF", "Flame Orange": "#FFA500", "Mahogany": "#C04000",
    "Cerise": "#DE3163", "Umber": "#635147", "Sepia": "#704214",
    "Lime": "#00FF00", "Canary Yellow": "#FFEF00", "Ultramarine": "#120A8F",
    "Sunflower": "#FFDA03", "Hunter Green": "#355E3B", "Forest Green": "#228B22",
    "Marigold": "#EAA221", "Kelly Green": "#4CBB17", "Garnet": "#733635",
    "Topaz": "#FFC87C", "Pearl": "#EAE0C8", "Platinum": "#E5E4E2",
    "Diamond White": "#F0EFF4", "Neon Green": "#39FF14", "Electric Pink": "#F535AA",
    "Lemon": "#FFF700", "Amber": "#FFBF00", "Pure White": "#FFFFFF",
    "Bright White": "#FAFAFA", "Snow White": "#FFFAFA", "Bright Gold": "#FFD700",
    "Bright Turquoise": "#08E8DE",
    
    # Colors to avoid - additional hex codes
    "Jet Black": "#0A0A0A", "Mustard Yellow": "#E1AD21", "Dark Olive": "#556B2F",
    "Bright Orange": "#FFA500", "Neon Yellow": "#FFFF00", "Deep Mustard": "#DAA520",
    "Dark Green": "#006400", "Olive Brown": "#6B4423", "Hot Red": "#FF0000",
    "Deep Brown": "#5C4033", "Ice Blue": "#A5F2F3", "Pale Yellow": "#FFFFE0",
    "Grey Beige": "#D3D3D3", "Ash Grey": "#B2BEB5", "Frost Blue": "#A5F2F3",
    "Dull Beige": "#D3BDA8", "Cool Grey": "#8C92AC", "Pale Mint": "#98FB98",
    "Icy Blue": "#A5F2F3", "Dusty Grey": "#A9A9A9", "Light Taupe": "#B19CD9",
    "Soft Lilac": "#C8A2C8", "Pale Peach": "#FFE5B4", "Ash Brown": "#B2BEB5",
    "Muted Grey": "#696969", "Cool Lavender": "#B19CD9", "Washed Beige": "#F5F5DC",
    "Ice Mint": "#98FB98", "Faded Blue": "#ADD8E6", "Dull Brown": "#8B4513",
    "Pale Coral": "#FF7F50", "Dusty Lavender": "#B19CD9", "Ash Beige": "#F5F5DC",
    "Powder Yellow": "#FFFF99", "Icy Pink": "#FFB6C1", "Faded Purple": "#DDA0DD",
    "Light Mint": "#98FB98", "Frost Blue": "#A5F2F3", "Faded Peach": "#FFE5B4",
    "Light Beige": "#F5F5DC", "Faded Blue": "#ADD8E6", "Muted Lavender": "#B19CD9",
    "Light Cream": "#FFFDD0", "Dusty Peach": "#FFE5B4", "Faded Mint": "#98FB98",
    "Pale Blue": "#ADD8E6", "Washed Pink": "#FFB6C1", "Dusty Lilac": "#C8A2C8",
    "Faded Coral": "#FF7F50", "Muted Purple": "#9370DB", "Ash Brown": "#B2BEB5",
    "Dusty Rose": "#DCAE96", "Off White": "#FAF9F6", "Muted Coral": "#FF7F50",
    "Faded Purple": "#DDA0DD", "Soft Brown": "#D2691E", "Dusty Blue": "#ADD8E6",
    "Pale Orange": "#FFA500", "Muted Red": "#DC143C", "Light Brown": "#D2691E",
    "Pale Peach": "#FFE5B4", "Dusty Violet": "#DDA0DD", "Grey Beige": "#D3D3D3",
    "Pastel Green": "#77DD77", "Light Lilac": "#C8A2C8", "Pastel Yellow": "#FFFF99",
    "Ice Mint": "#98FB98", "Pastel Purple": "#DDA0DD", "Light Peach": "#FFE5B4",
    "Pastel Blue": "#ADD8E6", "Soft Coral": "#F88379", "Baby Yellow": "#FFFF99",
    "Muted Peach": "#FFE5B4", "Pale Grey": "#D3D3D3", "Dusty Rose": "#DCAE96",
    "Faded Yellow": "#FFFF99", "Dusty Lavender": "#B19CD9", "Olive Brown": "#6B4423",
    "Mud Grey": "#847C74", "Faded Green": "#90EE90", "Muted Lavender": "#B19CD9",
    "Soft Brown": "#D2691E", "Olive Drab": "#6B8E23", "Dusty Coral": "#FF7F50",
    "Grey Brown": "#8C7853", "Soft Peach": "#FFE5B4", "Mud Brown": "#8B4513",
    "Muted Grey": "#696969", "Faded Blue": "#ADD8E6", "Dull Olive": "#556B2F",
    "Ash Brown": "#B2BEB5", "Muted Purple": "#9370DB", "Dusty Yellow": "#FFFF99"
}

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

def generate_report(uploaded_file, mst_score, skin_rgb, recommendations, colors_to_avoid_list):
    """Generate downloadable text report"""
    report = f"""
╔══════════════════════════════════════════════════════════╗
║          SKIN TONE ANALYSIS REPORT                       ║
╚══════════════════════════════════════════════════════════╝

Image: {uploaded_file.name}
Analysis Date: {datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")}

┌─ SKIN TONE ANALYSIS ────────────────────────────────────┐
│ Monk Skin Tone (MST) Score: {mst_score}/50
│ Detected RGB Values: {skin_rgb}
│ Hex Color Code: {rgb_to_hex(skin_rgb)}
└─────────────────────────────────────────────────────────┘

┌─ YOUR RECOMMENDED COLOR PALETTE ────────────────────────┐
"""
    for i, color in enumerate(recommendations, 1):
        hex_code = color_hex_map.get(color, "#000000")
        report += f"│ {i}. {color:<20} {hex_code}\n"
    
    report += """└─────────────────────────────────────────────────────────┘

┌─ COLORS TO AVOID ───────────────────────────────────────┐
"""
    for i, color in enumerate(colors_to_avoid_list, 1):
        hex_code = color_hex_map.get(color, "#000000")
        report += f"│ {i}. {color:<20} {hex_code}\n"
    
    report += """└─────────────────────────────────────────────────────────┘

💡 STYLE TIPS:
• These colors complement your natural skin tone beautifully
• Mix and match within your color palette for best results
• Consider the occasion and lighting when choosing colors
• Don't be afraid to experiment with different shades!

✨ ABOUT MST SCALE:
The Monk Skin Tone (MST) scale is a 50-shade scale designed to
represent a diverse range of skin tones for better inclusivity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated by SKINLYTICS - Your Personal Skin Tone Analyzer
Visit: https://github.com/yourusername/skin-tone-analyzer
"""
    return report

def render_full_width_image(image, caption=None):
    """Render an image that adapts to the container width across Streamlit versions."""
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption)

# ═══════════════════════════════════════════════════════════
#                    STREAMLIT UI
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SKINLYTICS",
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
st.markdown('<h1 class="main-title">🎨 SKINLYTICS</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover your perfect color palette in seconds ✨</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📖 About")
    st.info("""
    **SKINLYTICS** uses the Monk Skin Tone (MST) scale to determine 
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
    
    st.markdown("### 📊 Skin Tone Scale")
    st.markdown("""
    The scale ranges from **1 (lightest)** to **50 (darkest)**, 
    designed to represent highly diverse skin tones with precision.
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
                    value=f"{mst_score} / 50",
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

            # Colors to avoid section
            st.markdown("---")
            st.markdown("### 🚫 Colors to Avoid")
            st.markdown("*These colors may not complement your skin tone as well*")
            
            colors_to_avoid_list = colors_to_avoid[mst_score]
            
            # Display colors to avoid as swatches
            cols = st.columns(5)
            for idx, color in enumerate(colors_to_avoid_list):
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

            # 2. Add this selector immediately after the loop
            st.markdown("### 🖐️ 3D Virtual Try-on")
            category = st.radio("Choose color category:", ["Recommended Colors", "Colors to Avoid"], horizontal=True, label_visibility="collapsed")
            if category == "Recommended Colors":
                selected_color = st.selectbox("Pick a recommended color to try:", recommendations, key="rec")
            else:
                selected_color = st.selectbox("Pick a color to avoid to try:", colors_to_avoid_list, key="avoid")
            selected_bg_hex = color_hex_map.get(selected_color, "#FFFFFF")
            detected_skin_hex = rgb_to_hex(skin_rgb)

            three_js_code = f"""
            <div id="three-container" style="width: 100%; height: 500px; background: #ffffff; border-radius: 20px; overflow: hidden; border: 1px solid #FFE5F0; position: relative;">
                <div id="canvas-target"></div>
                <div id="loading-overlay" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-family: sans-serif; color: #FF69B4; text-align: center;">
                    ✨ Matching Skin Tone...
                </div>
            </div>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/gh/mrdoob/three.js@r128/examples/js/loaders/GLTFLoader.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>

            <script>
                const width = document.getElementById('three-container').clientWidth;
                const height = 500;

                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
                camera.position.set(0, 0, 3);

                const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                renderer.setSize(width, height);
                
                // CRITICAL: Ensure the renderer uses a linear color space to match Python's RGB
                renderer.outputEncoding = THREE.LinearEncoding; 
                
                document.getElementById('canvas-target').appendChild(renderer.domElement);

                // Flat neutral lighting to prevent "washing out" the dark/light tones
                const ambientLight = new THREE.AmbientLight(0xffffff, 1.0); 
                scene.add(ambientLight);

                const bgPlane = new THREE.Mesh(
                    new THREE.PlaneGeometry(20, 20),
                    new THREE.MeshBasicMaterial({{ color: "{selected_bg_hex}", side: THREE.DoubleSide }})
                );
                bgPlane.position.z = -1;
                bgPlane.position.x = -15; 
                scene.add(bgPlane);

                const loader = new THREE.GLTFLoader();
                loader.setCrossOrigin('anonymous'); 
                const modelUrl = 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/models/gltf/LeePerrySmith/LeePerrySmith.glb'; 

                loader.load(modelUrl, (gltf) => {{
                    const mesh = gltf.scene;
                    document.getElementById('loading-overlay').style.display = 'none';

                    mesh.traverse((child) => {{
                        if (child.isMesh) {{
                            // DISCARD default textures and use high-vibrancy material
                            child.material = new THREE.MeshLambertMaterial({{ 
                                color: "{detected_skin_hex}",
                                emissive: "{detected_skin_hex}",
                                emissiveIntensity: 0.15, // Adds a subtle "glow" to keep color true in shade
                                reflectivity: 0,
                                combine: THREE.NoCombine
                            }});
                        }}
                    }});

                    mesh.scale.set(0.4, 0.4, 0.4);
                    mesh.position.y = -0.5;
                    scene.add(mesh);

                    function animate() {{
                        requestAnimationFrame(animate);
                        mesh.rotation.y += 0.005;
                        renderer.render(scene, camera);
                    }}
                    animate();

                    gsap.to(bgPlane.position, {{ x: 0, duration: 1.2, ease: "power2.out" }});
                }});
            </script>
            """

            # Render the component
            components.html(three_js_code, height=520)
            
            # Download section
            st.markdown("---")
            st.markdown("### 📥 Download Your Report")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            
            with col_btn1:
                colors_to_avoid_list = colors_to_avoid[mst_score]
                report_text = generate_report(uploaded_file, mst_score, skin_rgb, recommendations, colors_to_avoid_list)
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
    st.markdown("### 📊 The 50-Tone Scale")
    st.caption("Representing the full spectrum of human skin tones:")
    
    # Show first 25 tones
    cols = st.columns(5)
    for i in range(25):
        with cols[i % 5]:
            mst_num = i + 1
            swatch = create_color_swatch(skin_tones[mst_num], (60, 60))
            st.image(swatch, use_container_width=True)
            st.caption(f"Tone {mst_num}")
    
    st.caption("Tones 26-50:")
    cols = st.columns(5)
    for i in range(25, 50):
        with cols[i % 5]:
            mst_num = i + 1
            swatch = create_color_swatch(skin_tones[mst_num], (60, 60))
            st.image(swatch, use_container_width=True)
            st.caption(f"Tone {mst_num}")


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <p style='color: #FF69B4; font-size: 0.9rem;'>
            Made by Mahfuza Laskar || Powered by OpenCV & Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)
