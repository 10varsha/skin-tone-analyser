# 🎨 Skin Tone Analyzer

A beautiful, minimalistic web application that analyzes your skin tone using the Monk Skin Tone (MST) scale and recommends personalized color palettes.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)

## ✨ Features

- 🔍 **Face Detection** - Automatically detects faces using Haar Cascade
- 🎯 **MST Analysis** - Calculates skin tone on 1-10 scale
- 🎨 **Color Recommendations** - Suggests complementary colors
- 📊 **Visual Results** - Beautiful color swatches and visualizations
- 📥 **Downloadable Reports** - Save your personalized analysis
- 💖 **Candy Theme UI** - Minimalistic pink & blue interface
- ⚡ **Real-time Processing** - Instant analysis on upload

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this repository**

git clone https://github.com/yourusername/skin-tone-analyzer.git
cd skin-tone-analyzer

### Running the Application

1. **Start the Streamlit app**

streamlit run app.py

2. **Open your browser**

The app will automatically open at `http://localhost:8501`

If it doesn't open automatically, manually visit: `http://localhost:8501`

## 📖 How to Use

1. **Upload Image** 📸
   - Click the upload button
   - Select a clear photo of your face
   - Supported formats: JPG, PNG, JPEG

2. **Wait for Analysis** 🔍
   - The app detects your face automatically
   - Calculates your MST score (1-10)
   - Processes in real-time

3. **View Results** 🎯
   - See your MST score
   - View your detected skin tone color
   - Browse recommended color palette

4. **Download Report** 📥
   - Click "Download Report" button
   - Save as TXT file for future reference

## 💡 Tips for Best Results

✅ **DO:**
- Use natural lighting
- Face clearly visible
- No heavy makeup or filters
- High-quality image

❌ **AVOID:**
- Dark or low-light photos
- Heavily filtered images
- Blurry or low-resolution images
- Side angles (face should be frontal)

## 🎨 About MST Scale

The **Monk Skin Tone (MST) Scale** is a 10-shade scale designed to represent a diverse and inclusive range of skin tones:

- **1-3**: Lighter skin tones
- **4-6**: Medium skin tones
- **7-10**: Darker skin tones

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Image Processing**: OpenCV (opencv-python)
- **Face Detection**: Haar Cascade Classifier
- **Data Processing**: NumPy
- **Image Handling**: Pillow (PIL)

## 📁 Project Structure

skin-tone-analyzer/
├── app.py # Main application file
├── requirements.txt # Python dependencies
├── README.md # This file
└── .streamlit/
└── config.toml # Streamlit theme configuration

## 🐛 Troubleshooting

### Issue: "No face detected"
**Solution**: Ensure your face is clearly visible and well-lit. Try a different photo with better lighting.

### Issue: Import errors
**Solution**: Make sure all dependencies are installed:

pip install -r requirements.txt --upgrade

### Issue: Port already in use
**Solution**: Stop other Streamlit instances or use a different port:
streamlit run app.py --server.port 8502
### Issue: Haar Cascade not found
**Solution**: OpenCV includes Haar Cascades by default. If missing, reinstall:

pip uninstall opencv-python
pip install opencv-python


## 📝 How It Works

1. **Face Detection**
   - Uses Haar Cascade Classifier from OpenCV
   - Detects frontal faces in uploaded images
   - Extracts face region for analysis

2. **Skin Tone Calculation**
   - Resizes face region to 200x200 pixels
   - Calculates average RGB color values
   - Compares with 10 reference skin tones using Euclidean distance

3. **Color Matching**
   - Finds closest MST score (1-10)
   - Returns predefined color recommendations
   - Displays colors as visual swatches

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Monk Skin Tone Scale by Dr. Ellis Monk
- OpenCV community
- Streamlit team

## 📞 Support

If you have any questions or issues:
1. Check the Troubleshooting section above
2. Open an issue on GitHub
3. Contact via email

---

Made with 💖 using Python, OpenCV & Streamlit


