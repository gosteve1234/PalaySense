import streamlit as st
import time
from src.detector import predict_image, model, class_names
from src.export import generate_png, generate_pdf

st.set_page_config(page_title="PALAi", page_icon="🌿", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .header {
        display: flex; align-items: center; gap: 10px;
        padding: 12px 0 24px 0; border-bottom: 1px solid #e5e5e0;
        margin-bottom: 40px;
    }
    .logo-box {
        background: #eef4e7; border: 1px solid #b5d07a;
        border-radius: 10px; padding: 6px 10px;
        font-size: 20px; line-height: 1;
    }
    .logo-title { font-size: 16px; font-weight: 600; color: #1a1a16; margin: 0; }
    .logo-sub   { font-size: 11px; color: #6b6b65; margin: 0; }

    .hero { text-align: center; margin-bottom: 40px; }
    .hero h1 { font-size: 30px; font-weight: 600; color: #1a1a16; }
    .hero p  { font-size: 14px; color: #5a5a54; max-width: 500px; margin: 8px auto 0; line-height: 1.65; }

    .panel-label { font-size: 13px; font-weight: 500; color: #5a5a54; margin-bottom: 8px; }

    .result-box {
        border: 1.5px solid #b5d07a; border-radius: 10px;
        padding: 20px; background: #fff; min-height: 200px;
    }
    .result-category { font-size: 22px; font-weight: 600; color: #2d5016; margin-bottom: 6px; }
    .result-accuracy  { font-size: 14px; color: #5a5a54; margin-bottom: 14px; }

    .bar-bg   { background: #e5e5e0; border-radius: 99px; height: 8px; }
    .bar-fill { background: #3d6b1e; border-radius: 99px; height: 8px; }

    .hiw-box {
        background: #fff; border: 1px solid #e5e5e0;
        border-radius: 12px; padding: 24px; margin-top: 32px;
    }
    .hiw-title { font-size: 15px; font-weight: 600; margin-bottom: 16px; }
    .step-num {
        background: #eef4e7; color: #2d5016;
        border-radius: 50%; width: 28px; height: 28px;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 12px; font-weight: 600; margin-right: 10px;
    }
    .step-title { font-size: 14px; font-weight: 500; }
    .step-sub   { font-size: 12px; color: #6b6b65; }

    div[data-testid="stFileUploader"] > label { display: none; }
    div[data-testid="stFileUploader"] section {
        border: 1.5px dashed #d1d5db !important;
        border-radius: 10px !important;
        background: #fafaf9 !important;
        padding: 32px !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #b5d07a !important;
        background: #eef4e7 !important;
    }

    div[data-testid="stButton"] > button {
        background: #2d5a1b !important; color: white !important;
        border: none !important; border-radius: 10px !important;
        padding: 14px 24px !important; font-size: 15px !important;
        font-weight: 500 !important; width: 100% !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: #245014 !important;
    }

    .stAlert { border-radius: 8px !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

#  Header
st.markdown("""
<div class="header">
    <div class="logo-box">🌿</div>
    <div>
        <p class="logo-title">PALAi</p>
        <p class="logo-sub">Brown Spot Detection</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <h1>Detect Brown Spots Instantly</h1>
    <p>Upload an image and our AI will analyze it for brown spots, providing annotated results with detection confidence.</p>
</div>
""", unsafe_allow_html=True)

#  Main Panels
col_upload, col_result = st.columns(2, gap="medium")

with col_upload:
    st.markdown('<p class="panel-label">Upload Image</p>', unsafe_allow_html=True)
    image_file = st.file_uploader(
        "Upload", type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )
    if image_file:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:12px 16px;
             background:#fafaf9;border:1.5px solid #b5d07a;border-radius:10px;margin-top:8px;">
            <span style="font-size:20px;">🖼️</span>
            <span style="font-size:13px;font-weight:500;color:#1a1a16;">{image_file.name}</span>
        </div>
        """, unsafe_allow_html=True)
with col_result:
    st.markdown('<p class="panel-label">Detection Result</p>', unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.markdown("""
        <div class="result-box" style="display:flex;flex-direction:column;
             align-items:center;justify-content:center;min-height:200px;">
            <div style="font-size:32px;margin-bottom:8px;">🖼️</div>
            <p style="color:#9ca3af;font-size:13px;">Results will appear here</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        label, confidence = st.session_state.result
        bar_width = int(confidence)
        st.markdown(f"""
        <div class="result-box">
            <p style="font-size:12px;color:#6b6b65;margin-bottom:4px;">Detection Category</p>
            <p class="result-category">{label}</p>
            <p class="result-accuracy">Confidence: {confidence:.2f}% (how sure we are at Guessing this image)</p>
            <div style="display:flex;align-items:center;gap:10px;">
                <div class="bar-bg" style="flex:1;">
                    <div class="bar-fill" style="width:{bar_width}%;"></div>
            <p class="Recommendations">Recommedations: </p>
                </div>
                <span style="font-size:12px;font-weight:500;color:#2d5016;min-width:36px;">{confidence:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True) #Currently working on with recommendations

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            if image_file is not None:
                image_file.seek(0)
                png_buf = generate_png(image_file, label, confidence)
                st.download_button(
                    label="⬇️ Download as PNG",
                    data=png_buf,
                    file_name="palai_result.png",
                    mime="image/png",
                    use_container_width=True,
                )

        with dl_col2:
            if image_file is not None:
                image_file.seek(0)
                pdf_buf = generate_pdf(image_file, label, confidence)
                st.download_button(
                    label="⬇️ Download as PDF",
                    data=pdf_buf,
                    file_name="palai_result.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

# Analyze Button
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

btn_col, _ = st.columns([1, 1])
with btn_col:
    if st.button("Analyze Image →"):
        if image_file is None:
            st.error("Please upload an image first.")
        else:
            progress = st.progress(0, text="Scanning image...")
            for pct, msg in [
                (20, "Preparing image..."),
                (45, "Extracting features..."),
                (75, "Running model inference..."),
                (95, "Finalizing result..."),
            ]:
                progress.progress(pct, text=msg)
                time.sleep(0.15)

            image_file.seek(0)
            label, confidence = predict_image(image_file, model, class_names)
            st.session_state.result = (label, confidence)
            progress.progress(100, text="Scan complete")
            time.sleep(0.15)
            progress.empty()
            st.rerun()

# Reset
with _:
    if st.button("↺ Reset", key="reset"):
        st.session_state.pop("result", None)
        st.rerun()

# How It Works
st.markdown("""
<div class="hiw-box">
    <p class="hiw-title">How it works</p>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">
        <div style="display:flex;align-items:flex-start;gap:10px;">
            <span class="step-num">1</span>
            <div>
                <p class="step-title">Upload</p>
                <p class="step-sub">Select or drag an image to upload</p>
            </div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:10px;">
            <span class="step-num">2</span>
            <div>
                <p class="step-title">Analyze</p>
                <p class="step-sub">Click analyze to process the image</p>
            </div>
        </div>
        <div style="display:flex;align-items:flex-start;gap:10px;">
            <span class="step-num">3</span>
            <div>
                <p class="step-title">Results</p>
                <p class="step-sub">View annotated results with confidence and recommendations</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)