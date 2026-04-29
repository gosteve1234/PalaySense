import streamlit as st
import time
import io
import base64
from datetime import datetime
from src.detector import predict_image, model, class_names
from src.export import generate_png, generate_pdf

st.set_page_config(page_title="PALAi", page_icon="🌿", layout="wide")

# ── Session state ──
for key, val in [
    ("scan_history", []),
    ("show_history", False),
    ("batch_mode", False),
    ("feedback", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helper: image to base64 ──
def img_to_b64(file_obj):
    file_obj.seek(0)
    return base64.b64encode(file_obj.read()).decode()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .stApp { background: #f0f4eb; }
    footer, #MainMenu, header { visibility: hidden; }

    /* ── Navbar ── */
    .navbar { display:flex; align-items:center; justify-content:space-between; padding:16px 0 22px 0; border-bottom:1.5px solid #d4e6b0; margin-bottom:32px; animation:fadeDown 0.5s ease both; }
    .nav-left { display:flex; align-items:center; gap:14px; }
    .nav-icon { width:40px; height:40px; background:linear-gradient(135deg,#2a5218,#4a8e28); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:20px; box-shadow:0 4px 14px rgba(42,82,24,0.3); }
    .nav-title { font-size:18px; font-weight:700; color:#1a2e0f; letter-spacing:-0.4px; }
    .nav-sub { font-size:11px; color:#7a9a5a; margin-top:1px; }
    .nav-pill { background:#e4f0d0; border:1px solid #aed176; color:#3a6b1a; font-size:10px; font-weight:700; padding:5px 14px; border-radius:99px; letter-spacing:1.2px; }

    /* ── Hero ── */
    .hero { text-align:center; padding:8px 0 40px 0; animation:fadeUp 0.6s ease both 0.1s; }
    .hero-tag { display:inline-block; background:#e4f0d0; border:1px solid #aed176; color:#3a6b1a; font-size:10px; font-weight:700; padding:5px 16px; border-radius:99px; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:20px; }
    .hero h1 { font-size:44px; font-weight:700; color:#1a2e0f; letter-spacing:-2px; line-height:1.1; margin:0 0 18px 0; }
    .hero h1 em { font-style:normal; color:#4a8e28; }
    .hero p { font-size:15px; color:#6b8050; max-width:460px; margin:0 auto; line-height:1.75; }

    /* ── Panel label ── */
    .panel-label { font-size:10px; font-weight:700; color:#8aaa60; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px; }

    /* ── Upload ── */
    div[data-testid="stFileUploader"] > label { display:none; }
    div[data-testid="stFileUploader"] section { border:2px dashed #c0d890 !important; border-radius:18px !important; background:#ffffff !important; padding:36px 32px !important; transition:all 0.25s ease !important; }
    div[data-testid="stFileUploader"] section:hover { border-color:#4a8e28 !important; background:#f4fae8 !important; box-shadow:0 8px 28px rgba(74,142,40,0.12) !important; }
    .file-tag { display:flex; align-items:center; gap:12px; padding:12px 16px; margin-top:12px; background:#fff; border:1.5px solid #c0d890; border-radius:12px; animation:fadeUp 0.3s ease both; }
    .file-dot { width:8px; height:8px; border-radius:50%; background:#4a8e28; flex-shrink:0; }
    .file-name { font-size:13px; font-weight:500; color:#1a2e0f; }



    /* ── Result card ── */
    .rcard { border-radius:22px; padding:28px 28px 24px; min-height:240px; position:relative; overflow:hidden; animation:fadeUp 0.55s cubic-bezier(.22,1,.36,1) both; box-shadow:0 12px 40px rgba(0,0,0,0.1); }
    .rcard::after { content:''; position:absolute; top:-60px; right:-60px; width:200px; height:200px; border-radius:50%; opacity:0.07; pointer-events:none; }
    .rcard-healthy { background:linear-gradient(140deg,#1b4a0c 0%,#2e7018 60%,#3d8e20 100%); border:1.5px solid rgba(255,255,255,0.1); }
    .rcard-healthy::after { background:#a8f060; }
    .rcard-diseased { background:linear-gradient(140deg,#4a0e05 0%,#7a2010 60%,#9e3018 100%); border:1.5px solid rgba(255,255,255,0.1); }
    .rcard-diseased::after { background:#f08060; }
    .rcard-unknown { background:#ffffff; border:1.5px solid #d4e6b0; box-shadow:0 4px 20px rgba(0,0,0,0.06); }

    .rpill { display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:99px; font-size:10px; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:14px; }
    .rpill-light { background:rgba(255,255,255,0.15); color:rgba(255,255,255,0.85); }
    .rpill-dark  { background:#e4f0d0; color:#3a6b1a; }
    .rlabel-light { font-size:24px; font-weight:700; color:#fff; letter-spacing:-0.8px; margin-bottom:6px; }
    .rlabel-dark  { font-size:24px; font-weight:700; color:#1a2e0f; letter-spacing:-0.8px; margin-bottom:6px; }
    .rconf-light { font-size:13px; color:rgba(255,255,255,0.7); margin-bottom:6px; }
    .rconf-dark  { font-size:13px; color:#7a9a5a; margin-bottom:6px; }

    /* Reasoning box */
    .reasoning-box-light { background:rgba(255,255,255,0.1); border-radius:10px; padding:10px 14px; margin-bottom:14px; font-size:12px; color:rgba(255,255,255,0.8); line-height:1.6; border-left:3px solid rgba(255,255,255,0.3); }
    .reasoning-box-dark  { background:#f4fae8; border-radius:10px; padding:10px 14px; margin-bottom:14px; font-size:12px; color:#4a6030; line-height:1.6; border-left:3px solid #aed176; }

    .rbar-bg-light { background:rgba(255,255,255,0.2); border-radius:99px; height:6px; overflow:hidden; margin-bottom:4px; }
    .rbar-bg-dark  { background:#d4e6b0; border-radius:99px; height:6px; overflow:hidden; margin-bottom:4px; }
    .rbar-fill { height:100%; border-radius:99px; animation:barIn 0.9s cubic-bezier(.22,1,.36,1) both 0.35s; transform-origin:left; }
    .rbar-fill-light { background:rgba(255,255,255,0.9); }
    .rbar-fill-dark  { background:#4a8e28; }
    .rpct-light { font-size:11px; font-weight:600; color:rgba(255,255,255,0.55); font-family:'DM Mono',monospace; margin-bottom:16px; }
    .rpct-dark  { font-size:11px; font-weight:600; color:#8aaa60; font-family:'DM Mono',monospace; margin-bottom:16px; }

    .rdivider-light { height:1px; background:rgba(255,255,255,0.12); margin:16px 0; }
    .rdivider-dark  { height:1px; background:#d4e6b0; margin:16px 0; }
    .rrec-title-light { font-size:10px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; color:rgba(255,255,255,0.5); margin-bottom:12px; }
    .rrec-title-dark  { font-size:10px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; color:#8aaa60; margin-bottom:12px; }
    .rrec-item { display:flex; gap:10px; align-items:flex-start; margin-bottom:10px; font-size:13px; line-height:1.6; }
    .rrec-item-light { color:rgba(255,255,255,0.85); }
    .rrec-item-dark  { color:#4a6030; }
    .rrec-num { flex-shrink:0; width:20px; height:20px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:10px; font-weight:700; margin-top:2px; }
    .rrec-num-light { background:rgba(255,255,255,0.15); color:rgba(255,255,255,0.75); }
    .rrec-num-dark  { background:#e4f0d0; color:#3a6b1a; }

    /* Feedback */
    .feedback-row { display:flex; align-items:center; gap:10px; margin-top:14px; padding-top:14px; border-top:1px solid rgba(255,255,255,0.12); }
    .feedback-row-dark { border-top-color:#d4e6b0; }
    .feedback-label-light { font-size:11px; color:rgba(255,255,255,0.5); }
    .feedback-label-dark  { font-size:11px; color:#8aaa60; }
    .fb-btn { background:rgba(255,255,255,0.12); border:none; border-radius:8px; padding:6px 14px; font-size:16px; cursor:pointer; transition:all 0.15s ease; }
    .fb-btn:hover { background:rgba(255,255,255,0.22); transform:scale(1.1); }
    .fb-btn-dark { background:#f0f4eb; }
    .fb-btn-dark:hover { background:#e4f0d0; }

    /* Empty state */
    .empty-card { background:#fff; border:1.5px solid #d4e6b0; border-radius:22px; min-height:240px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:32px; box-shadow:0 4px 20px rgba(0,0,0,0.05); }
    .empty-icon { width:68px; height:68px; background:#e4f0d0; border-radius:20px; display:flex; align-items:center; justify-content:center; font-size:30px; margin-bottom:16px; }
    .empty-text { font-size:13px; color:#9ab878; font-weight:500; line-height:1.6; }

    /* Batch result card */
    .batch-card { background:#fff; border:1.5px solid #d4e6b0; border-radius:18px; padding:20px; margin-bottom:16px; animation:fadeUp 0.4s ease both; box-shadow:0 4px 16px rgba(0,0,0,0.06); display:flex; gap:16px; align-items:flex-start; }
    .batch-card-diseased { border-color:#e8a090; background:#fff8f6; }
    .batch-card-healthy  { border-color:#a8d888; background:#f6fbf0; }
    .batch-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
    .batch-fname { font-size:13px; font-weight:600; color:#1a2e0f; }
    .batch-badge { padding:3px 12px; border-radius:99px; font-size:11px; font-weight:700; }
    .batch-badge-diseased { background:#fce8e0; color:#8b2e0f; }
    .batch-badge-healthy  { background:#e4f0d0; color:#2a5218; }
    .batch-badge-unknown  { background:#f0f0ec; color:#6b6b65; }
    .batch-conf { font-size:12px; color:#8aaa60; margin-bottom:8px; }
    .batch-bar-bg { background:#e8f0d8; border-radius:99px; height:5px; overflow:hidden; margin-bottom:10px; }
    .batch-bar-fill { height:100%; border-radius:99px; background:#4a8e28; }
    .batch-bar-fill-diseased { background:#c44a1a; }
    .batch-reasoning { font-size:12px; color:#5a7040; background:#f4fae8; border-radius:8px; padding:8px 12px; border-left:3px solid #aed176; line-height:1.6; margin-bottom:10px; }

    /* Buttons */
    div[data-testid="stButton"] > button { background:linear-gradient(135deg,#2a5218,#3d8020) !important; color:white !important; border:none !important; border-radius:12px !important; padding:14px 24px !important; font-size:14px !important; font-weight:600 !important; width:100% !important; font-family:'Sora',sans-serif !important; letter-spacing:-0.2px !important; box-shadow:0 4px 16px rgba(42,82,24,0.28) !important; transition:all 0.2s ease !important; }
    div[data-testid="stButton"] > button:hover { background:linear-gradient(135deg,#1e3e12,#2e6018) !important; box-shadow:0 6px 22px rgba(42,82,24,0.38) !important; transform:translateY(-1px) !important; }
    div[data-testid="stDownloadButton"] > button { background:#fff !important; color:#2a5218 !important; border:1.5px solid #c0d890 !important; border-radius:10px !important; padding:10px 16px !important; font-size:13px !important; font-weight:600 !important; width:100% !important; font-family:'Sora',sans-serif !important; transition:all 0.2s ease !important; }
    div[data-testid="stDownloadButton"] > button:hover { background:#f4fae8 !important; border-color:#4a8e28 !important; transform:translateY(-1px) !important; }

    /* History */
    .hist-card { background:#fff; border:1px solid #ddecc0; border-radius:14px; padding:14px 16px; margin-bottom:10px; animation:fadeUp 0.3s ease both; transition:box-shadow 0.2s ease; }
    .hist-card:hover { box-shadow:0 4px 16px rgba(0,0,0,0.07); }
    .hist-label { font-size:13px; font-weight:600; color:#1a2e0f; margin:0 0 4px 0; }
    .hist-meta { font-size:11px; color:#8aaa60; margin:0; }
    .hist-badge { display:inline-block; padding:2px 10px; border-radius:99px; font-size:10px; font-weight:700; letter-spacing:0.5px; }
    .hist-badge-diseased { background:#fce8e0; color:#8b2e0f; }
    .hist-badge-healthy  { background:#e4f0d0; color:#2a5218; }
    .hist-badge-unknown  { background:#f0f0ec; color:#6b6b65; }

    /* How it works */
    .hiw { background:#fff; border:1.5px solid #d4e6b0; border-radius:20px; padding:32px; margin-top:40px; }
    .hiw-label { font-size:10px; font-weight:700; color:#8aaa60; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:24px; }
    .hiw-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:24px; }
    .hiw-step { display:flex; gap:14px; align-items:flex-start; }
    .hiw-num { width:34px; height:34px; border-radius:10px; background:linear-gradient(135deg,#2a5218,#4a8e28); color:white; font-size:14px; font-weight:700; display:flex; align-items:center; justify-content:center; flex-shrink:0; box-shadow:0 4px 12px rgba(42,82,24,0.25); }
    .hiw-step-t { font-size:14px; font-weight:600; color:#1a2e0f; margin-bottom:4px; }
    .hiw-step-s { font-size:12px; color:#8aaa60; line-height:1.55; }

    /* ── Mobile responsiveness ── */
    @media (max-width: 768px) {
        .hero h1 { font-size:28px; letter-spacing:-1px; }
        .hero p { font-size:13px; }
        .navbar { flex-wrap:wrap; gap:10px; }
        .hiw-grid { grid-template-columns:1fr; }
        .rcard { padding:20px 18px; }
        .rlabel-light, .rlabel-dark { font-size:20px; }
        div[data-testid="stFileUploader"] section { padding:24px 16px !important; }
        .batch-header { flex-direction:column; align-items:flex-start; gap:6px; }
    }
    @media (max-width: 480px) {
        .hero h1 { font-size:22px; }
        .nav-title { font-size:15px; }
        .nav-sub { display:none; }
    }

    /* Animations */
    @keyframes fadeUp   { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
    @keyframes fadeDown { from{opacity:0;transform:translateY(-12px)} to{opacity:1;transform:translateY(0)} }
    @keyframes barIn    { from{transform:scaleX(0)} to{transform:scaleX(1)} }
    .stAlert { border-radius:12px !important; }
</style>

<!-- Image preview modal -->
<div id="imgModal" onclick="closeModal()" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.82);z-index:9999;align-items:center;justify-content:center;cursor:zoom-out;">
    <span onclick="closeModal()" style="position:fixed;top:18px;right:28px;color:#fff;font-size:36px;font-weight:300;cursor:pointer;line-height:1;z-index:10000;">×</span>
    <img id="modalImg" src="" alt="Preview" style="max-width:90vw;max-height:90vh;border-radius:14px;box-shadow:0 20px 60px rgba(0,0,0,0.5);object-fit:contain;" onclick="event.stopPropagation()">
</div>
<script>
function openModal(src) {
    var m = document.getElementById('imgModal');
    document.getElementById('modalImg').src = src;
    m.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}
function closeModal() {
    document.getElementById('imgModal').style.display = 'none';
    document.body.style.overflow = '';
}
document.addEventListener('keydown', function(e) { if(e.key==='Escape') closeModal(); });
</script>
""", unsafe_allow_html=True)

# ── Navbar ──
st.markdown("""
<div class="navbar">
    <div class="nav-left">
        <div class="nav-icon">🌿</div>
        <div>
            <div class="nav-title">PALAi</div>
            <div class="nav-sub">Brown Spot Detection System</div>
        </div>
    </div>
    <div class="nav-pill">AI POWERED</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
    <div class="hero-tag">Rice Crop Health Analysis</div>
    <h1>Detect Brown Spots <em>Instantly</em></h1>
    <p>Upload a photo of your rice plant and our AI will analyze it for brown spot disease — giving you results and actionable recommendations in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ── Mode toggle ──
mode_col, _ = st.columns([2, 4])
with mode_col:
    batch_mode = st.toggle("📦 Batch Mode (multiple images)", value=st.session_state.batch_mode)
    st.session_state.batch_mode = batch_mode

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════
# ── SINGLE MODE ──
# ═══════════════════════════════
if not batch_mode:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<p class="panel-label">Upload Image</p>', unsafe_allow_html=True)
        image_file = st.file_uploader(
            "Upload", type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed"
        )
        if image_file:
            # ── Thumbnail in file tag ──
            b64 = img_to_b64(image_file)
            img_src = f"data:image/jpeg;base64,{b64}"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;margin-top:12px;
                 background:#fff;border:1.5px solid #c0d890;border-radius:12px;cursor:pointer;"
                 onclick="openModal('{img_src}')" title="Click to preview full image">
                <img src="{img_src}" alt="{image_file.name}"
                     style="width:48px;height:48px;object-fit:cover;border-radius:8px;flex-shrink:0;">
                <div>
                    <div style="font-size:13px;font-weight:600;color:#1a2e0f;">{image_file.name}</div>
                    <div style="font-size:11px;color:#9ab878;margin-top:2px;">Click to preview</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        st.markdown('<p class="panel-label">Detection Result</p>', unsafe_allow_html=True)

        if "result" not in st.session_state:
            st.markdown("""
            <div class="empty-card">
                <div class="empty-icon">🔬</div>
                <div class="empty-text">Upload an image and click<br><strong>Analyze Image</strong> to see results</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            label, confidence, recommendation, reasoning_text = st.session_state.result
            bar_width = min(int(confidence), 100)

            if "Brown Spot" in label:
                card_cls, pill_cls, tc, icon = "rcard-diseased", "rpill-light", "light", "🔴"
            elif "Healthy" in label:
                card_cls, pill_cls, tc, icon = "rcard-healthy", "rpill-light", "light", "🟢"
            else:
                card_cls, pill_cls, tc, icon = "rcard-unknown", "rpill-dark", "dark", "⚪"

            rec_items = "".join(
                '<div class="rrec-item rrec-item-' + tc + '">'
                '<div class="rrec-num rrec-num-' + tc + '">' + str(i) + '</div>'
                '<span>' + r + '</span></div>'
                for i, r in enumerate(recommendation, 1)
            )

            # Feedback state
            fb_key = "single"
            fb = st.session_state.feedback.get(fb_key, None)
            fb_html = (
                '<div class="feedback-row' + (' feedback-row-dark' if tc=='dark' else '') + '">'
                '<span class="feedback-label-' + tc + '">Was this result helpful?</span>'
                '</div>'
            )

            card = (
                '<div class="rcard ' + card_cls + '">'
                '<div class="rpill ' + pill_cls + '">' + icon + ' &nbsp;' + label + '</div>'
                '<div class="rlabel-' + tc + '">' + label + '</div>'
                '<div class="rconf-' + tc + '">Confidence: <strong>' + f'{confidence:.1f}%</strong></div>'
                '<div class="rbar-bg-' + tc + '"><div class="rbar-fill rbar-fill-' + tc + '" style="width:' + str(bar_width) + '%;"></div></div>'
                '<div class="rpct-' + tc + '">' + f'{confidence:.1f}% match</div>'
                '<div class="reasoning-box-' + tc + '"><strong>AI Reasoning:</strong> ' + reasoning_text + '</div>'
                '<div class="rdivider-' + tc + '"></div>'
                '<div class="rrec-title-' + tc + '">Recommendations</div>'
                + rec_items +
                '</div>'
            )
            st.markdown(card, unsafe_allow_html=True)

            # ── Feedback buttons ──
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            fb_col1, fb_col2, fb_col3 = st.columns([2, 1, 1])
            with fb_col1:
                st.markdown('<p style="font-size:12px;color:#8aaa60;margin:10px 0 0 0;">Was this result helpful?</p>', unsafe_allow_html=True)
            with fb_col2:
                if st.button("👍" + (" ✓" if fb == "up" else ""), key="fb_up", help="Yes, helpful"):
                    st.session_state.feedback[fb_key] = "up"
                    st.toast("Thanks for your feedback! 🌿")
                    st.rerun()
            with fb_col3:
                if st.button("👎" + (" ✓" if fb == "down" else ""), key="fb_down", help="Not helpful"):
                    st.session_state.feedback[fb_key] = "down"
                    st.toast("Thanks! We'll use this to improve. 🔧")
                    st.rerun()

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                if image_file is not None:
                    image_file.seek(0)
                    png_buf = generate_png(image_file, label, confidence, recommendation)
                    st.download_button("⬇️ Download PNG", data=png_buf,
                        file_name="palai_result.png", mime="image/png", use_container_width=True)
            with dl_col2:
                if image_file is not None:
                    image_file.seek(0)
                    pdf_buf = generate_pdf(image_file, label, confidence, recommendation)
                    st.download_button("⬇️ Download PDF", data=pdf_buf,
                        file_name="palai_result.pdf", mime="application/pdf", use_container_width=True)

    # ── Single mode action buttons ──
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    btn_col, reset_col, history_col = st.columns([2, 1, 1])
    with btn_col:
        if st.button("Analyze Image →"):
            if image_file is None:
                st.error("Please upload an image first.")
            else:
                progress = st.progress(0, text="Initializing scan...")
                for pct, msg in [(20,"Preparing image..."),(45,"Extracting features..."),(75,"Running AI model..."),(95,"Finalizing result...")]:
                    progress.progress(pct, text=msg)
                    time.sleep(0.15)
                image_file.seek(0)
                label, confidence, recommendation, reasoning_text = predict_image(image_file, model, class_names)
                image_file.seek(0)
                image_bytes = image_file.read()
                st.session_state.result = (label, confidence, recommendation, reasoning_text)
                st.session_state.feedback.pop("single", None)
                st.session_state.scan_history.insert(0, {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file_name": image_file.name,
                    "label": label,
                    "confidence": float(confidence),
                    "recommendation": recommendation,
                    "reasoning": reasoning_text,
                    "image_bytes": image_bytes,
                })
                st.session_state.scan_history = st.session_state.scan_history[:15]
                progress.progress(100, text="Done!")
                time.sleep(0.15)
                progress.empty()
                st.rerun()
    with reset_col:
        if st.button("↺ Reset", key="reset"):
            st.session_state.pop("result", None)
            st.rerun()
    with history_col:
        if st.button("🕘 History", key="history"):
            st.session_state.show_history = not st.session_state.show_history

# ═══════════════════════════════
# ── BATCH MODE ──
# ═══════════════════════════════
else:
    st.markdown('<p class="panel-label">Upload Multiple Images</p>', unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "Upload multiple", type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if batch_files:
        st.markdown(f"""
        <div class="file-tag" style="margin-bottom:16px;">
            <div class="file-dot"></div>
            <span class="file-name">{len(batch_files)} image(s) selected</span>
        </div>
        """, unsafe_allow_html=True)

    batch_btn_col, _ = st.columns([1, 2])
    with batch_btn_col:
        run_batch = st.button("Analyze All Images →", key="batch_analyze")

    if run_batch:
        if not batch_files:
            st.error("Please upload at least one image.")
        else:
            st.session_state.batch_results = []
            batch_progress = st.progress(0, text="Starting batch scan...")
            for idx, f in enumerate(batch_files):
                pct = int((idx / len(batch_files)) * 100)
                batch_progress.progress(pct, text=f"Scanning {f.name}...")
                f.seek(0)
                try:
                    lbl, conf, rec, rsn = predict_image(f, model, class_names)
                    f.seek(0)
                    img_bytes = f.read()
                    st.session_state.batch_results.append({
                        "file_name": f.name,
                        "label": lbl,
                        "confidence": float(conf),
                        "recommendation": rec,
                        "reasoning": rsn,
                        "image_bytes": img_bytes,
                    })
                    # Add to history
                    st.session_state.scan_history.insert(0, {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "file_name": f.name,
                        "label": lbl,
                        "confidence": float(conf),
                        "recommendation": rec,
                        "reasoning": rsn,
                        "image_bytes": img_bytes,
                    })
                except Exception as e:
                    st.session_state.batch_results.append({
                        "file_name": f.name,
                        "label": "Error",
                        "confidence": 0,
                        "recommendation": [f"Error: {str(e)}"],
                        "reasoning": "Could not process this image.",
                        "image_bytes": None,
                    })
            st.session_state.scan_history = st.session_state.scan_history[:15]
            batch_progress.progress(100, text="Batch scan complete!")
            time.sleep(0.3)
            batch_progress.empty()
            st.rerun()

    # ── Show batch results ──
    if "batch_results" in st.session_state and st.session_state.batch_results:
        results = st.session_state.batch_results
        total = len(results)
        diseased = sum(1 for r in results if "Brown Spot" in r["label"])
        healthy = sum(1 for r in results if "Healthy" in r["label"])

        # Summary row
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin:16px 0;flex-wrap:wrap;">
            <div style="background:#fff;border:1.5px solid #d4e6b0;border-radius:12px;padding:12px 20px;flex:1;min-width:100px;text-align:center;">
                <div style="font-size:24px;font-weight:700;color:#1a2e0f;">{total}</div>
                <div style="font-size:11px;color:#8aaa60;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Total</div>
            </div>
            <div style="background:#f6fbf0;border:1.5px solid #a8d888;border-radius:12px;padding:12px 20px;flex:1;min-width:100px;text-align:center;">
                <div style="font-size:24px;font-weight:700;color:#2a5218;">{healthy}</div>
                <div style="font-size:11px;color:#5a8a3a;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Healthy</div>
            </div>
            <div style="background:#fff8f6;border:1.5px solid #e8a090;border-radius:12px;padding:12px 20px;flex:1;min-width:100px;text-align:center;">
                <div style="font-size:24px;font-weight:700;color:#8b2e0f;">{diseased}</div>
                <div style="font-size:11px;color:#a04030;font-weight:600;letter-spacing:1px;text-transform:uppercase;">Diseased</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for i, r in enumerate(results):
            if "Brown Spot" in r["label"]:
                bc, bb = "batch-card-diseased", "batch-badge-diseased"
                bar_cls = "batch-bar-fill-diseased"
            elif "Healthy" in r["label"]:
                bc, bb = "batch-card-healthy", "batch-badge-healthy"
                bar_cls = ""
            else:
                bc, bb = "", "batch-badge-unknown"
                bar_cls = ""

            bw = min(int(r["confidence"]), 100)

            # Image preview
            img_preview = ""
            if r.get("image_bytes"):
                b64 = base64.b64encode(r["image_bytes"]).decode()
                img_src = f"data:image/jpeg;base64,{b64}"
                fname_safe = r["file_name"]
                img_preview = (
                    '<div style="width:110px;height:110px;flex-shrink:0;border-radius:12px;overflow:hidden;cursor:pointer;" ' +
                    'onclick="openModal('' + img_src + '')" title="Click to preview">' +
                    '<img src="' + img_src + '" alt="' + fname_safe + '" style="width:100%;height:100%;object-fit:cover;"></div>'
                )

            st.markdown(f"""
            <div class="batch-card {bc}">
                <div style="flex:1;min-width:0;">
                    <div class="batch-header">
                        <span class="batch-fname">{r['file_name']}</span>
                        <span class="batch-badge {bb}">{r['label']}</span>
                    </div>
                    <div class="batch-conf">Confidence: {r['confidence']:.1f}%</div>
                    <div class="batch-bar-bg"><div class="batch-bar-fill {bar_cls}" style="width:{bw}%;"></div></div>
                    <div class="batch-reasoning"><strong>AI Reasoning:</strong> {r['reasoning']}</div>
                </div>
                {img_preview}
            </div>
            """, unsafe_allow_html=True)

            # Download PDF per result
            if r.get("image_bytes"):
                dl_col, _ = st.columns([1, 3])
                with dl_col:
                    pdf_buf = generate_pdf(
                        io.BytesIO(r["image_bytes"]),
                        r["label"], r["confidence"], r["recommendation"]
                    )
                    safe_name = r["file_name"].rsplit(".", 1)[0]
                    st.download_button(f"⬇️ PDF — {r['file_name']}", data=pdf_buf,
                        file_name=f"{safe_name}_result.pdf", mime="application/pdf",
                        key=f"batch_dl_{i}", use_container_width=True)

# ── History Panel ──
if st.session_state.get("show_history", False):
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="hiw" style="margin-top:0;">
        <div class="hiw-label">Recent Scan History</div>
    </div>
    """, unsafe_allow_html=True)
    if not st.session_state.scan_history:
        st.info("No scans yet. Analyze an image to start your history.")
    else:
        for i, item in enumerate(st.session_state.scan_history, start=1):
            hist_col, pdf_col = st.columns([5, 1], gap="small")
            if "Brown Spot" in item["label"]:
                badge_cls = "hist-badge-diseased"
            elif "Healthy" in item["label"]:
                badge_cls = "hist-badge-healthy"
            else:
                badge_cls = "hist-badge-unknown"
            with hist_col:
                st.markdown(f"""
                <div class="hist-card">
                    <p class="hist-label">
                        <span class="hist-badge {badge_cls}">{item['label']}</span>
                        &nbsp; {item['confidence']:.1f}% confidence
                    </p>
                    <p class="hist-meta">{item['file_name']} &nbsp;·&nbsp; {item['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
            with pdf_col:
                if item.get("image_bytes"):
                    past_pdf = generate_pdf(
                        io.BytesIO(item["image_bytes"]),
                        item["label"], item["confidence"],
                        item.get("recommendation", []),
                    )
                    safe_name = item.get("file_name", f"scan_{i}").rsplit(".", 1)[0]
                    st.download_button("⬇️ PDF", data=past_pdf,
                        file_name=f"{safe_name}_result.pdf", mime="application/pdf",
                        key=f"hist_pdf_{i}_{item['timestamp']}", use_container_width=True)

# ── How It Works ──
st.markdown("""
<div class="hiw">
    <div class="hiw-label">How it works</div>
    <div class="hiw-grid">
        <div class="hiw-step">
            <div class="hiw-num">1</div>
            <div>
                <div class="hiw-step-t">Upload a Photo</div>
                <div class="hiw-step-s">Take a clear photo of your rice leaf and upload it here.</div>
            </div>
        </div>
        <div class="hiw-step">
            <div class="hiw-num">2</div>
            <div>
                <div class="hiw-step-t">AI Analyzes</div>
                <div class="hiw-step-s">Our model scans the image for signs of brown spot disease.</div>
            </div>
        </div>
        <div class="hiw-step">
            <div class="hiw-num">3</div>
            <div>
                <div class="hiw-step-t">Get Results</div>
                <div class="hiw-step-s">See the diagnosis, AI reasoning, and follow the step-by-step recommendations.</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)