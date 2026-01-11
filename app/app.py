"""
Amazon Review Ranker - Premium Sentiment Analysis Dashboard
==========================================================
Modern UI with 3D animations, icons, and professional design.

Team: Jalloh (Naive Bayes) • Madiou (SVM) • Mustafa (Frontend)
Course: CCS3153 Natural Language Processing
"""

import streamlit as st
import os
import sys
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Amazon Review Ranker - Sentiment Analysis",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_text






# =============================================================================
# CSS Styles
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700;800&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    :root {
        --amazon-orange: #ff9900;
        --amazon-blue: #146eb4;
        --amazon-ink: #0f1111;
        --amazon-yellow: #ffd814;
        --amazon-yellow-dark: #f7ca00;
        --surface: #ffffff;
        --surface-muted: #f7f7f7;
        --border: #e6e6e6;
        --text-muted: #3f3f3f;
    }
    
    /* Global Text Visibility Fix - Targeted to avoid breaking icons */
    html, body, .stApp, p, h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stText, .stDataFrame, .stTextInput, .stTextArea, .stSelectbox {
        font-family: 'Source Sans 3', sans-serif !important;
        color: #0f1111;
    }
    
    /* Specific Streamlit Elements that might default to white */
    p, h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stText, .stDataFrame {
        color: #0f1111 !important;
    }

    /* Fix Dropdown/Selectbox Visibility */
    /* Fix Dropdown/Selectbox Visibility */
    div[data-baseweb="popover"], div[data-baseweb="menu"], div[data-baseweb="select"], ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
        color: #0f1111 !important;
    }
    div[data-baseweb="option"], li {
        color: #0f1111 !important;
        background-color: #ffffff !important;
    }

    /* Fix Code Block Contrast - Force Light Theme for Code */
    code, pre, .stCodeBlock, div[data-testid="stCodeBlock"] {
        background-color: #f6f8fa !important;
        color: #0f1111 !important;
        border-radius: 8px;
    }
    
    /* Ensure text inside code blocks is explicitly dark */
    code span, pre span {
        color: #0f1111 !important;
    }

    /* Hide Streamlit Header/Footer */
    header[data-testid="stHeader"], footer { display: none !important; }
    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1200px;
        z-index: 1; /* Ensure content stacks above background */
        position: relative;
    }

    /* Background */
    /* Background - Force Light Theme */
    div[data-testid="stAppViewContainer"], .stApp {
        background:
            radial-gradient(1200px circle at 10% -10%, rgba(255, 153, 0, 0.14), transparent 55%),
            radial-gradient(900px circle at 90% 0%, rgba(20, 110, 180, 0.12), transparent 60%),
            linear-gradient(180deg, #f7f7f7 0%, #ffffff 45%, #f2f3f5 100%) !important;
        color: #0f1111 !important;
    }
    
    div[data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image: linear-gradient(120deg, rgba(0,0,0,0.03) 1px, transparent 1px);
        background-size: 28px 28px;
        opacity: 0.4;
        pointer-events: none;
        z-index: 0;
    }

    /* Floating ambient shapes */
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); opacity: 0.25; }
        50% { transform: translateY(-18px) rotate(6deg); opacity: 0.55; }
        100% { transform: translateY(0px) rotate(0deg); opacity: 0.25; }
    }
    
    .bg-particle {
        position: fixed;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(255, 153, 0, 0.18), transparent 60%);
        filter: blur(1px);
        pointer-events: none;
        z-index: 0;
        animation: float 12s infinite ease-in-out;
    }
    
    /* Center Column Panel */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div {
        background: #ffffff;
        border-radius: 20px;
        padding: 2.4rem 2.6rem;
        box-shadow: 0 24px 50px rgba(15, 17, 17, 0.12);
        border: 1px solid var(--border);
        position: relative;
        z-index: 2; /* Content layer above background */
    }

    /* Header */
    .header-container {
        text-align: center;
        margin-bottom: 2.2rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        z-index: 2;
        position: relative;
    }

    /* Center Streamlit Image explicitly */
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    
    div[data-testid="stImage"] > img {
        margin: 0 auto;
        filter: drop-shadow(0 10px 18px rgba(0,0,0,0.12));
        pointer-events: none; /* Disable click to full screen */
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.9rem;
        background: #fff3d6;
        border: 1px solid #ffd814;
        color: #8a4b00 !important;
        border-radius: 999px;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-weight: 700;
    }

    .hero-title {
        color: var(--amazon-ink) !important;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.8rem 0 0.35rem;
        letter-spacing: -0.01em;
        text-align: center;
    }
    
    .subtitle {
        color: var(--text-muted) !important;
        font-size: 1.05rem;
        margin-top: 0.35rem;
        font-weight: 500;
        text-align: center;
        max-width: 680px;
    }

    .hero-tags {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 1.1rem;
    }

    .hero-tag {
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        background: #f6f7f9;
        border: 1px solid #e3e6e6;
        color: #0f1111 !important;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    
    /* Input Styling */
    .stTextArea textarea {
        background: #ffffff !important;
        color: #0f1111 !important;
        border: 1.8px solid #d5d9d9 !important;
        border-radius: 14px !important;
        padding: 0.95rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        box-shadow: 0 10px 18px rgba(15, 17, 17, 0.05);
    }
    
    .stTextArea textarea::placeholder {
        color: #6b6f73 !important;
        opacity: 1;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--amazon-yellow-dark) !important;
        box-shadow: 0 0 0 3px rgba(247, 202, 0, 0.25) !important;
    }
    
    /* Selectbox Styling */
    .stSelectbox div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #0f1111 !important;
        border: 1.8px solid #d5d9d9 !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 18px rgba(15, 17, 17, 0.05);
    }

    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: var(--amazon-orange) !important;
    }

    /* Input Labels */
    .stSelectbox label, .stTextArea label {
        color: #0f1111 !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.4rem !important;
    }
    
    /* Button */
    .stButton button {
        background: linear-gradient(180deg, var(--amazon-yellow) 0%, var(--amazon-yellow-dark) 100%) !important;
        color: #0f1111 !important;
        border: 1px solid #fcd200 !important;
        font-weight: 700 !important;
        padding: 0.85rem 2rem !important;
        border-radius: 12px !important;
        width: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 2px 0 #e1b000;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(255, 153, 0, 0.3);
    }
    
    /* Results */
    .result-card {
        padding: 0;
        border-radius: 12px;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        background: white;
    }
    
    .result-card::after {
        display: none;
    }

    .result-positive { background: #0f8a6b; }
    .result-neutral { background: #d9890d; }
    .result-negative { background: #b42318; }
    
    .sentiment-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .confidence-bar {
        background: rgba(255,255,255,0.25);
        height: 6px;
        border-radius: 3px;
        margin-top: 1rem;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: white;
        height: 100%;
    }

    /* Plotly font alignment */
    .js-plotly-plot text {
        font-family: 'Source Sans 3', sans-serif !important;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e3e6e6;
        box-shadow: 0 10px 18px rgba(15, 17, 17, 0.05);
    }
    div[data-testid="stExpander"] summary {
        font-weight: 700;
        color: #0f1111 !important;
    }
    
    /* Status Pills */
    .status-container {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-bottom: 2rem;
        z-index: 1;
        position: relative;
    }
    
    .status-pill {
        background: #ffffff;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        color: #0f1111 !important;
        font-weight: 600;
        border: 1px solid #e3e6e6;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 8px 16px rgba(15, 17, 17, 0.06);
    }

    .dot {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 1.2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div {
            padding: 1.6rem 1.4rem;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 2rem;
        }

        .subtitle {
            font-size: 1rem;
        }
    }
</style>

<!-- Background Particles -->
<div class="bg-particle" style="top: 10%; left: 10%; width: 100px; height: 100px;"></div>
<div class="bg-particle" style="top: 60%; right: 10%; width: 150px; height: 150px; animation-delay: -2s;"></div>
<div class="bg-particle" style="bottom: 10%; left: 20%; width: 80px; height: 80px; animation-delay: -5s;"></div>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions
# =============================================================================
import re
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:
    nltk = None
    stopwords = None
    WordNetLemmatizer = None

# Enhanced NLTK download for Streamlit Cloud
if nltk:
    try:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('words')
        nltk.download('averaged_perceptron_tagger')
    except Exception:
        pass

def load_model(model_name):
    try:
        import joblib
        model_path = os.path.join(PROJECT_ROOT, "saved_models", f"{model_name}.pkl")
        if os.path.exists(model_path):
            return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
    return None

def get_prediction(model, text):
    if model is None: return None, None
    try:
        cleaned_text = preprocess_text(text, use_pos_tagging=True)
        if not cleaned_text.strip(): return None, None
        prediction = model.predict([cleaned_text])[0]
        
        prob = None
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba([cleaned_text])[0]
        elif hasattr(model, 'decision_function'):
            d = model.decision_function([cleaned_text])[0]
            ex = np.exp(d - np.max(d))
            prob = ex / ex.sum()
        return str(prediction), prob
    except Exception:
        return None, None

def get_conf(probs, pred, model):
    if probs is None: return 0.0
    try:
        classes = list(model.classes_) if hasattr(model, 'classes_') else []
        if not classes and hasattr(model, 'named_steps'):
            classes = list(model.named_steps['clf'].classes_)
        
        if pred in classes:
            return float(probs[classes.index(pred)])
    except: pass
    return 0.0

# =============================================================================
# Helper Functions
# =============================================================================

# ... (load_model, get_prediction, get_conf remain the same)

def render_result(model_name, pred, conf):
    # Determine styles based on prediction
    if pred == "Positive":
        css_class = "result-positive"
        icon = "fa-check-circle"
        stars = "⭐⭐⭐⭐⭐"
        color = "#0f8a6b"
    elif pred == "Neutral":
        css_class = "result-neutral"
        icon = "fa-minus-circle"
        stars = "⭐⭐⭐"
        color = "#d9890d"
    else:  # Negative
        css_class = "result-negative"
        icon = "fa-exclamation-circle"
        stars = "⭐"
        color = "#b42318"
    
    return f"""<div class="result-card" style="background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 1rem; padding: 0; overflow: hidden; border: 1px solid #e0e0e0;"><div style="background: {color}; padding: 0.8rem 1.5rem; display: flex; justify-content: space-between; align-items: center;"><div style="color: white; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.85rem;">{model_name}</div><i class="fa-solid {icon}" style="color: white; font-size: 1.2rem;"></i></div><div style="padding: 1.5rem; text-align: center;"><div style="font-size: 2.2rem; font-weight: 800; color: {color} !important; margin-bottom: 0.5rem; letter-spacing: -0.02em;">{pred}</div><div style="background: #f8f9fa; border-radius: 8px; padding: 1rem; margin-top: 1rem;"><div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; color: #555; font-size: 0.9rem; font-weight: 600;"><span>Confidence</span><span style="color: #0f1111;">{conf*100:.1f}%</span></div><div style="background: #e9ecef; height: 8px; border-radius: 4px; overflow: hidden;"><div style="background: {color}; width: {conf*100}%; height: 100%; border-radius: 4px;"></div></div></div></div></div>"""

# =============================================================================
# Chart Function
# =============================================================================
def get_model_classes(model):
    if hasattr(model, 'classes_'):
        return list(model.classes_)
    if hasattr(model, 'named_steps'):
        return list(model.named_steps['clf'].classes_)
    return []

def create_comparison_chart(bayes_probs, svm_probs, classes_ref):
    # Standardize classes to Negative, Neutral, Positive order
    target_order = ['Negative', 'Neutral', 'Positive']
    
    bayes_data = []
    svm_data = []
    
    # Helper to safe get prob
    def get_prob(probs, source_classes, target):
        if probs is None or not source_classes: return 0.0
        try:
            if target in source_classes:
                idx = source_classes.index(target)
                return probs[idx] * 100
        except: pass
        return 0.0

    # Extract data for Naive Bayes
    b_classes = classes_ref.get('bayes', [])
    for cls in target_order:
        bayes_data.append(get_prob(bayes_probs, b_classes, cls))
        
    # Extract data for SVM
    s_classes = classes_ref.get('svm', [])
    for cls in target_order:
        svm_data.append(get_prob(svm_probs, s_classes, cls))
        
    fig = go.Figure()
    
    # Naive Bayes Bars
    fig.add_trace(go.Bar(
        name='Naive Bayes',
        x=target_order,
        y=bayes_data,
        marker_color='#146eb4',
        text=[f"{v:.1f}%" for v in bayes_data],
        textposition='auto',
        textfont=dict(color='white')
    ))
    
    # SVM Bars
    fig.add_trace(go.Bar(
        name='SVM',
        x=target_order,
        y=svm_data,
        marker_color='#ff9900',
        text=[f"{v:.1f}%" for v in svm_data],
        textposition='auto',
        textfont=dict(color='black')
    ))
    
    fig.update_layout(
        title=dict(
            text='Prediction Probability Distribution',
            font=dict(size=18, color='#0f1111', family='Source Sans 3', weight=700)
        ),
        barmode='group',
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,255,255,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#0f1111')
        ),
        yaxis=dict(
            title='Probability (%)',
            range=[0, 105],
            gridcolor='rgba(0,0,0,0.05)',
            title_font=dict(color='#0f1111'),
            tickfont=dict(color='#0f1111')
        ),
        xaxis=dict(
            gridcolor='rgba(0,0,0,0)',
            tickfont=dict(color='#0f1111', size=13, weight=600)
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=350
    )
    return fig

# =============================================================================
# Detailed Pipeline Logic
# =============================================================================
def get_pipeline_steps(text):
    steps = {}
    try:
        # 1. Original Text
        steps['original'] = text
        
        # 2. Lowercased
        text_lower = text.lower()
        steps['lowercased'] = text_lower
        
        # 3. URLs Removed
        text_no_urls = re.sub(r'http\S+|www\S+', '', text_lower)
        steps['urls_removed'] = text_no_urls
        
        # 4. HTML Removed
        text_no_html = re.sub(r'<.*?>', '', text_no_urls)
        steps['html_removed'] = text_no_html
        
        # 5. Punctuation Removed
        text_clean = re.sub(r'[^a-z\s]', '', text_no_html)
        steps['punctuation_removed'] = text_clean
        
        # 6. Tokenization
        tokens = text_clean.split()
        steps['tokenization'] = str(tokens)
        
        # 7. POS Tagging
        try:
            from nltk import pos_tag
            pos_tags = pos_tag(tokens)
            steps['pos_tagging'] = str(pos_tags)
        except:
            steps['pos_tagging'] = "POS tagger not available"
        
        # 8. Stopword Removal
        try:
            stop_words = set(stopwords.words('english')) if stopwords else set()
        except:
            stop_words = set()
        
        tokens_no_stop = [w for w in tokens if w not in stop_words and len(w) > 2]
        removed_words = [w for w in tokens if w in stop_words or len(w) <= 2]
        steps['stopword_removal'] = str(tokens_no_stop)
        steps['stopwords_removed'] = str(removed_words)
        
        # 9. Lemmatization (Final)
        cleaned_text = preprocess_text(text, use_pos_tagging=True)
        steps['lemmatization'] = cleaned_text
        
        return steps
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# Main Application
# =============================================================================
def main():
    # 1. Header
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    
    # Logo
    logo_path = os.path.join(PROJECT_ROOT, "image.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=240)
    st.markdown('<div class="hero-title">Amazon Review Ranker</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="hero-tags">
            <div class="hero-tag">Verified Reviews</div>
            <div class="hero-tag">Naive Bayes</div>
            <div class="hero-tag">SVM</div>
            <div class="hero-tag">Fast Insights</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 2. Models Check
    bayes = load_model("naive_bayes_final")
    svm = load_model("svm_final")
    
    st.markdown(f"""
    <div class="status-container">
        <div class="status-pill"><div class="dot"></div> Naive Bayes Ready</div>
        <div class="status-pill"><div class="dot"></div> SVM Ready</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Main Card (Centered via Column Styling)
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        # Inputs
        mode = st.selectbox("Analysis Mode", ["Both Models", "Naive Bayes", "SVM"])
        text = st.text_area("Product Review", height=120, placeholder="Example: This phone case is durable and looks great! 5 stars.")
        
        if st.button("Analyze Sentiment"):
            if text.strip():
                st.markdown("---")
                
                # Logic
                r1, r2 = st.columns(2) if mode == "Both Models" else (st.container(), None)
                
                bayes_prob = None
                svm_prob = None
                
                # Naive Bayes
                if "Naive" in mode or "Both" in mode:
                    with r1:
                        p, bayes_prob = get_prediction(bayes, text)
                        if p:
                            bayes_conf = get_conf(bayes_prob, p, bayes)
                            st.markdown(render_result("Naive Bayes", p, bayes_conf), unsafe_allow_html=True)
                            
                # SVM
                if "SVM" in mode or "Both" in mode:
                    target = r2 if mode == "Both Models" else r1
                    with target:
                        p, svm_prob = get_prediction(svm, text)
                        if p:
                            svm_conf = get_conf(svm_prob, p, svm)
                            st.markdown(render_result("SVM", p, svm_conf), unsafe_allow_html=True)
                
                # Comparison Chart
                if mode == "Both Models" and (bayes_prob is not None or svm_prob is not None):
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Gather classes
                    classes_ref = {
                        'bayes': get_model_classes(bayes),
                        'svm': get_model_classes(svm)
                    }
                    
                    st.plotly_chart(create_comparison_chart(bayes_prob, svm_prob, classes_ref), config={'displayModeBar': False})
                
                # Pipeline Visualization
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("View Analysis Details"):
                    steps = get_pipeline_steps(text)
                    if "error" in steps:
                        st.warning(steps["error"])
                    else:
                        st.markdown("### 1. Original Text")
                        st.code(steps.get('original', ''), language="text")
                        
                        st.markdown("### 2. Lowercased")
                        st.code(steps.get('lowercased', ''), language="text")
                        
                        st.markdown("### 3. URLs Removed")
                        st.code(steps.get('urls_removed', ''), language="text")
                        
                        st.markdown("### 4. HTML Removed")
                        st.code(steps.get('html_removed', ''), language="text")
                        
                        st.markdown("### 5. Punctuation Removed")
                        st.code(steps.get('punctuation_removed', ''), language="text")
                        
                        st.markdown("### 6. Tokenization")
                        st.code(steps.get('tokenization', ''), language="text")
                        
                        st.markdown("### 7. POS Tagging")
                        st.code(steps.get('pos_tagging', ''), language="text")
                        
                        st.markdown("### 8. Stopword Removal")
                        st.code(steps.get('stopword_removal', ''), language="text")
                        st.markdown("**Stopwords Removed:**")
                        st.code(steps.get('stopwords_removed', ''), language="text")
                        
                        st.markdown("### 9. Lemmatization (Model Input)")
                        st.code(steps.get('lemmatization', ''), language="text")
                    
            else:
                st.warning("Please enter some text")

try:
    main()
except Exception as e:
    import traceback
    st.error(f"Application Error: {e}")
    st.code(traceback.format_exc())
