import os
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from document_processor import load_and_split_pdf
from rag_pipeline import (
    create_embeddings,
    create_llm,
    create_vectorstore,
    retrieve_documents,
    generate_answer
)


# =========================================================
# 1. Environment
# =========================================================

load_dotenv()


# =========================================================
# 2. Page configuration
# =========================================================

st.set_page_config(
    page_title="AI Document Intelligence Assistant",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# 3. Custom CSS (visual styling + animations only)
# =========================================================

CUSTOM_CSS = """
<style>

:root {
    --accent: #6366F1;
    --accent-2: #8B5CF6;
    --accent-dark: #4F46E5;
    --accent-soft: #EEF2FF;
    --text: #0F172A;
    --muted: #64748B;
    --border: #E2E8F0;
    --bg: #F5F7FF;
    --card: #FFFFFF;
    --success: #059669;
    --success-soft: #ECFDF5;
}

/* ---------- Keyframes ---------- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulseDot {
    0%   { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
@keyframes floatIcon {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-5px); }
}
@keyframes shine {
    0%   { left: -80%; }
    100% { left: 130%; }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ---------- Base ---------- */
.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(99, 102, 241, 0.10), transparent 40%),
        radial-gradient(circle at 90% 10%, rgba(139, 92, 246, 0.10), transparent 40%),
        var(--bg);
    color: var(--text);
}

.stApp p, .stApp label, .stApp li {
    color: var(--text);
}

footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.block-container > div > div[data-testid="stVerticalBlock"] > div {
    animation: fadeInUp 0.6s ease both;
}

/* ---------- Hero header ---------- */
.hero {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 22px;
    padding: 34px 38px;
    border-radius: 24px;
    background: linear-gradient(120deg, #4F46E5, #7C3AED, #6366F1, #8B5CF6);
    background-size: 300% 300%;
    animation: gradientShift 12s ease infinite;
    box-shadow: 0 20px 45px rgba(79, 70, 229, 0.30);
    margin-bottom: 26px;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px;
    right: -40px;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.10);
}
.hero::after {
    content: "";
    position: absolute;
    bottom: -80px;
    right: 140px;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.07);
}
.hero-icon {
    width: 68px;
    height: 68px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
    flex-shrink: 0;
    animation: floatIcon 4s ease-in-out infinite;
    z-index: 1;
}
.hero-text { z-index: 1; }
.hero-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #FFFFFF !important;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 16px;
    color: rgba(255, 255, 255, 0.88) !important;
    margin-top: 6px;
}
.hero-tags {
    margin-top: 14px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.hero-tag {
    font-size: 12px;
    font-weight: 600;
    color: #FFFFFF;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.28);
}

/* ---------- Cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    box-shadow: 0 2px 4px rgba(15, 23, 42, 0.03),
                0 12px 30px rgba(79, 70, 229, 0.07);
    transition: transform 0.25s ease, box-shadow 0.25s ease,
                border-color 0.25s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #C7D2FE !important;
    box-shadow: 0 4px 8px rgba(15, 23, 42, 0.04),
                0 18px 40px rgba(79, 70, 229, 0.13);
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 2px 4px rgba(15, 23, 42, 0.03),
                0 12px 30px rgba(79, 70, 229, 0.07);
}

.card-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text);
    margin-bottom: 2px;
}
.card-subtitle {
    font-size: 14px;
    color: var(--muted);
    margin-bottom: 16px;
}

.section-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 14px 0 8px 0;
}

.section-heading {
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin: 30px 0 14px 0;
    color: var(--text);
}

/* ---------- Upload ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #FAFBFF, #F3F0FF);
    border: 2px dashed #A5B4FC;
    border-radius: 18px;
    padding: 32px 20px;
    transition: all 0.25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent-2);
    background: linear-gradient(135deg, #EEF2FF, #EDE9FE);
    transform: scale(1.005);
}

/* ---------- Inputs ---------- */
.stTextInput input {
    border-radius: 14px;
    border: 1.5px solid var(--border);
    padding: 14px 16px;
    font-size: 15px;
    background: #FFFFFF;
    color: var(--text);
    transition: all 0.2s ease;
}
.stTextInput input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.16);
}

/* ---------- Buttons ---------- */
.stButton > button {
    position: relative;
    overflow: hidden;
    border-radius: 14px;
    font-weight: 600;
    padding: 0.7rem 1rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(120deg, #4F46E5, #7C3AED);
    border: none;
    color: #FFFFFF;
    font-size: 16px;
    padding: 0.9rem 1rem;
    box-shadow: 0 10px 24px rgba(79, 70, 229, 0.35);
}
.stButton > button[kind="primary"] p,
.stButton > button[data-testid="stBaseButton-primary"] p {
    color: #FFFFFF;
    font-weight: 700;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(120deg, #4338CA, #6D28D9);
    color: #FFFFFF;
    box-shadow: 0 14px 30px rgba(79, 70, 229, 0.45);
}
.stButton > button[kind="primary"]::after,
.stButton > button[data-testid="stBaseButton-primary"]::after {
    content: "";
    position: absolute;
    top: 0;
    left: -80%;
    width: 50%;
    height: 100%;
    background: linear-gradient(
        120deg,
        transparent,
        rgba(255, 255, 255, 0.35),
        transparent
    );
    transform: skewX(-20deg);
}
.stButton > button[kind="primary"]:hover::after,
.stButton > button[data-testid="stBaseButton-primary"]:hover::after {
    animation: shine 0.9s ease;
}

.stButton > button[kind="secondary"],
.stButton > button[data-testid="stBaseButton-secondary"] {
    background: #FFFFFF;
    border: 1.5px solid var(--border);
}
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    border-color: #FCA5A5;
    color: #B91C1C;
    background: #FEF2F2;
}

/* ---------- Badges ---------- */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 999px;
    margin-bottom: 8px;
}
.badge-success {
    background: var(--success-soft);
    color: var(--success);
    border: 1px solid #A7F3D0;
}
.badge-accent {
    background: var(--accent-soft);
    color: var(--accent-dark);
    border: 1px solid #C7D2FE;
}
.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10B981;
    display: inline-block;
    animation: pulseDot 1.8s infinite;
}

.field-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 2px;
}

.status-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--text);
    margin: 2px 0 10px 0;
}

.answer-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--accent-dark);
    margin: 18px 0 8px 0;
}
.answer-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: linear-gradient(120deg, var(--accent), var(--accent-2));
    display: inline-block;
    animation: pulseDot 2s infinite;
}

/* ---------- Alerts (answer / info / warnings) ---------- */
[data-testid="stAlert"] {
    border-radius: 16px;
    border: 1px solid var(--border);
}
[data-testid="stAlert"] p {
    font-size: 15.5px;
    line-height: 1.65;
}

/* ---------- Sources ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: #FFFFFF;
    margin-bottom: 10px;
    transition: all 0.2s ease;
}
[data-testid="stExpander"]:hover {
    border-color: #A5B4FC !important;
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.10);
}
[data-testid="stExpander"] summary {
    font-weight: 600;
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #F8FAFF, #F3F0FF);
    border: 1px solid #E0E7FF;
    border-radius: 14px;
    padding: 10px 14px;
}
[data-testid="stMetricValue"] {
    font-weight: 800;
    color: var(--accent-dark);
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF, #F8F9FF);
    border-right: 1px solid var(--border);
}
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
}
.sidebar-brand-icon {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: linear-gradient(120deg, var(--accent), var(--accent-2));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 19px;
    box-shadow: 0 6px 14px rgba(99, 102, 241, 0.35);
}
.sidebar-brand-text {
    font-size: 15px;
    font-weight: 800;
    color: var(--text);
    line-height: 1.2;
}
.sidebar-brand-sub {
    font-size: 11px;
    color: var(--muted);
    font-weight: 500;
}
.sidebar-heading {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--accent-dark);
    margin: 4px 0 12px 0;
}

/* ---------- Pipeline timeline ---------- */
.pipe-step {
    display: flex;
    align-items: center;
    gap: 12px;
    position: relative;
    padding-bottom: 14px;
    animation: slideIn 0.5s ease both;
}
.pipe-step:nth-child(1) { animation-delay: 0.05s; }
.pipe-step:nth-child(2) { animation-delay: 0.12s; }
.pipe-step:nth-child(3) { animation-delay: 0.19s; }
.pipe-step:nth-child(4) { animation-delay: 0.26s; }
.pipe-step:nth-child(5) { animation-delay: 0.33s; }
.pipe-step:nth-child(6) { animation-delay: 0.40s; }
.pipe-step:nth-child(7) { animation-delay: 0.47s; }

.pipe-step:not(:last-child)::after {
    content: "";
    position: absolute;
    left: 14px;
    top: 30px;
    width: 2px;
    height: calc(100% - 28px);
    background: var(--border);
}
.pipe-step.done:not(:last-child)::after {
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}
.pipe-num {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: #F1F5F9;
    border: 1.5px solid var(--border);
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    z-index: 1;
    transition: all 0.25s ease;
}
.pipe-step.done .pipe-num {
    background: linear-gradient(120deg, var(--accent), var(--accent-2));
    border-color: transparent;
    color: #FFFFFF;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
}
.pipe-step:hover .pipe-num {
    transform: scale(1.12);
}
.pipe-label {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
}

/* ---------- Footer ---------- */
.app-footer {
    text-align: center;
    color: #94A3B8;
    font-size: 13px;
    padding: 28px 0 8px 0;
    border-top: 1px solid var(--border);
    margin-top: 40px;
}
.app-footer strong {
    color: var(--muted);
    font-weight: 700;
}
.footer-tech {
    margin-top: 6px;
    letter-spacing: 0.02em;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# UI helper (visual only, no application logic)
# ---------------------------------------------------------

def render_status_card(name, pages, chunks):
    """Render the document processing status card."""

    with st.container(border=True):

        st.markdown(
            '<span class="badge badge-success">'
            '<span class="live-dot"></span> Processed</span>'
            '<div class="status-title">'
            '✅ Document processed successfully!</div>',
            unsafe_allow_html=True
        )

        col_name, col_pages, col_chunks, col_status = st.columns(
            [2, 1, 1, 1.4]
        )

        with col_name:
            st.markdown(
                '<div class="field-label">Document</div>',
                unsafe_allow_html=True
            )
            st.text(name)

        with col_pages:
            st.markdown(
                '<div class="field-label">Pages</div>',
                unsafe_allow_html=True
            )
            st.text(pages)

        with col_chunks:
            st.markdown(
                '<div class="field-label">Chunks</div>',
                unsafe_allow_html=True
            )
            st.text(chunks)

        with col_status:
            st.markdown(
                '<div class="field-label">Status</div>',
                unsafe_allow_html=True
            )
            st.text("Ready for questions")


# =========================================================
# 4. Session state
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_pages" not in st.session_state:
    st.session_state.document_pages = 0

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = 0

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# 5. Load AI components
# =========================================================

@st.cache_resource
def load_embedding_model():

    return create_embeddings()


@st.cache_resource
def load_language_model():

    return create_llm()


try:

    embeddings = load_embedding_model()
    llm = load_language_model()

except Exception as error:

    st.error(
        f"Failed to initialize AI components: {error}"
    )

    st.stop()


# =========================================================
# 6. Sidebar
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="sidebar-brand-icon">📄</div>'
        '<div>'
        '<div class="sidebar-brand-text">DocIntel AI</div>'
        '<div class="sidebar-brand-sub">RAG Control Panel</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ----- Document -----
    st.markdown(
        '<div class="sidebar-heading">📚 Document</div>',
        unsafe_allow_html=True
    )

    if st.session_state.document_name:

        with st.container(border=True):

            st.markdown(
                '<span class="badge badge-success">'
                '<span class="live-dot"></span> Active document</span>',
                unsafe_allow_html=True
            )

            st.write("Current document:")

            st.code(
                st.session_state.document_name
            )

            col_pages, col_chunks = st.columns(2)

            with col_pages:
                st.metric(
                    "📄 Pages",
                    st.session_state.document_pages
                )

            with col_chunks:
                st.metric(
                    "🧩 Chunks",
                    st.session_state.document_chunks
                )

    else:

        st.info(
            "No document uploaded."
        )

    st.divider()

    # ----- RAG Pipeline -----
    st.markdown(
        '<div class="sidebar-heading">🔎 RAG Pipeline</div>',
        unsafe_allow_html=True
    )

    pipeline_steps = [
        "PDF Extraction",
        "Text Cleaning",
        "Text Chunking",
        "Embeddings",
        "ChromaDB",
        "Semantic Search",
        "Gemini LLM"
    ]

    step_state = (
        "done" if st.session_state.document_name else ""
    )

    pipeline_html = "".join(
        f'<div class="pipe-step {step_state}">'
        f'<div class="pipe-num">{number}</div>'
        f'<div class="pipe-label">{label}</div>'
        '</div>'
        for number, label in enumerate(pipeline_steps, start=1)
    )

    st.markdown(pipeline_html, unsafe_allow_html=True)

    st.divider()

    # ----- Settings -----
    st.markdown(
        '<div class="sidebar-heading">⚙️ Settings</div>',
        unsafe_allow_html=True
    )

    retrieval_count = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=6,
        value=4
    )

    st.divider()

    if st.button(
        "🗑️ Clear Document",
        use_container_width=True
    ):

        st.session_state.vectorstore = None
        st.session_state.document_name = None
        st.session_state.document_pages = 0
        st.session_state.document_chunks = 0
        st.session_state.document_hash = None
        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# 7. Header
# =========================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-icon">📄</div>'
    '<div class="hero-text">'
    '<div class="hero-title">AI Document Intelligence Assistant</div>'
    '<div class="hero-subtitle">Ask questions about your documents '
    'using Retrieval-Augmented Generation (RAG).</div>'
    '<div class="hero-tags">'
    '<span class="hero-tag">Semantic Search</span>'
    '<span class="hero-tag">Grounded Answers</span>'
    '<span class="hero-tag">Source Citations</span>'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 8. Document upload
# =========================================================

with st.container(border=True):

    st.markdown(
        '<div class="card-title">📤 Upload your document</div>'
        '<div class="card-subtitle">'
        'Upload a PDF to start asking questions.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload a text-based PDF document."
    )

    # -----------------------------------------------------
    # Process PDF
    # -----------------------------------------------------

    if uploaded_file is not None:

        # Calculate file hash
        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()

        # Process only when a new PDF is uploaded
        if file_hash != st.session_state.document_hash:

            with st.spinner(
                "Processing document..."
            ):

                try:

                    # Create upload directory
                    upload_directory = Path(
                        "uploaded_documents"
                    )

                    upload_directory.mkdir(
                        exist_ok=True
                    )

                    # Save uploaded PDF
                    pdf_path = (
                        upload_directory
                        / uploaded_file.name
                    )

                    with open(
                        pdf_path,
                        "wb"
                    ) as file:

                        file.write(file_bytes)

                    # Load and split PDF
                    documents, chunks = (
                        load_and_split_pdf(
                            str(pdf_path)
                        )
                    )

                    # Create vector database
                    vectorstore = (
                        create_vectorstore(
                            chunks,
                            embeddings
                        )
                    )

                    # Store application state
                    st.session_state.vectorstore = (
                        vectorstore
                    )

                    st.session_state.document_name = (
                        uploaded_file.name
                    )

                    st.session_state.document_pages = (
                        len(documents)
                    )

                    st.session_state.document_chunks = (
                        len(chunks)
                    )

                    st.session_state.document_hash = (
                        file_hash
                    )

                    st.session_state.chat_history = []

                    render_status_card(
                        uploaded_file.name,
                        len(documents),
                        len(chunks)
                    )

                except Exception as error:

                    st.error(
                        "Unable to process this PDF."
                    )

                    st.exception(error)

        else:

            render_status_card(
                uploaded_file.name,
                st.session_state.document_pages,
                st.session_state.document_chunks
            )


# =========================================================
# 9. Question area
# =========================================================

st.write("")

with st.container(border=True):

    st.markdown(
        '<div class="card-title">💬 Ask your document</div>'
        '<div class="card-subtitle">Ask a natural-language question '
        'and get an answer grounded in your uploaded PDF.</div>',
        unsafe_allow_html=True
    )

    question = st.text_input(
        "Enter your question:",
        placeholder=(
            "Example: How many annual leave days "
            "do employees get?"
        )
    )

    # -----------------------------------------------------
    # Ask AI
    # -----------------------------------------------------

    if st.button(
        "🔍 Ask AI",
        type="primary",
        use_container_width=True
    ):

        if st.session_state.vectorstore is None:

            st.warning(
                "Please upload a PDF before asking a question."
            )

        elif not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching document and generating answer..."
            ):

                try:

                    # Retrieve relevant chunks
                    results = retrieve_documents(
                        st.session_state.vectorstore,
                        question,
                        k=retrieval_count
                    )

                    # Generate grounded answer
                    answer = generate_answer(
                        llm,
                        question,
                        results
                    )

                    # Save chat history
                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "sources": results
                        }
                    )

                except Exception as error:

                    st.error(
                        "Something went wrong while "
                        "generating the answer."
                    )

                    st.exception(error)


# =========================================================
# 10. Answer area (conversation)
# =========================================================

if st.session_state.chat_history:

    st.markdown(
        '<div class="section-heading">🤖 Answers</div>',
        unsafe_allow_html=True
    )

    for index, chat in enumerate(
        reversed(
            st.session_state.chat_history
        ),
        start=1
    ):

        with st.container(border=True):

            st.markdown(
                f'<span class="badge badge-accent">'
                f'Question {index}</span>',
                unsafe_allow_html=True
            )

            # ----- User question -----
            st.markdown(
                '<div class="section-label">👤 User Question</div>',
                unsafe_allow_html=True
            )

            with st.container(border=True):

                st.write(
                    chat["question"]
                )

            # ----- AI answer -----
            st.markdown(
                '<div class="answer-label">'
                '<span class="answer-dot"></span>AI Answer</div>',
                unsafe_allow_html=True
            )

            st.success(
                chat["answer"]
            )

            # =================================================
            # 11. Sources
            # =================================================

            st.markdown(
                '<div class="section-label">📌 Sources</div>',
                unsafe_allow_html=True
            )

            displayed_sources = set()

            for source_index, result in enumerate(
                chat["sources"],
                start=1
            ):

                page = result.metadata.get(
                    "page_number",
                    result.metadata.get("page", 0) + 1
                )

                source = result.metadata.get(
                    "source",
                    st.session_state.document_name
                )

                source_key = (
                    source,
                    page,
                    result.page_content[:100]
                )

                if source_key in displayed_sources:
                    continue

                displayed_sources.add(
                    source_key
                )

                with st.expander(
                    f"Source {source_index} — Page {page}"
                ):

                    col_source, col_page, col_file = st.columns(
                        [1, 1, 3]
                    )

                    with col_source:
                        st.markdown(
                            '<div class="field-label">Source</div>',
                            unsafe_allow_html=True
                        )
                        st.text(f"#{source_index}")

                    with col_page:
                        st.markdown(
                            '<div class="field-label">Page</div>',
                            unsafe_allow_html=True
                        )
                        st.text(page)

                    with col_file:
                        st.markdown(
                            '<div class="field-label">File</div>',
                            unsafe_allow_html=True
                        )
                        st.text(Path(source).name)

                    st.write(
                        "**Retrieved content:**"
                    )

                    st.write(
                        result.page_content
                    )


# =========================================================
# 12. Footer
# =========================================================

st.markdown(
    '<div class="app-footer">'
    '<strong>AI Document Intelligence Assistant</strong>'
    '<div class="footer-tech">'
    'Python • LangChain • ChromaDB • '
    'Hugging Face • Gemini • Streamlit'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)