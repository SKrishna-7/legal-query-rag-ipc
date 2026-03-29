import streamlit as st
import json
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
from enum import Enum
from dataclasses import asdict
import PyPDF2

# Import Groq for the chat box
try:
    from groq import Groq
except ImportError:
    Groq = None

# Import Transformers for local models
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# Set page config
st.set_page_config(
    page_title="Legal IPC-RAG | FIR Audit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Metric Cards */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border-top: 4px solid #3b82f6;
        text-align: center;
    }
    .metric-card.high-risk {
        border-top-color: #ef4444;
        background-color: #fef2f2;
    }
    .metric-card.medium-risk {
        border-top-color: #f59e0b;
        background-color: #fffbeb;
    }
    .metric-card.low-risk {
        border-top-color: #10b981;
        background-color: #ecfdf5;
    }
    .metric-title {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-value.high { color: #b91c1c; }
    .metric-value.medium { color: #b45309; }
    .metric-value.low { color: #047857; }

    /* Chatbox styles */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        max-width: 85%;
        line-height: 1.5;
        font-size: 1rem;
    }
    .chat-message.user {
        background-color: #3b82f6;
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 0.25rem;
    }
    .chat-message.bot {
        background-color: white;
        color: #1e293b;
        align-self: flex-start;
        border-bottom-left-radius: 0.25rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    .chat-role {
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        opacity: 0.8;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Import backend modules
from src.ipc_cam.ipc_cam import IPCContextualAlignmentModule, AlignmentStatus, SatisfactionStatus
from src.rationale.legal_rationale_generator import LegalRationaleGenerator
from src.misuse_detection.misuse_engine import MisuseRiskAssessmentEngine, RiskLevel
from src.generation.citizen_response_generator import CitizenResponseGenerator
from src.preprocessing.ipc_extractor import IPCSectionExtractor

# Helper function to extract text from PDF
def extract_text_from_pdf(file_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(file_bytes)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Initialize Session State
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'report_data' not in st.session_state:
    st.session_state.report_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding-bottom: 20px;'>
            <img src="https://img.icons8.com/external-flatart-icons-flat-flatarticons/128/external-law-law-and-justice-flatart-icons-flat-flatarticons-1.png" width="80" style="margin-bottom: 10px;">
            <h2 style='margin: 0; color: #1e3a8a;'>FIR Audit System</h2>
            <p style='color: #64748b; font-size: 0.9rem;'>AI-Powered Legal Verification</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📄 1. Upload Document")
    uploaded_file = st.file_uploader("Select FIR (PDF or Text)", type=["pdf", "txt"], label_visibility="collapsed")
    
    with st.expander("⚙️ Advanced Settings", expanded=False):
        inference_source = st.radio("Inference Source", ["Cloud (Groq)", "Local (Llama 3.2 3B)"], index=0)
        use_local = (inference_source == "Local (Llama 3.2 3B)")
        local_model_path = st.text_input("Local Model Path", value=r"C:\Users\PATHU\.cache\huggingface\hub\models--meta-llama--Llama-3.2-3B")
        
        st.markdown("**Manual Section Override:** (Optional, comma-separated)")
        manual_sections = st.text_input("e.g. 302, 323", placeholder="Leave blank to auto-extract")
        language = st.selectbox("Response Language", ["English", "Hindi", "Tamil", "Bengali"])
        st.session_state.groq_key = st.text_input("Groq API Key", value="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", type="password")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Analyze FIR Document", use_container_width=True, type="primary"):
        if uploaded_file is not None:
            with st.spinner("Processing document & verifying legal context..."):
                try:
                    # File Processing
                    if uploaded_file.name.lower().endswith(".pdf"):
                        narrative = extract_text_from_pdf(uploaded_file)
                    else:
                        try:
                            narrative = uploaded_file.read().decode("utf-8")
                        except UnicodeDecodeError:
                            uploaded_file.seek(0)
                            narrative = uploaded_file.read().decode("latin-1", errors="replace")
                    
                    # Section Extraction
                    if manual_sections.strip():
                        applied_sections = [s.strip() for s in manual_sections.split(',')]
                    else:
                        with st.spinner("Auto-extracting IPC sections..."):
                            extractor = IPCSectionExtractor()
                            applied_sections = extractor.extract_mentioned_sections(narrative)
                            if not applied_sections:
                                st.error("No IPC sections were automatically detected. This may be a non-IPC FIR (e.g., PC Act, NDPS). Please enter the sections manually in the Advanced Settings sidebar.")
                                st.stop()
                    
                    fir_num = "FIR-" + uploaded_file.name.split('.')[0]

                    # Run Backend Pipeline
                    cam = IPCContextualAlignmentModule(
                        groq_api_key=st.session_state.groq_key,
                        use_local=use_local,
                        local_model_path=local_model_path
                    )
                    rationale_gen = LegalRationaleGenerator()
                    misuse_engine = MisuseRiskAssessmentEngine()
                    response_gen = CitizenResponseGenerator(
                        api_key=st.session_state.groq_key,
                        use_local=use_local,
                        local_model_path=local_model_path
                    )

                    # 1. CAM
                    cam_report = cam.generate_full_cam_report(fir_num, applied_sections, narrative)
                    
                    # 2. Rationale
                    fir_rationale = rationale_gen.generate_fir_level_rationale(cam_report)
                    
                    # 3. Misuse
                    misuse_report = misuse_engine.generate_misuse_report(fir_num, cam_report, fir_rationale)
                    
                    # 4. Final Citizen Response
                    final_response = response_gen.generate_full_analysis_response(
                        fir_num, applied_sections, cam_report, misuse_report, fir_rationale, language=language
                    )

                    st.session_state.report_data = {
                        "cam": cam_report,
                        "rationale": fir_rationale,
                        "misuse": misuse_report,
                        "final": final_response,
                        "narrative": narrative,
                        "sections": applied_sections
                    }
                    
                    # Add initial context to chat history
                    st.session_state.chat_history = [
                        {"role": "assistant", "content": f"Hello! I have completed the legal audit for **{fir_num}**. I detected **{len(applied_sections)}** IPC section(s). The overall misuse risk is evaluated as **{fir_rationale.overall_misuse_risk}**. Feel free to ask me to explain any part of the report!"}
                    ]
                    
                    st.session_state.analysis_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Critical Backend Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.error("Please upload an FIR document first.")
            
    st.markdown("---")
    st.caption("Legal IPC-RAG Prototype v1.0")

# Main Dashboard
if not st.session_state.analysis_done:
    st.markdown("""
        <div class="main-header">
            <h1>⚖️ Legal IPC-RAG</h1>
            <p>Intelligent FIR Audit & Police Misuse Detection Framework</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 Please upload an FIR document in the sidebar to begin the AI-driven legal verification process.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); height: 100%;'>
            <h3 style='color: #1e3a8a; margin-top: 0;'>🔍 Contextual Alignment</h3>
            <p style='color: #475569;'>Verifies if the narrative facts in the FIR logically satisfy the 'essential legal ingredients' of the applied IPC sections using LLMs and NLI.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); height: 100%;'>
            <h3 style='color: #dc2626; margin-top: 0;'>🚩 Misuse Detection</h3>
            <p style='color: #475569;'>Identifies patterns of police overcharging, manipulation of non-bailable sections, and missing elements of criminal intent (Mens Rea).</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); height: 100%;'>
            <h3 style='color: #059669; margin-top: 0;'>📖 Explainable AI</h3>
            <p style='color: #475569;'>Generates plain-language, structured reports for citizens to understand their rights, and provides detailed rationales for legal professionals.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    data = st.session_state.report_data
    
    # Header Section
    st.markdown(f"""
        <div class="main-header" style="padding: 1.5rem; margin-bottom: 1.5rem;">
            <h2 style="color: white; margin: 0;">Audit Report: {data['cam'].fir_number}</h2>
            <p style="margin-top: 0.25rem;">Analyzed Sections: <strong>{', '.join(data['sections'])}</strong></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Risk Metrics
    m1, m2, m3 = st.columns(3)
    
    risk_lvl = data['misuse'].risk_assessment['risk_level']
    risk_class = "high-risk" if risk_lvl == "HIGH" else ("medium-risk" if risk_lvl == "MEDIUM" else "low-risk")
    risk_val_class = "high" if risk_lvl == "HIGH" else ("medium" if risk_lvl == "MEDIUM" else "low")
    
    score = data['misuse'].risk_assessment['overall_score'] * 100
    alignment = sum(s.alignment_score for s in data['cam'].sections_evaluated)/max(1, len(data['cam'].sections_evaluated)) * 100
    
    with m1:
        st.markdown(f"""
            <div class="metric-card {risk_class}">
                <div class="metric-title">Detected Risk Level</div>
                <div class="metric-value {risk_val_class}">{risk_lvl}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Overcharging Probability</div>
                <div class="metric-value">{score:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Factual Alignment</div>
                <div class="metric-value">{alignment:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Citizen Report", 
        "💬 Legal Assistant", 
        "📊 Section Verification", 
        "🚩 Malpractice Patterns", 
        "📝 FIR Narrative"
    ])
    
    with tab1:
        st.markdown("<div style='background: white; padding: 30px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        st.markdown(data['final'].summary_markdown)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button("📥 Download Report (PDF/MD)", data['final'].summary_markdown, file_name=f"{data['cam'].fir_number}_audit.md", type="primary", use_container_width=True)

    with tab2:
        st.markdown("<div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-height: 400px;'>", unsafe_allow_html=True)
        st.markdown("### Interactive Legal Guidance")
        st.caption("Ask questions specific to this FIR audit. The assistant has full context of the uploaded document and analysis.")
        
        # Display chat history
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            css_class = "user" if message["role"] == "user" else "bot"
            role_name = "You" if message["role"] == "user" else "AI Legal Assistant"
            st.markdown(f"""
                <div class='chat-message {css_class}'>
                    <div class='chat-role'>{role_name}</div>
                    <div>{message['content']}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Chat input
        st.markdown("---")
        user_query = st.chat_input("E.g., Can I get bail for these charges?", key="chat_input")
        
        # Determine if we should use local for chat
        chat_use_local = (inference_source == "Local (Llama 3.2 3B)")
        
        if user_query:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.rerun() # Fast rerun to show user message instantly
            
        elif st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            # If the last message is from user, generate response
            system_prompt = f"""
            You are a legal awareness assistant in India. Discussing an FIR audit report with a citizen.
            FIR Narrative: {data['narrative'][:500]}...
            Applied Sections: {data['sections']}
            Risk Assessment: {data['misuse'].risk_assessment['risk_level']}
            
            Answer clearly, concisely, based on IPC/CrPC. NO formal legal advice. Use formatting.
            """
            
            with st.spinner("AI is typing..."):
                if chat_use_local:
                    try:
                        from transformers import pipeline
                        local_pipe = pipeline("text-generation", model=local_model_path, device_map="auto")
                        
                        full_chat_prompt = f"System: {system_prompt}\n\n"
                        for msg in st.session_state.chat_history[-4:]:
                            role = "User" if msg["role"] == "user" else "Assistant"
                            full_chat_prompt += f"{role}: {msg['content']}\n"
                        full_chat_prompt += "Assistant: "
                        
                        outputs = local_pipe(full_chat_prompt, max_new_tokens=512, do_sample=True, temperature=0.3, return_full_text=False)
                        bot_response = outputs[0]["generated_text"]
                        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Local Model Chat Error: {e}")
                
                elif Groq and st.session_state.groq_key:
                    try:
                        client = Groq(api_key=st.session_state.groq_key)
                        messages = [{"role": "system", "content": system_prompt}]
                        
                        for msg in st.session_state.chat_history[-6:]: # Last 6 msgs
                            messages.append({"role": msg["role"], "content": msg["content"]})
                            
                        completion = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=messages,
                            temperature=0.3
                        )
                        
                        bot_response = completion.choices[0].message.content
                        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"API Error: {e}")
                else:
                    st.error("No inference source available (Groq key missing or transformers not installed).")
            
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        if not data['cam'].sections_evaluated:
            st.warning("No sections were evaluated.")
        else:
            for section_res in data['cam'].sections_evaluated:
                st.markdown(f"<div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px;'>", unsafe_allow_html=True)
                
                status_val = section_res.alignment_status.value if hasattr(section_res.alignment_status, 'value') else str(section_res.alignment_status)
                
                col_chart, col_table = st.columns([1, 1.5])
                
                with col_chart:
                    st.markdown(f"<h3 style='margin-top:0;'>Section {section_res.section_number}</h3>", unsafe_allow_html=True)
                    st.markdown(f"**Status:** {status_val}")
                    
                    if section_res.ingredient_scores:
                        labels = [i.ingredient[:25]+"..." for i in section_res.ingredient_scores]
                        values = [1.0 if i.satisfaction_status == SatisfactionStatus.SATISFIED else (0.5 if i.satisfaction_status == SatisfactionStatus.PARTIALLY_SATISFIED else 0.1) for i in section_res.ingredient_scores]
                        
                        df = pd.DataFrame(dict(r=values, theta=labels))
                        fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r=[0,1])
                        fig.update_traces(fill='toself', fillcolor='rgba(59, 130, 246, 0.2)', line_color='#3b82f6')
                        fig.update_layout(polar=dict(radialaxis=dict(visible=False)), margin=dict(l=40, r=40, t=20, b=20), height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No specific ingredients mapped.")
                        
                with col_table:
                    st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
                    if section_res.ingredient_scores:
                        for i in section_res.ingredient_scores:
                            stat = i.satisfaction_status.value if hasattr(i.satisfaction_status, 'value') else str(i.satisfaction_status)
                            icon = "✅" if stat == "SATISFIED" else ("⚠️" if stat == "PARTIALLY_SATISFIED" else "❌")
                            st.markdown(f"**{icon} {i.ingredient}**")
                            st.caption(f"Reasoning: {i.reasoning}")
                            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("<div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        st.markdown("### Overcharging & Malpractice Flags")
        
        if not data['misuse'].risk_assessment['misuse_patterns']:
            st.success("No significant misuse patterns detected by the engine.")
        else:
            for pattern in data['misuse'].risk_assessment['misuse_patterns']:
                st.error(f"**{pattern['misuse_type']}**")
                st.write(pattern['description'])
                st.caption(f"Detected in sections: {', '.join(pattern['affected_sections'])} | Severity Weight: {pattern['severity_multiplier']}x")
                st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab5:
        st.markdown("<div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        st.markdown("### Document Extract")
        st.text_area("Parsed Narrative", data['narrative'], height=400, disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)
