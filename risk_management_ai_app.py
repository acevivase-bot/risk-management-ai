
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import openai
import requests
import json
import hashlib
import time
from datetime import datetime, timedelta
from io import StringIO

# ============= KONFIGURASI HALAMAN =============
st.set_page_config(
    page_title="Data Analysis for Risk Management",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CSS STYLING =============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #DC3545, #FFC107);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card-critical {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-card-high {
        background: linear-gradient(135deg, #fd7e14 0%, #e55a2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-card-medium {
        background: linear-gradient(135deg, #ffc107 0%, #ffb300 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-card-low {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-msg {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-msg {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem 1.25rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .iso-framework {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
    }
    .mitigation-box {
        background: #e7f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============= CHAT HISTORY MANAGEMENT =============
def manage_risk_chat_history(uploaded_file):
    """Clear chat history when risk dataset changes"""
    if uploaded_file is not None:
        dataset_key = f"{uploaded_file.name}_{uploaded_file.size}"
    else:
        dataset_key = "no_risk_dataset"

    if "current_risk_dataset" not in st.session_state:
        st.session_state.current_risk_dataset = dataset_key
        st.session_state.risk_messages = []
    elif st.session_state.current_risk_dataset != dataset_key:
        st.session_state.current_risk_dataset = dataset_key
        st.session_state.risk_messages = []
        st.info("⚠️ Risk analysis chat cleared for new dataset")

def add_risk_clear_button():
    """Add manual clear chat button for risk management"""
    if st.session_state.get("risk_messages", []):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("🗑️ Clear Risk Chat", use_container_width=True):
                st.session_state.risk_messages = []
                st.success("Risk chat history cleared!")
                st.rerun()
        with col3:
            chat_count = len(st.session_state.risk_messages)
            st.info(f"💬 {chat_count} messages")

# ============= RISK ANALYSIS FUNCTIONS =============
def get_risk_openai_response(prompt, data_context="", api_key="", model="gpt-3.5-turbo"):
    """Generate risk-specific response from OpenAI API"""
    try:
        if not api_key:
            return "⚠️ OpenAI API key tidak dikonfigurasi. Silakan konfigurasi di sidebar."

        openai.api_key = api_key

        system_prompt = """Anda adalah AI Risk Management Expert yang mengkhususkan diri dalam analisis dan manajemen risiko sesuai standar ISO 31000. 

        Fokus utama Anda:
        - Risk Assessment berdasarkan Impact dan Likelihood
        - Risk Treatment Strategy (Mitigate, Transfer, Accept, Avoid)
        - Control Effectiveness Analysis
        - Risk Owner accountability
        - Compliance dan regulatory requirements

        Berikan jawaban yang:
        - Praktis dan actionable untuk risk manager
        - Sesuai dengan best practices ISO 31000
        - Fokus pada mitigasi konkret
        - Tidak perlu saran visualisasi (kecuali diminta spesifik)
        - Singkat namun comprehensive

        Jawab dalam bahasa Indonesia dengan gaya profesional risk management."""

        # Risk-specific prompt berdasarkan query
        if any(keyword in prompt.lower() for keyword in ['impact', 'dampak', 'konsekuensi']):
            context_prompt = """
            User bertanya tentang IMPACT/DAMPAK risiko.

            Berikan:
            1. Ringkasan dampak yang teridentifikasi
            2. Kategori severity dampak
            3. Mitigasi spesifik untuk mengurangi impact
            4. Control yang dapat mengurangi consequences
            """
        elif any(keyword in prompt.lower() for keyword in ['cause', 'penyebab', 'root cause']):
            context_prompt = """
            User bertanya tentang CAUSE/PENYEBAB risiko.

            Berikan:
            1. Ringkasan root causes yang teridentifikasi
            2. Preventive controls untuk eliminate causes
            3. Mitigasi upstream untuk mencegah occurrence
            4. Monitoring system untuk early warning
            """
        elif any(keyword in prompt.lower() for keyword in ['likelihood', 'probability', 'kemungkinan']):
            context_prompt = """
            User bertanya tentang LIKELIHOOD/PROBABILITY risiko.

            Berikan:
            1. Assessment probabilitas occurrence
            2. Factors yang mempengaruhi likelihood
            3. Mitigasi untuk reduce probability
            4. Monitoring indicators
            """
        elif any(keyword in prompt.lower() for keyword in ['control', 'kontrol', 'mitigasi']):
            context_prompt = """
            User bertanya tentang CONTROL/MITIGASI risiko.

            Berikan:
            1. Effectiveness assessment controls yang ada
            2. Gap analysis dan improvement areas
            3. Additional controls yang diperlukan
            4. Control testing dan monitoring
            """
        elif any(keyword in prompt.lower() for keyword in ['priority', 'prioritas', 'urgent']):
            context_prompt = """
            User bertanya tentang PRIORITIZATION risiko.

            Berikan:
            1. Risk ranking berdasarkan severity
            2. Action priority berdasarkan risk rating
            3. Resource allocation strategy
            4. Timeline untuk treatment
            """
        else:
            context_prompt = """
            User bertanya tentang aspek umum risk management.

            Berikan:
            1. Risk analysis summary
            2. Key findings dan insights
            3. Priority actions untuk risk treatment
            4. Next steps recommendations
            """

        full_prompt = f"""
        Risk Data Context: {data_context}

        User Query: {prompt}

        {context_prompt}

        Format jawaban:
        - Ringkasan (2-3 kalimat)
        - Key Points (bullet points)
        - Mitigasi/Action Items (specific recommendations)
        """

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            return response.choices[0].message.content

        except ImportError:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        if "no longer supported" in error_msg:
            return """❌ Error OpenAI API: Versi library openai yang terinstall tidak kompatibel.

Solusi:
1. Update requirements.txt dengan: openai>=1.0.0
2. Atau gunakan versi lama: openai==0.28.0

Untuk sementara, silakan gunakan Perplexity API sebagai alternatif."""
        else:
            return f"❌ Error OpenAI: {error_msg}"

def get_risk_perplexity_response(prompt, data_context="", api_key="", model="llama-3.1-sonar-small-128k-online"):
    """Generate risk-specific response from Perplexity API"""
    try:
        if not api_key:
            return "⚠️ Perplexity API key tidak dikonfigurasi. Silakan konfigurasi di sidebar."

        url = "https://api.perplexity.ai/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        system_prompt = """Anda adalah AI Risk Management Expert yang mengkhususkan diri dalam analisis dan manajemen risiko sesuai standar ISO 31000. 

        Fokus utama Anda:
        - Risk Assessment berdasarkan Impact dan Likelihood
        - Risk Treatment Strategy (Mitigate, Transfer, Accept, Avoid)
        - Control Effectiveness Analysis
        - Risk Owner accountability
        - Compliance dan regulatory requirements

        Berikan jawaban yang:
        - Praktis dan actionable untuk risk manager
        - Sesuai dengan best practices ISO 31000
        - Fokus pada mitigasi konkret
        - Tidak perlu saran visualisasi (kecuali diminta spesifik)
        - Singkat namun comprehensive

        Jawab dalam bahasa Indonesia dengan gaya profesional risk management."""

        # Risk-specific prompt logic (same as OpenAI)
        if any(keyword in prompt.lower() for keyword in ['impact', 'dampak', 'konsekuensi']):
            context_prompt = """
            User bertanya tentang IMPACT/DAMPAK risiko.
            Berikan: Ringkasan dampak, severity kategori, mitigasi spesifik, control untuk consequences.
            """
        elif any(keyword in prompt.lower() for keyword in ['cause', 'penyebab', 'root cause']):
            context_prompt = """
            User bertanya tentang CAUSE/PENYEBAB risiko.
            Berikan: Root causes, preventive controls, upstream mitigasi, early warning monitoring.
            """
        elif any(keyword in prompt.lower() for keyword in ['control', 'kontrol', 'mitigasi']):
            context_prompt = """
            User bertanya tentang CONTROL/MITIGASI risiko.
            Berikan: Control effectiveness, gap analysis, additional controls, monitoring strategy.
            """
        else:
            context_prompt = """
            User bertanya tentang risk management secara umum.
            Berikan: Risk analysis summary, key findings, priority actions, next steps.
            """

        full_prompt = f"""
        Risk Data Context: {data_context}

        User Query: {prompt}

        {context_prompt}

        Format: Ringkasan singkat, Key Points, Mitigasi/Action Items.
        """

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"❌ Error Perplexity: {response.status_code} - {response.text}"

    except Exception as e:
        return f"❌ Error Perplexity: {str(e)}"

def analyze_risk_data(data):
    """Analyze risk data and provide risk-specific insights"""
    try:
        risk_analysis = {}

        # Risk severity analysis
        if 'Risk_Rating' in data.columns:
            risk_analysis['total_risks'] = len(data)
            risk_analysis['avg_risk_rating'] = data['Risk_Rating'].mean()

            # Categorize risks
            critical_risks = data[data['Risk_Rating'] >= 20]['Risk_Rating'].count()
            high_risks = data[(data['Risk_Rating'] >= 15) & (data['Risk_Rating'] < 20)]['Risk_Rating'].count()
            medium_risks = data[(data['Risk_Rating'] >= 10) & (data['Risk_Rating'] < 15)]['Risk_Rating'].count()
            low_risks = data[data['Risk_Rating'] < 10]['Risk_Rating'].count()

            risk_analysis['critical_risks'] = critical_risks
            risk_analysis['high_risks'] = high_risks
            risk_analysis['medium_risks'] = medium_risks
            risk_analysis['low_risks'] = low_risks

        # Status analysis
        if 'Status' in data.columns:
            status_counts = data['Status'].value_counts()
            risk_analysis['open_risks'] = status_counts.get('Open', 0)
            risk_analysis['in_progress_risks'] = status_counts.get('In Progress', 0)
            risk_analysis['closed_risks'] = status_counts.get('Closed', 0)

        # Top threats and assets
        if 'Threat' in data.columns:
            risk_analysis['top_threats'] = data['Threat'].value_counts().head(5).to_dict()

        if 'Asset' in data.columns:
            risk_analysis['top_assets_at_risk'] = data['Asset'].value_counts().head(5).to_dict()

        # Risk owners
        if 'Risk_Owner' in data.columns:
            risk_analysis['risk_owners'] = data['Risk_Owner'].value_counts().to_dict()

        return risk_analysis

    except Exception as e:
        return {"error": str(e)}

def create_risk_visualization(data, chart_type, x_axis, y_axis=None, color=None):
    """Create risk-specific visualizations"""
    try:
        risk_colors = {
            'Critical': '#dc3545',
            'High': '#fd7e14',
            'Medium': '#ffc107',
            'Low': '#28a745',
            'Very Low': '#17a2b8'
        }

        if chart_type == 'risk_matrix':
            # Risk matrix visualization
            if 'Impact' in data.columns and 'Likelihood' in data.columns:
                fig = px.scatter(data, x='Likelihood', y='Impact', 
                               color='Risk_Rating' if 'Risk_Rating' in data.columns else None,
                               size='Risk_Rating' if 'Risk_Rating' in data.columns else None,
                               hover_data=['Risk_ID', 'Asset', 'Threat'] if all(col in data.columns for col in ['Risk_ID', 'Asset', 'Threat']) else None,
                               title="Risk Matrix - Impact vs Likelihood",
                               color_continuous_scale='Reds')

                fig.update_layout(
                    xaxis_title="Likelihood",
                    yaxis_title="Impact",
                    showlegend=True
                )

                return fig

        elif chart_type == 'risk_distribution':
            # Risk severity distribution
            if 'Risk_Rating' in data.columns:
                # Create risk severity categories
                def categorize_risk(rating):
                    if rating >= 20:
                        return 'Critical'
                    elif rating >= 15:
                        return 'High'
                    elif rating >= 10:
                        return 'Medium'
                    elif rating >= 5:
                        return 'Low'
                    else:
                        return 'Very Low'

                data['Risk_Category'] = data['Risk_Rating'].apply(categorize_risk)

                fig = px.pie(data, names='Risk_Category', 
                           title="Risk Distribution by Severity",
                           color='Risk_Category',
                           color_discrete_map=risk_colors)

                return fig

        elif chart_type == 'top_risks':
            # Top risks by rating
            if 'Risk_Rating' in data.columns:
                top_risks = data.nlargest(10, 'Risk_Rating')

                fig = px.bar(top_risks, x='Risk_Rating', y='Risk_ID',
                           orientation='h',
                           title="Top 10 Highest Rated Risks",
                           color='Risk_Rating',
                           color_continuous_scale='Reds')

                fig.update_layout(yaxis={'categoryorder': 'total ascending'})

                return fig

        elif chart_type == 'status_tracking':
            # Risk status tracking
            if 'Status' in data.columns:
                status_counts = data['Status'].value_counts()

                fig = px.bar(x=status_counts.index, y=status_counts.values,
                           title="Risk Status Tracking",
                           labels={'x': 'Status', 'y': 'Number of Risks'},
                           color=status_counts.values,
                           color_continuous_scale='Blues')

                return fig

        else:
            # Default chart creation
            if chart_type == 'bar':
                fig = px.bar(data, x=x_axis, y=y_axis, color=color,
                           title=f"Risk Analysis: {x_axis} vs {y_axis}")
            elif chart_type == 'line':
                fig = px.line(data, x=x_axis, y=y_axis, color=color,
                            title=f"Risk Trend: {x_axis} vs {y_axis}")
            elif chart_type == 'scatter':
                fig = px.scatter(data, x=x_axis, y=y_axis, color=color,
                               title=f"Risk Correlation: {x_axis} vs {y_axis}")
            else:
                return None

        # Update layout untuk risk theme
        if 'fig' in locals():
            fig.update_layout(
                font=dict(size=12),
                title_font_size=16,
                template='plotly_white'
            )

            return fig

    except Exception as e:
        st.error(f"Error creating risk visualization: {e}")
        return None

# ============= MAIN APPLICATION =============
def main():
    # Header
    st.markdown('<h1 class="main-header">⚠️ Data Analysis for Risk Management</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered risk analysis and management dashboard based on ISO 31000 standards</p>', unsafe_allow_html=True)

    # ISO 31000 Info Box
    st.markdown("""
    <div class="iso-framework">
    📋 <strong>ISO 31000 Framework</strong><br>
    Aplikasi ini dirancang sesuai dengan standar internasional ISO 31000:2018 untuk manajemen risiko yang komprehensif, 
    mencakup identifikasi, analisis, evaluasi, dan treatment risiko dengan pendekatan sistematis.
    </div>
    """, unsafe_allow_html=True)

    # ============= SIDEBAR KONFIGURASI API =============
    st.sidebar.title("🤖 AI Risk Assistant Config")

    # API Provider Selection
    api_provider = st.sidebar.selectbox(
        "🎯 Choose AI Provider:",
        ["OpenAI", "Perplexity"]
    )

    # API Key Configuration
    with st.sidebar.expander("🔑 API Configuration", expanded=True):
        if api_provider == "OpenAI":
            api_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                placeholder="sk-...",
                help="Masukkan OpenAI API key Anda"
            )
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
            model = st.selectbox("Model:", model_options)

        else:  # Perplexity
            api_key = st.text_input(
                "Perplexity API Key:",
                type="password", 
                placeholder="pplx-...",
                help="Masukkan Perplexity API key Anda"
            )
            model_options = [
                "llama-3.1-sonar-small-128k-online",
                "llama-3.1-sonar-large-128k-online", 
                "llama-3.1-sonar-huge-128k-online"
            ]
            model = st.selectbox("Model:", model_options)

    # API Status Indicator
    if api_key:
        st.sidebar.success("✅ AI Risk Assistant Ready!")
    else:
        st.sidebar.warning("⚠️ Configure API key to use AI features")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Risk Management Focus")
    st.sidebar.markdown("""
    - **Impact Analysis** & Mitigation
    - **Root Cause** Investigation  
    - **Control Effectiveness** Assessment
    - **Priority** Risk Ranking
    - **Treatment Strategy** Planning
    """)
    
    # Creator information at bottom of sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="creator-info">
    <h4>👨‍💻 Creator Information</h4>
    <p><strong>Created by:</strong> Vito Devara</p>
    <p><strong>Phone:</strong> 081259795994</p>
    </div>
    # ============= FILE UPLOAD =============
    st.markdown("### 📤 Upload Risk Assessment Data")

    uploaded_file = st.file_uploader(
        "Upload your Risk Assessment CSV or Excel file",
        type=['csv', 'xlsx'],
        help="Expected columns: Risk_ID, Asset, Threat, Cause, Impact, Likelihood, Risk_Rating, Control, Risk_Owner, Risk_Treatment, Status, Comments"
    )

    # Manage risk chat history
    manage_risk_chat_history(uploaded_file)

    if uploaded_file is not None:
        # Load risk data
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            # Success message
            st.markdown('<div class="success-msg">✅ Risk assessment data uploaded successfully!</div>', unsafe_allow_html=True)

            # Analyze risk data
            risk_analysis = analyze_risk_data(data)

            # ============= RISK DASHBOARD =============
            st.markdown("### 🎯 Risk Management Dashboard")

            if not risk_analysis.get('error'):
                # Risk Metrics Cards
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    critical_count = risk_analysis.get('critical_risks', 0)
                    st.markdown(f'<div class="risk-card-critical"><h3>{critical_count}</h3><p>Critical Risks<br>(Rating ≥ 20)</p></div>', unsafe_allow_html=True)

                with col2:
                    high_count = risk_analysis.get('high_risks', 0)
                    st.markdown(f'<div class="risk-card-high"><h3>{high_count}</h3><p>High Risks<br>(Rating 15-19)</p></div>', unsafe_allow_html=True)

                with col3:
                    medium_count = risk_analysis.get('medium_risks', 0)
                    st.markdown(f'<div class="risk-card-medium"><h3>{medium_count}</h3><p>Medium Risks<br>(Rating 10-14)</p></div>', unsafe_allow_html=True)

                with col4:
                    low_count = risk_analysis.get('low_risks', 0)
                    st.markdown(f'<div class="risk-card-low"><h3>{low_count}</h3><p>Low Risks<br>(Rating < 10)</p></div>', unsafe_allow_html=True)

                # Additional metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    total_risks = risk_analysis.get('total_risks', 0)
                    st.metric("📊 Total Risks", total_risks)

                with col2:
                    open_risks = risk_analysis.get('open_risks', 0)
                    st.metric("🔓 Open Risks", open_risks)

                with col3:
                    avg_rating = risk_analysis.get('avg_risk_rating', 0)
                    st.metric("📈 Avg Risk Rating", f"{avg_rating:.1f}")

            # Display data preview
            st.markdown("### 📋 Risk Assessment Data")
            st.dataframe(data, use_container_width=True, height=300)

            # ============= RISK ANALYSIS TABS =============
            st.markdown("### 📊 Risk Analysis")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                '🎯 Risk Overview', '📈 Risk Matrix', 
                '🔍 Top Risks', '👥 Risk Owners', '📋 Controls'
            ])

            with tab1:
                col1, col2 = st.columns(2)

                with col1:
                    # Risk distribution
                    fig_dist = create_risk_visualization(data, 'risk_distribution', '', '')
                    if fig_dist:
                        st.plotly_chart(fig_dist, use_container_width=True)

                with col2:
                    # Status tracking
                    fig_status = create_risk_visualization(data, 'status_tracking', '', '')
                    if fig_status:
                        st.plotly_chart(fig_status, use_container_width=True)

            with tab2:
                # Risk matrix
                fig_matrix = create_risk_visualization(data, 'risk_matrix', '', '')
                if fig_matrix:
                    st.plotly_chart(fig_matrix, use_container_width=True)
                else:
                    st.info("Risk Matrix requires 'Impact' and 'Likelihood' columns in your data.")

            with tab3:
                # Top risks
                fig_top = create_risk_visualization(data, 'top_risks', '', '')
                if fig_top:
                    st.plotly_chart(fig_top, use_container_width=True)

                # Top risks table
                if 'Risk_Rating' in data.columns:
                    st.markdown("#### 📋 Top 10 Highest Risk Items")
                    top_risks_table = data.nlargest(10, 'Risk_Rating')[['Risk_ID', 'Asset', 'Threat', 'Risk_Rating', 'Status']]
                    st.dataframe(top_risks_table, use_container_width=True)

            with tab4:
                # Risk owners analysis
                if 'Risk_Owner' in data.columns:
                    st.markdown("#### 👥 Risk Distribution by Owner")
                    owner_counts = data['Risk_Owner'].value_counts()

                    fig_owners = px.bar(x=owner_counts.values, y=owner_counts.index,
                                      orientation='h',
                                      title="Risks Assigned by Owner",
                                      labels={'x': 'Number of Risks', 'y': 'Risk Owner'})

                    st.plotly_chart(fig_owners, use_container_width=True)

                    # Owner performance table
                    if 'Status' in data.columns:
                        owner_perf = data.groupby(['Risk_Owner', 'Status']).size().unstack(fill_value=0)
                        st.markdown("#### 📊 Risk Owner Performance")
                        st.dataframe(owner_perf, use_container_width=True)

            with tab5:
                # Controls analysis
                if 'Control' in data.columns:
                    st.markdown("#### 🛡️ Control Effectiveness Analysis")

                    # Most common controls
                    control_counts = data['Control'].value_counts().head(10)

                    fig_controls = px.bar(x=control_counts.values, y=control_counts.index,
                                        orientation='h',
                                        title="Most Common Risk Controls",
                                        labels={'x': 'Frequency', 'y': 'Control Type'})

                    st.plotly_chart(fig_controls, use_container_width=True)

                    # Control effectiveness by risk level
                    if 'Risk_Rating' in data.columns:
                        control_effectiveness = data.groupby('Control')['Risk_Rating'].agg(['mean', 'count']).round(2)
                        control_effectiveness.columns = ['Avg Risk Rating', 'Number of Risks']
                        control_effectiveness = control_effectiveness.sort_values('Avg Risk Rating', ascending=False)

                        st.markdown("#### 📈 Control Effectiveness (by Avg Risk Rating)")
                        st.dataframe(control_effectiveness, use_container_width=True)

            # ============= AI RISK ASSISTANT =============
            st.markdown("### 🤖 AI Risk Management Assistant")

            # Add clear chat button
            add_risk_clear_button()

            if not api_key:
                st.markdown('<div class="warning-msg">⚠️ Konfigurasi API key di sidebar untuk menggunakan AI Risk Assistant</div>', unsafe_allow_html=True)
            else:
                # Initialize risk chat history
                if "risk_messages" not in st.session_state:
                    st.session_state.risk_messages = []

                # Risk-specific quick prompts
                st.markdown("#### 💡 Risk Analysis Quick Prompts")
                col1, col2, col3, col4 = st.columns(4)

                risk_prompts = [
                    "Analisis risiko dengan rating tertinggi dan mitigasinya",
                    "Identifikasi penyebab utama risiko yang perlu prioritas",
                    "Evaluasi efektivitas kontrol yang sudah ada",
                    "Rekomendasi treatment untuk risiko critical"
                ]

                for i, (col, risk_prompt) in enumerate(zip([col1, col2, col3, col4], risk_prompts)):
                    with col:
                        if st.button(f"⚠️ {risk_prompt}", key=f"risk_quick_{i}", use_container_width=True):
                            st.session_state.risk_messages.append({"role": "user", "content": risk_prompt})
                            st.session_state.last_activity = time.time()
                            st.rerun()

                # Chat input
                if prompt := st.chat_input("Tanyakan tentang analisis risiko, mitigasi, atau strategi treatment..."):
                    # Update activity timestamp
                    st.session_state.last_activity = time.time()

                    # Add user message to chat history
                    st.session_state.risk_messages.append({"role": "user", "content": prompt})

                    # Generate AI response
                    with st.spinner(f"🤖 AI Risk Expert menganalisis..."):
                        # Prepare risk data context
                        risk_context = f"""
                        Risk Assessment Dataset: {uploaded_file.name}
                        Total Risks: {len(data)}

                        #≥
                        Risk Distribution:
                        - Critical (>=20): {risk_analysis.get('critical_risks', 0)}
                        - High (15-19): {risk_analysis.get('high_risks', 0)}
                        - Medium (10-14): {risk_analysis.get('medium_risks', 0)}
                        - Low (<10): {risk_analysis.get('low_risks', 0)}

                        Open Risks: {risk_analysis.get('open_risks', 0)}
                        Average Risk Rating: {risk_analysis.get('avg_risk_rating', 0):.1f}

                        Top Risk Areas: {list(risk_analysis.get('top_assets_at_risk', {}).keys())[:3] if risk_analysis.get('top_assets_at_risk') else 'N/A'}
                        Common Threats: {list(risk_analysis.get('top_threats', {}).keys())[:3] if risk_analysis.get('top_threats') else 'N/A'}

                        Sample High-Risk Items:
                        {data.nlargest(3, 'Risk_Rating')[['Risk_ID', 'Asset', 'Threat', 'Impact', 'Likelihood', 'Risk_Rating']].to_string() if 'Risk_Rating' in data.columns else 'No risk rating data'}
                        """

                        if api_provider == "OpenAI":
                            response = get_risk_openai_response(prompt, risk_context, api_key, model)
                        else:
                            response = get_risk_perplexity_response(prompt, risk_context, api_key, model)

                        # Add assistant response to chat history
                        st.session_state.risk_messages.append({"role": "assistant", "content": response})

                # Display risk chat history
                for message in st.session_state.risk_messages:
                    with st.chat_message(message["role"]):
                        if message["role"] == "assistant":
                            # Format risk assistant response in mitigation box
                            st.markdown(f'<div class="mitigation-box">{message["content"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(message["content"])

        except Exception as e:
            st.error(f"Error loading risk assessment data: {e}")
            st.info("Please ensure your file contains risk assessment data with appropriate columns.")

    else:
        # Sample data template
        st.markdown("### 📋 Expected Data Format")
        st.info("""
        Upload CSV/Excel file dengan kolom berikut untuk analisis risk management yang optimal:

        **Required columns:**
        - Risk_ID, Asset, Threat, Cause, Impact, Likelihood, Risk_Rating

        **Optional columns:**
        - Control, Risk_Owner, Risk_Treatment, Status, Comments, Target_Date
        """)

        # Show sample data format
        sample_data = pd.DataFrame({
            'Risk_ID': ['RISK0001', 'RISK0002', 'RISK0003'],
            'Asset': ['Database Server', 'Web Application', 'Email System'],
            'Threat': ['Data Breach', 'System Downtime', 'Unauthorized Access'],
            'Cause': ['Weak Passwords', 'Outdated Software', 'No Access Control'],
            'Impact': [5, 4, 3],
            'Likelihood': [4, 3, 2],
            'Risk_Rating': [20, 12, 6],
            'Control': ['MFA', 'Backup System', 'Access Matrix'],
            'Risk_Owner': ['IT Manager', 'CISO', 'Security Officer'],
            'Status': ['Open', 'In Progress', 'Closed']
        })

        st.markdown("#### 📊 Sample Risk Assessment Data Format")
        st.dataframe(sample_data, use_container_width=True)

    # ============= FOOTER =============
    st.markdown("---")

    # Footer information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Features")
        st.markdown("""
        - 📤 Risk Assessment Data Upload
        - 🎯 Real-time Risk Dashboard
        - 📊 Risk Matrix Visualization
        - 🤖 AI-powered Risk Analysis
        - 🛡️ Control Effectiveness Review
        """)

    with col2:
        st.markdown("### AI Risk Assistant")
        st.markdown("""
        - 💬 Impact & Mitigation Analysis
        - 🔍 Root Cause Investigation
        - 📈 Priority Risk Identification
        - 🛠️ Treatment Strategy Planning
        - 📋 ISO 31000 Compliance Check
        """)

    with col3:
        st.markdown("### Enterprise Ready")
        st.markdown("""
        - 📊 Executive Risk Reporting
        - 👥 Risk Owner Assignment
        - 📅 Treatment Timeline Tracking
        - 🔄 Continuous Monitoring
        - 📈 Risk Trend Analysis
        """)

    # Contact info
    st.markdown("---")
    st.markdown("**🛡️ Professional Risk Management with AI Intelligence**")

if __name__ == "__main__":
    main()
