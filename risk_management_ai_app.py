
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
import uuid
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
    .session-id {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 0.5rem;
        text-align: center;
        font-family: monospace;
        color: #495057;
        margin: 1rem 0;
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
    .creator-info {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============= SESSION MANAGEMENT =============
def initialize_session():
    """Initialize session ID untuk multi-user support"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if 'risk_data' not in st.session_state:
        st.session_state.risk_data = None

    if 'risk_messages' not in st.session_state:
        st.session_state.risk_messages = []

    if 'current_risk_dataset' not in st.session_state:
        st.session_state.current_risk_dataset = None

def manage_risk_session(uploaded_file):
    """Manage session per dataset untuk multi-user"""
    if uploaded_file is not None:
        dataset_key = f"{uploaded_file.name}_{uploaded_file.size}_{st.session_state.session_id}"
    else:
        dataset_key = f"no_dataset_{st.session_state.session_id}"

    if st.session_state.current_risk_dataset != dataset_key:
        st.session_state.current_risk_dataset = dataset_key
        st.session_state.risk_messages = []  # Auto clear chat for new dataset
        return True  # Dataset changed
    return False  # Same dataset

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

        full_prompt = f"""
        Risk Data Context: {data_context}
        User Query: {prompt}

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
        return f"❌ Error OpenAI: {str(e)}"

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

        system_prompt = """Anda adalah AI Risk Management Expert yang mengkhususkan diri dalam analisis dan manajemen risiko sesuai standar ISO 31000. Fokus pada analisis praktis, mitigasi konkret, dan rekomendasi actionable. Jawab dalam bahasa Indonesia dengan gaya profesional risk management."""

        full_prompt = f"""Risk Data Context: {data_context}\nUser Query: {prompt}\nFormat: Ringkasan singkat, Key Points, Mitigasi/Action Items."""

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

            # Categorize risks - FIXED UNICODE
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

def create_custom_visualization(data, chart_type, x_col, y_col=None, color_col=None, title="Custom Chart"):
    """Create custom visualizations berdasarkan user selection"""
    try:
        if chart_type == "Bar Chart":
            if y_col:
                fig = px.bar(data, x=x_col, y=y_col, color=color_col, title=title)
            else:
                # Count plot untuk categorical
                counts = data[x_col].value_counts()
                fig = px.bar(x=counts.index, y=counts.values, color=counts.values,
                           title=f"{title} - {x_col} Distribution",
                           labels={'x': x_col, 'y': 'Count'})

        elif chart_type == "Line Chart":
            if y_col:
                fig = px.line(data, x=x_col, y=y_col, color=color_col, title=title)
            else:
                # Line chart dengan index
                fig = px.line(data, x=data.index, y=x_col, title=f"{title} - {x_col} Trend")

        elif chart_type == "Scatter Plot":
            if y_col:
                fig = px.scatter(data, x=x_col, y=y_col, color=color_col, 
                               size='Risk_Rating' if 'Risk_Rating' in data.columns else None, title=title)
            else:
                st.warning("Scatter plot requires both X and Y columns.")
                return None

        elif chart_type == "Histogram":
            fig = px.histogram(data, x=x_col, color=color_col, title=title)

        elif chart_type == "Box Plot":
            fig = px.box(data, x=color_col, y=x_col, title=title)

        elif chart_type == "Pie Chart":
            if x_col in data.columns:
                counts = data[x_col].value_counts()
                fig = px.pie(values=counts.values, names=counts.index, title=title)
            else:
                st.warning("Pie chart requires a categorical column.")
                return None
        else:
            return None

        # Update layout untuk risk theme
        fig.update_layout(
            font=dict(size=12),
            title_font_size=16,
            template='plotly_white'
        )

        return fig

    except Exception as e:
        st.error(f"Error creating custom visualization: {e}")
        return None

# ============= MAIN APPLICATION =============
def main():
    # Initialize session
    initialize_session()

    # Header
    st.markdown('<h1 class="main-header">⚠️ Data Analysis for Risk Management</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered risk analysis and management dashboard based on ISO 31000 standards</p>', unsafe_allow_html=True)

    # Session ID Display
    st.markdown(f'<div class="session-id">🔒 Session ID: <strong>{st.session_state.session_id}</strong> | Multi-user Support Active</div>', unsafe_allow_html=True)

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
                help="Masukkan OpenAI API key Anda",
                key="openai_key_input"
            )
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
            model = st.selectbox("Model:", model_options, key="openai_model_select")

        else:  # Perplexity
            api_key = st.text_input(
                "Perplexity API Key:",
                type="password", 
                placeholder="pplx-...",
                help="Masukkan Perplexity API key Anda",
                key="perplexity_key_input"
            )
            model_options = [
                "llama-3.1-sonar-small-128k-online",
                "llama-3.1-sonar-large-128k-online", 
                "llama-3.1-sonar-huge-128k-online"
            ]
            model = st.selectbox("Model:", model_options, key="perplexity_model_select")

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
    """, unsafe_allow_html=True)

    # ============= FILE UPLOAD =============
    st.markdown("### 📤 Upload Risk Assessment Data")

    uploaded_file = st.file_uploader(
        "Upload your Risk Assessment CSV or Excel file",
        type=['csv', 'xlsx'],
        help="Expected columns: Risk_ID, Asset, Threat, Cause, Impact, Likelihood, Risk_Rating, Control, Risk_Owner, Risk_Treatment, Status, Comments",
        key="risk_file_uploader"
    )

    # Manage session per dataset
    dataset_changed = manage_risk_session(uploaded_file)

    if dataset_changed and uploaded_file:
        st.info("📊 New dataset loaded - chat history cleared for this session")

    if uploaded_file is not None:
        # Load risk data
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            st.session_state.risk_data = data

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
                    st.markdown(f'<div class="risk-card-critical"><h3>{critical_count}</h3><p>Critical Risks<br>(Rating >= 20)</p></div>', unsafe_allow_html=True)

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

            # ============= CUSTOM VISUALIZATION SECTION =============
            st.markdown("### 📊 Custom Risk Visualization")

            viz_col1, viz_col2 = st.columns([2, 1])

            with viz_col2:
                st.markdown("#### 🎨 Visualization Settings")

                # Chart type selection
                chart_type = st.selectbox(
                    "Chart Type:",
                    ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"],
                    key="custom_chart_type"
                )

                # Column selections
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
                all_cols = data.columns.tolist()

                x_column = st.selectbox("X-axis Column:", all_cols, key="x_axis_select")

                if chart_type in ["Scatter Plot", "Line Chart"]:
                    y_column = st.selectbox("Y-axis Column:", numeric_cols, key="y_axis_select")
                else:
                    y_column = st.selectbox("Y-axis Column (optional):", [None] + numeric_cols, key="y_axis_optional")

                color_column = st.selectbox("Color Column (optional):", [None] + categorical_cols, key="color_select")

                chart_title = st.text_input("Chart Title:", value=f"Risk Analysis - {chart_type}", key="chart_title_input")

                if st.button("🎨 Generate Custom Chart", use_container_width=True, key="generate_custom_chart"):
                    with st.spinner("Creating custom visualization..."):
                        custom_fig = create_custom_visualization(
                            data, chart_type, x_column, y_column, color_column, chart_title
                        )
                        if custom_fig:
                            st.session_state.custom_chart = custom_fig

            with viz_col1:
                if 'custom_chart' in st.session_state:
                    st.plotly_chart(st.session_state.custom_chart, use_container_width=True)
                else:
                    st.info("👈 Configure visualization settings and click 'Generate Custom Chart'")

            # ============= RISK ANALYSIS TABS =============
            st.markdown("### 📊 Risk Analysis Tabs")

            tab1, tab2, tab3, tab4 = st.tabs([
                '🎯 Risk Matrix', '📈 Risk Distribution', 
                '🔍 Top Risks', '👥 Risk Owners'
            ])

            with tab1:
                # Risk matrix
                if 'Impact' in data.columns and 'Likelihood' in data.columns:
                    fig_matrix = px.scatter(data, x='Likelihood', y='Impact', 
                                   color='Risk_Rating' if 'Risk_Rating' in data.columns else None,
                                   size='Risk_Rating' if 'Risk_Rating' in data.columns else None,
                                   hover_data=['Risk_ID', 'Asset', 'Threat'] if all(col in data.columns for col in ['Risk_ID', 'Asset', 'Threat']) else None,
                                   title="Risk Matrix - Impact vs Likelihood",
                                   color_continuous_scale='Reds')
                    st.plotly_chart(fig_matrix, use_container_width=True)
                else:
                    st.info("Risk Matrix requires 'Impact' and 'Likelihood' columns in your data.")

            with tab2:
                # Risk distribution
                if 'Risk_Rating' in data.columns:
                    def categorize_risk(rating):
                        if rating >= 20:
                            return 'Critical'
                        elif rating >= 15:
                            return 'High'
                        elif rating >= 10:
                            return 'Medium'
                        else:
                            return 'Low'

                    data['Risk_Category'] = data['Risk_Rating'].apply(categorize_risk)
                    risk_colors = {'Critical': '#dc3545', 'High': '#fd7e14', 'Medium': '#ffc107', 'Low': '#28a745'}

                    fig_pie = px.pie(data, names='Risk_Category', 
                                   title="Risk Distribution by Severity",
                                   color='Risk_Category',
                                   color_discrete_map=risk_colors)
                    st.plotly_chart(fig_pie, use_container_width=True)

            with tab3:
                # Top risks
                if 'Risk_Rating' in data.columns:
                    top_risks = data.nlargest(10, 'Risk_Rating')
                    fig_top = px.bar(top_risks, x='Risk_Rating', y='Risk_ID',
                                   orientation='h',
                                   title="Top 10 Highest Rated Risks",
                                   color='Risk_Rating',
                                   color_continuous_scale='Reds')
                    st.plotly_chart(fig_top, use_container_width=True)

            with tab4:
                # Risk owners analysis
                if 'Risk_Owner' in data.columns:
                    owner_counts = data['Risk_Owner'].value_counts()
                    fig_owners = px.bar(x=owner_counts.values, y=owner_counts.index,
                                      orientation='h',
                                      title="Risks Assigned by Owner")
                    st.plotly_chart(fig_owners, use_container_width=True)

            # ============= AI RISK ASSISTANT =============
            if api_key:
                st.markdown("### 🤖 AI Risk Management Assistant")

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
                        if st.button(f"⚠️ {risk_prompt}", key=f"risk_quick_{i}_{st.session_state.session_id}", use_container_width=True):
                            st.session_state.risk_messages.append({"role": "user", "content": risk_prompt})
                            st.rerun()

                # Chat input
                if prompt := st.chat_input("Tanyakan tentang analisis risiko, mitigasi, atau strategi treatment...", key=f"risk_chat_input_{st.session_state.session_id}"):
                    # Add user message to chat history
                    st.session_state.risk_messages.append({"role": "user", "content": prompt})

                    # Generate AI response
                    with st.spinner(f"🤖 AI Risk Expert menganalisis..."):
                        # Prepare risk data context - FIXED UNICODE ISSUE
                        risk_context = f"""
                        Risk Assessment Dataset: {uploaded_file.name}
                        Session: {st.session_state.session_id}
                        Total Risks: {len(data)}

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
            else:
                st.markdown("### 🤖 AI Risk Management Assistant")
                st.markdown('<div class="warning-msg">⚠️ Konfigurasi API key di sidebar untuk menggunakan AI Risk Assistant</div>', unsafe_allow_html=True)

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

if __name__ == "__main__":
    main()
