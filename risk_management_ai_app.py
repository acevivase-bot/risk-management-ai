
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import openai
import requests
import json
import hashlib
import time
from datetime import datetime
from io import StringIO, BytesIO
import base64

# ============= KONFIGURASI HALAMAN =============
st.set_page_config(
    page_title="Data Analysis for Risk Management",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CSS STYLING =============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #e74c3c, #c0392b);
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
    .risk-card {
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .critical-risk {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    }
    .high-risk {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
    }
    .medium-risk {
        background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
        color: #2c3e50;
    }
    .low-risk {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
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
    .info-msg {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 0.75rem 1.25rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .step-box {
        background: #f8f9fa;
        border-left: 4px solid #e74c3c;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
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

# ============= RISK ANALYSIS FUNCTIONS =============

def analyze_risk_data(df):
    """Analyze risk data and provide insights"""
    analysis = {}

    # Basic statistics
    analysis['total_risks'] = len(df)
    analysis['open_risks'] = len(df[df['Status'] == 'Open']) if 'Status' in df.columns else 0
    analysis['closed_risks'] = len(df[df['Status'] == 'Closed']) if 'Status' in df.columns else 0

    # Risk severity analysis
    if 'Risk_Rating' in df.columns:
        analysis['avg_risk_rating'] = df['Risk_Rating'].mean()
        analysis['max_risk_rating'] = df['Risk_Rating'].max()
        analysis['min_risk_rating'] = df['Risk_Rating'].min()

        # Risk categorization (fixed without unicode)
        analysis['critical_risks'] = len(df[df['Risk_Rating'] >= 20])
        analysis['high_risks'] = len(df[(df['Risk_Rating'] >= 15) & (df['Risk_Rating'] < 20)])
        analysis['medium_risks'] = len(df[(df['Risk_Rating'] >= 10) & (df['Risk_Rating'] < 15)])
        analysis['low_risks'] = len(df[df['Risk_Rating'] < 10])

    # Top risk areas
    if 'Asset' in df.columns:
        analysis['top_assets'] = df['Asset'].value_counts().head(5).to_dict()

    if 'Threat' in df.columns:
        analysis['top_threats'] = df['Threat'].value_counts().head(5).to_dict()

    if 'Risk_Owner' in df.columns:
        analysis['risk_by_owner'] = df['Risk_Owner'].value_counts().head(5).to_dict()

    return analysis

def get_risk_mitigation_suggestions(risk_data, risk_context, api_key="", model="gpt-3.5-turbo"):
    """Get AI suggestions for risk mitigation"""
    try:
        if not api_key:
            return "Silakan konfigurasi API key untuk mendapatkan saran AI."

        system_prompt = """Anda adalah AI Risk Management Expert yang mengkhususkan diri dalam ISO 31000 dan manajemen risiko enterprise.

        Berikan analisis dan rekomendasi untuk:
        - Risk assessment dan prioritization
        - Mitigation strategies yang efektif
        - Control effectiveness evaluation
        - Risk monitoring dan reporting
        - Compliance dengan framework ISO 31000

        Fokus pada actionable recommendations yang dapat diimplementasikan langsung oleh risk managers.
        Jawab dalam bahasa Indonesia dengan pendekatan praktis dan professional."""

        user_prompt = f"""
        Konteks Risk Data: {risk_context}

        Pertanyaan: {risk_data}

        Berikan analisis mendalam dan rekomendasi strategis untuk manajemen risiko ini.
        """

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            return response.choices[0].message.content

        except ImportError:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

# ============= MAIN APPLICATION =============

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Data Analysis for Risk Management</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered risk assessment and mitigation planning with ISO 31000 compliance</p>', unsafe_allow_html=True)

    # ============= SIDEBAR =============
    st.sidebar.title("🛠️ Risk Management Tools")

    # Risk Assessment Framework
    with st.sidebar.expander("📋 ISO 31000 Framework", expanded=False):
        st.markdown("""
        **ISO 31000 Risk Management Process:**

        1. **Context Establishment**
        2. **Risk Identification** 
        3. **Risk Analysis**
        4. **Risk Evaluation**
        5. **Risk Treatment**
        6. **Monitoring & Review**
        7. **Communication & Consultation**
        """)

    # AI Assistant Config
    with st.sidebar.expander("🤖 AI Risk Assistant", expanded=False):
        api_provider = st.selectbox("AI Provider:", ["OpenAI", "Perplexity"])

        if api_provider == "OpenAI":
            api_key = st.text_input("OpenAI API Key:", type="password", placeholder="sk-...")
            model = st.selectbox("Model:", ["gpt-3.5-turbo", "gpt-4"])
        else:
            api_key = st.text_input("Perplexity API Key:", type="password", placeholder="pplx-...")
            model = st.selectbox("Model:", ["llama-3.1-sonar-small-128k-online"])

    # Creator information
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="creator-info">
    <h4>👨‍💻 Creator Information</h4>
    <p><strong>Created by:</strong> Vito Devara</p>
    <p><strong>Phone:</strong> 081259795994</p>
    </div>
    """, unsafe_allow_html=True)

    # ============= DATA UPLOAD SECTION =============
    st.markdown("### 📤 Upload Risk Assessment Data")

    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file with risk data:",
        type=['csv', 'xlsx'],
        help="File should contain columns: Risk_ID, Asset, Threat, Impact, Likelihood, Risk_Rating"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            st.session_state.risk_data = data.copy()
            st.success("✅ Risk data loaded successfully!")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    # ============= MAIN RISK ANALYSIS SECTION =============
    if 'risk_data' in st.session_state:
        data = st.session_state.risk_data

        # Risk Analysis
        risk_analysis = analyze_risk_data(data)

        # Risk Dashboard
        st.markdown("### 📊 Risk Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f'<div class="risk-card critical-risk"><h3>{risk_analysis["total_risks"]}</h3><p>Total Risks</p></div>', unsafe_allow_html=True)

        with col2:
            open_risks = risk_analysis.get("open_risks", 0)
            st.markdown(f'<div class="risk-card high-risk"><h3>{open_risks}</h3><p>Open Risks</p></div>', unsafe_allow_html=True)

        with col3:
            if 'avg_risk_rating' in risk_analysis:
                avg_rating = risk_analysis["avg_risk_rating"]
                st.markdown(f'<div class="risk-card medium-risk"><h3>{avg_rating:.1f}</h3><p>Avg Risk Rating</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-card medium-risk"><h3>N/A</h3><p>Avg Risk Rating</p></div>', unsafe_allow_html=True)

        with col4:
            closed_risks = risk_analysis.get("closed_risks", 0)
            st.markdown(f'<div class="risk-card low-risk"><h3>{closed_risks}</h3><p>Closed Risks</p></div>', unsafe_allow_html=True)

        # Risk Severity Breakdown (FIXED)
        if 'critical_risks' in risk_analysis:
            st.markdown("#### 🎯 Risk Severity Distribution")
            st.markdown(f"""
            <div class="info-msg">
            <strong>Risk Severity Breakdown:</strong><br>
            - Critical (>=20): {risk_analysis.get('critical_risks', 0)}<br>
            - High (15-19): {risk_analysis.get('high_risks', 0)}<br>
            - Medium (10-14): {risk_analysis.get('medium_risks', 0)}<br>
            - Low (<10): {risk_analysis.get('low_risks', 0)}
            </div>
            """, unsafe_allow_html=True)

        # Display risk data
        st.markdown("### 📋 Risk Assessment Data")
        st.dataframe(data, use_container_width=True, height=400)

        # ============= RISK VISUALIZATIONS =============
        st.markdown("### 📈 Risk Analysis Visualizations")

        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(['🎯 Risk Matrix', '📊 Risk Distribution', '🏆 Top Risks', '👥 Risk Owners'])

        with viz_tab1:
            # Risk Matrix (Impact vs Likelihood)
            if 'Impact' in data.columns and 'Likelihood' in data.columns:
                st.markdown("#### 🎯 Risk Matrix - Impact vs Likelihood")

                fig_matrix = px.scatter(
                    data, 
                    x='Likelihood', 
                    y='Impact',
                    color='Risk_Rating' if 'Risk_Rating' in data.columns else None,
                    size='Risk_Rating' if 'Risk_Rating' in data.columns else None,
                    hover_data=['Asset', 'Threat'] if all(col in data.columns for col in ['Asset', 'Threat']) else None,
                    title="Risk Matrix: Impact vs Likelihood"
                )
                fig_matrix.update_layout(
                    xaxis_title="Likelihood",
                    yaxis_title="Impact", 
                    width=800,
                    height=600
                )
                st.plotly_chart(fig_matrix, use_container_width=True)
            else:
                st.info("Risk Matrix requires 'Impact' and 'Likelihood' columns.")

        with viz_tab2:
            # Risk Distribution by Severity
            if 'Risk_Rating' in data.columns:
                st.markdown("#### 📊 Risk Distribution by Severity")

                # Create severity categories
                def categorize_risk(rating):
                    if rating >= 20:
                        return 'Critical'
                    elif rating >= 15:
                        return 'High'
                    elif rating >= 10:
                        return 'Medium'
                    else:
                        return 'Low'

                data['Risk_Severity'] = data['Risk_Rating'].apply(categorize_risk)
                severity_counts = data['Risk_Severity'].value_counts()

                fig_pie = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    title="Risk Distribution by Severity",
                    color_discrete_map={
                        'Critical': '#e74c3c',
                        'High': '#f39c12', 
                        'Medium': '#f1c40f',
                        'Low': '#27ae60'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with viz_tab3:
            # Top Risks
            if 'Risk_Rating' in data.columns:
                st.markdown("#### 🏆 Top Risks by Rating")

                top_risks = data.nlargest(10, 'Risk_Rating')

                fig_bar = px.bar(
                    top_risks,
                    x='Risk_Rating',
                    y='Asset' if 'Asset' in data.columns else top_risks.index,
                    orientation='h',
                    title="Top 10 Risks by Rating",
                    color='Risk_Rating',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Top risks table
                st.markdown("##### 📋 Top Risks Details")
                display_columns = ['Asset', 'Threat', 'Risk_Rating', 'Status']
                available_columns = [col for col in display_columns if col in top_risks.columns]
                if available_columns:
                    st.dataframe(top_risks[available_columns], use_container_width=True)

        with viz_tab4:
            # Risk by Owner
            if 'Risk_Owner' in data.columns:
                st.markdown("#### 👥 Risk Distribution by Owner")

                owner_counts = data['Risk_Owner'].value_counts()

                fig_owner = px.bar(
                    x=owner_counts.values,
                    y=owner_counts.index,
                    orientation='h',
                    title="Risk Distribution by Owner"
                )
                st.plotly_chart(fig_owner, use_container_width=True)

        # ============= AI RISK ASSISTANT =============
        if api_key:
            st.markdown("### 🤖 AI Risk Management Assistant")

            # Initialize chat history
            if "risk_messages" not in st.session_state:
                st.session_state.risk_messages = []

            # Quick risk prompts (FIXED)
            st.markdown("#### 💡 Quick Risk Analysis Prompts")
            col1, col2, col3, col4 = st.columns(4)

            risk_prompts = [
                "Analisis risiko dengan rating tertinggi dan rekomendasikan mitigasi",
                "Identifikasi pola risiko berdasarkan asset dan threat yang dominan", 
                "Evaluasi efektivitas kontrol yang sudah ada dan gap analysis",
                "Berikan prioritas treatment untuk risiko critical dan high"
            ]

            for i, (col, prompt) in enumerate(zip([col1, col2, col3, col4], risk_prompts)):
                with col:
                    if st.button(f"💡 {prompt[:30]}...", key=f"risk_quick_{i}", use_container_width=True):
                        # Add to chat and trigger AI response
                        st.session_state.risk_messages.append({"role": "user", "content": prompt})

                        # Generate AI response immediately
                        risk_context = f"""
                        Risk Assessment Summary:
                        - Total Risks: {risk_analysis['total_risks']}
                        - Open Risks: {risk_analysis.get('open_risks', 0)}
                        - Average Risk Rating: {risk_analysis.get('avg_risk_rating', 'N/A')}
                        - Risk Distribution: Critical: {risk_analysis.get('critical_risks', 0)}, High: {risk_analysis.get('high_risks', 0)}, Medium: {risk_analysis.get('medium_risks', 0)}, Low: {risk_analysis.get('low_risks', 0)}
                        - Top Assets: {risk_analysis.get('top_assets', {})}
                        - Top Threats: {risk_analysis.get('top_threats', {})}
                        - Sample Data: {data.head(3).to_string()}
                        """

                        with st.spinner("🤖 AI sedang menganalisis risiko..."):
                            response = get_risk_mitigation_suggestions(prompt, risk_context, api_key, model)
                            st.session_state.risk_messages.append({"role": "assistant", "content": response})

                        st.rerun()

            # Chat input
            if prompt := st.chat_input("Tanya tentang risk management dan mitigation strategies..."):
                # Add user message
                st.session_state.risk_messages.append({"role": "user", "content": prompt})

                # Prepare risk context
                risk_context = f"""
                Risk Assessment Summary:
                - Total Risks: {risk_analysis['total_risks']}
                - Open Risks: {risk_analysis.get('open_risks', 0)}
                - Average Risk Rating: {risk_analysis.get('avg_risk_rating', 'N/A')}
                - Risk Distribution: Critical: {risk_analysis.get('critical_risks', 0)}, High: {risk_analysis.get('high_risks', 0)}, Medium: {risk_analysis.get('medium_risks', 0)}, Low: {risk_analysis.get('low_risks', 0)}
                - Top Assets: {risk_analysis.get('top_assets', {})}
                - Top Threats: {risk_analysis.get('top_threats', {})}
                - Sample Data: {data.head(3).to_string()}
                """

                # Generate AI response
                with st.spinner("🤖 AI sedang menganalisis risiko..."):
                    response = get_risk_mitigation_suggestions(prompt, risk_context, api_key, model)
                    st.session_state.risk_messages.append({"role": "assistant", "content": response})

            # Display chat history
            for message in st.session_state.risk_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # ============= DOWNLOAD SECTION =============
        st.markdown("### 📥 Download Risk Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📄 Download Risk Data CSV", use_container_width=True):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV File",
                    data=csv,
                    file_name=f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            if st.button("📊 Download Excel Report", use_container_width=True):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    data.to_excel(writer, sheet_name='Risk_Data', index=False)

                    # Create summary sheet
                    summary_data = {
                        'Metric': ['Total Risks', 'Open Risks', 'Closed Risks', 'Critical Risks', 'High Risks', 'Medium Risks', 'Low Risks'],
                        'Value': [
                            risk_analysis['total_risks'],
                            risk_analysis.get('open_risks', 0),
                            risk_analysis.get('closed_risks', 0),
                            risk_analysis.get('critical_risks', 0),
                            risk_analysis.get('high_risks', 0),
                            risk_analysis.get('medium_risks', 0),
                            risk_analysis.get('low_risks', 0)
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Risk_Summary', index=False)

                output.seek(0)
                st.download_button(
                    label="📊 Download Excel File",
                    data=output,
                    file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        with col3:
            if st.button("🔄 Clear Data", use_container_width=True):
                if 'risk_data' in st.session_state:
                    del st.session_state.risk_data
                if 'risk_messages' in st.session_state:
                    del st.session_state.risk_messages
                st.success("Data cleared!")
                st.rerun()

    else:
        # Instructions when no data loaded
        st.markdown("### 🎯 Get Started with Risk Management")

        st.markdown("""
        <div class="step-box">
        <h4>📊 Step 1: Upload Risk Data</h4>
        <ul>
        <li>Prepare your CSV or Excel file with risk assessment data</li>
        <li>Required columns: Risk_ID, Asset, Threat, Impact, Likelihood, Risk_Rating</li>
        <li>Optional columns: Control, Risk_Owner, Risk_Treatment, Status, Comments</li>
        </ul>
        </div>

        <div class="step-box">
        <h4>🔍 Step 2: Analyze Risk Profile</h4>
        <ul>
        <li>Review the risk dashboard with key metrics</li>
        <li>Examine risk severity distribution</li>
        <li>Identify top risks and risk patterns</li>
        </ul>
        </div>

        <div class="step-box">
        <h4>📈 Step 3: Visualize Risk Data</h4>
        <ul>
        <li>Explore risk matrix (Impact vs Likelihood)</li>
        <li>Analyze risk distribution by severity</li>
        <li>Review top risks and risk ownership</li>
        </ul>
        </div>

        <div class="step-box">
        <h4>🤖 Step 4: Get AI Recommendations</h4>
        <ul>
        <li>Configure your AI assistant API key</li>
        <li>Use quick prompts for immediate insights</li>
        <li>Ask specific questions about risk mitigation</li>
        <li>Download analysis reports and recommendations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Sample data format
        st.markdown("#### 📋 Expected Data Format")

        sample_data = {
            'Risk_ID': ['RISK001', 'RISK002', 'RISK003'],
            'Asset': ['Database Server', 'Web Application', 'Email System'],
            'Threat': ['Data Breach', 'DDoS Attack', 'Phishing'],
            'Cause': ['Weak Authentication', 'No DDoS Protection', 'User Training Gap'],
            'Impact': [5, 4, 3],
            'Likelihood': [4, 3, 5],
            'Risk_Rating': [20, 12, 15],
            'Control': ['MFA Implementation', 'Firewall Upgrade', 'Security Training'],
            'Risk_Owner': ['IT Manager', 'Network Admin', 'HR Manager'],
            'Status': ['Open', 'In Progress', 'Open']
        }

        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()
