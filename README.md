# ⚠️ Data Analysis for Risk Management

AI-powered risk management application built with Streamlit for comprehensive risk analysis, mitigation planning, and ISO 31000 compliance.

## 🚀 Live Demo

Access the application: [Risk Management AI Demo](https://your-app-name.streamlit.app)

## ✨ Key Features

### 🤖 **Smart AI Risk Assistant**
- **Context-aware responses** based on query type (Impact, Cause, Control, Priority)
- **Mitigation-focused recommendations** without unnecessary visualization suggestions
- **Token-optimized** for cost-effective API usage
- **ISO 31000 compliance** framework integration

### 📊 **Professional Risk Dashboard**
- **Real-time risk metrics** with severity-based color coding
- **Risk distribution analysis** (Critical, High, Medium, Low)
- **Risk owner performance** tracking
- **Control effectiveness** evaluation

### 📈 **Advanced Risk Visualizations**
- **Risk Matrix**: Impact vs Likelihood analysis
- **Risk Distribution**: Severity-based pie charts
- **Top Risks**: Highest-rated risks identification
- **Status Tracking**: Risk treatment progress
- **Owner Analysis**: Risk distribution by responsible parties

### 🛡️ **Enterprise-Grade Features**
- **Multi-user support** with session isolation
- **Automatic chat clearing** when datasets change
- **Professional reporting** ready outputs
- **Data privacy** protection

## 🛠️ Quick Start

### Option 1: Access Live App
1. Visit the live application URL
2. Configure your OpenAI or Perplexity API key in sidebar
3. Upload your risk assessment CSV/Excel file
4. Start analyzing risks with AI assistance

### Option 2: Local Deployment
```bash
git clone https://github.com/yourusername/risk-management-ai
cd risk-management-ai
pip install -r requirements.txt
streamlit run risk_management_ai_app.py
```

## 📊 Data Format

### Required Columns:
- `Risk_ID`: Unique risk identifier
- `Asset`: Asset or system at risk
- `Threat`: Potential threat description
- `Cause`: Root cause or vulnerability
- `Impact`: Impact rating (1-5 scale)
- `Likelihood`: Probability rating (1-5 scale)
- `Risk_Rating`: Calculated risk score (Impact × Likelihood)

### Optional Columns:
- `Control`: Existing risk controls
- `Risk_Owner`: Responsible party
- `Risk_Treatment`: Treatment strategy
- `Status`: Current status
- `Comments`: Additional notes

## 🎯 AI Capabilities

### **Impact Analysis**
Ask: *"Analisis dampak untuk risiko rating tinggi"*
Get: Consequence analysis, severity categorization, specific mitigations

### **Root Cause Investigation**
Ask: *"Identifikasi penyebab utama risiko sistem database"*
Get: Root cause analysis, preventive controls, upstream mitigation

### **Control Effectiveness**
Ask: *"Evaluasi efektivitas kontrol yang sudah ada"*
Get: Control assessment, gap analysis, improvement recommendations

### **Priority Planning**
Ask: *"Prioritas treatment untuk risiko critical"*
Get: Risk ranking, resource allocation, timeline planning

## 🏢 Business Applications

- **Enterprise Risk Management**: Complete risk assessment workflows
- **Compliance Management**: ISO 31000 framework alignment
- **Executive Reporting**: Dashboard-ready risk insights
- **Training Programs**: Interactive risk management education
- **Audit Support**: Control effectiveness validation

## 🔐 Security & Privacy

- **API keys** stored securely in session state only
- **No permanent data storage** - data exists only during session
- **Session isolation** - each user's data remains private
- **Automatic cleanup** - chat history cleared on dataset changes

## 📈 Sample Data

Generate sample risk assessment data:
```bash
python generate_sample_risk_data.py
```

This creates `sample_risk_assessment_data.csv` with 50 realistic risk records for testing.

## 🚀 Deployment Options

### Streamlit Cloud (Recommended)
- Free hosting with GitHub integration
- Automatic updates from repository
- Professional URL sharing
- Built-in authentication

### Local Development
- Full control over environment
- Custom configurations
- Enterprise integration capabilities
- Offline usage support

## 🤝 Contributing

Contributions welcome! Please feel free to submit pull requests or open issues for:
- New risk analysis features
- Additional visualization types
- Integration improvements
- Documentation updates

## 📝 License

This project is open source and available under the MIT License.

---

**🛡️ Empowering risk management professionals with AI-driven insights and analysis.**

Built with ❤️ for the risk management community.
