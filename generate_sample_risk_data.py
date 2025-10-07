
# Sample Risk Assessment Data Generator untuk testing
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Generate sample risk assessment data sesuai dengan aplikasi
np.random.seed(42)
n_risks = 50

# Risk data components
assets = [
    'Database Server', 'Web Application', 'Email System', 'HR Records', 'Financial System', 
    'Customer Data', 'Backup System', 'Network Infrastructure', 'Mobile App', 'Cloud Storage',
    'Payment Gateway', 'API Gateway', 'User Authentication', 'Data Warehouse', 'File Server',
    'ERP System', 'CRM Platform', 'Document Management', 'Reporting System', 'Analytics Platform'
]

threats = [
    'Data Breach', 'System Downtime', 'Unauthorized Access', 'Malware Attack', 'DDoS Attack',
    'Insider Threat', 'Social Engineering', 'Physical Theft', 'Natural Disaster', 'Power Outage',
    'Hardware Failure', 'Software Bug', 'Human Error', 'Third-party Breach', 'Regulatory Change',
    'Ransomware', 'SQL Injection', 'Phishing Attack', 'System Misconfiguration', 'Data Loss'
]

causes = [
    'Outdated Software', 'Weak Passwords', 'No Backup', 'Insufficient Training', 'Legacy System',
    'Poor Access Control', 'Missing Updates', 'Inadequate Monitoring', 'Network Vulnerability',
    'Physical Security Gap', 'Process Weakness', 'Resource Constraints', 'Vendor Dependency',
    'Compliance Gap', 'Technology Debt', 'Configuration Error', 'Missing Patches', 'No Encryption',
    'Inadequate Documentation', 'Lack of Incident Response'
]

controls = [
    'Multi-Factor Authentication', 'Firewall Update', 'Backup System', 'Staff Training',
    'Access Control Matrix', 'Security Monitoring', 'Patch Management', 'Encryption',
    'Incident Response Plan', 'Regular Audits', 'Vendor Assessment', 'Data Classification',
    'Network Segmentation', 'Disaster Recovery', 'Security Awareness', 'Antivirus Software',
    'Intrusion Detection', 'Data Loss Prevention', 'Vulnerability Scanning', 'Risk Assessment'
]

risk_owners = [
    'IT Manager', 'CISO', 'CFO', 'HR Director', 'Operations Manager', 'CEO', 
    'Data Protection Officer', 'Compliance Officer', 'Network Administrator', 'Security Analyst',
    'Risk Manager', 'Audit Manager', 'Business Unit Manager', 'Technical Lead'
]

treatments = ['Mitigate', 'Transfer', 'Accept', 'Avoid']
statuses = ['Open', 'In Progress', 'Closed', 'On Hold']

# Generate comprehensive risk data
risk_data = []
for i in range(1, n_risks + 1):
    impact = random.randint(1, 5)
    likelihood = random.randint(1, 5)
    risk_rating = impact * likelihood

    # Create risk severity categories
    if risk_rating >= 20:
        severity = 'Critical'
        priority = 'High'
    elif risk_rating >= 15:
        severity = 'High'
        priority = 'High'
    elif risk_rating >= 10:
        severity = 'Medium'
        priority = 'Medium'
    elif risk_rating >= 5:
        severity = 'Low'
        priority = 'Low'
    else:
        severity = 'Very Low'
        priority = 'Low'

    # Generate target dates
    target_date = datetime.now() + timedelta(days=random.randint(30, 365))

    risk_data.append({
        'Risk_ID': f'RISK{i:04d}',
        'Asset': random.choice(assets),
        'Threat': random.choice(threats),
        'Cause': random.choice(causes),
        'Impact': impact,
        'Likelihood': likelihood,
        'Risk_Rating': risk_rating,
        'Risk_Severity': severity,
        'Control': random.choice(controls),
        'Risk_Owner': random.choice(risk_owners),
        'Risk_Treatment': random.choice(treatments),
        'Status': random.choice(statuses),
        'Priority': priority,
        'Target_Date': target_date.strftime('%Y-%m-%d'),
        'Comments': f'Risk assessment for {random.choice(assets).lower()} - requires {random.choice(["immediate", "urgent", "routine", "planned"])} attention. {random.choice(["High business impact potential.", "Regulatory compliance concern.", "Operational continuity risk.", "Financial impact expected."])}'
    })

# Create DataFrame and save
df_risk_complete = pd.DataFrame(risk_data)
df_risk_complete.to_csv('sample_risk_assessment_data.csv', index=False)

print("✅ Sample Risk Assessment Data Generated")
print(f"📊 Total Risks: {len(df_risk_complete)}")
print(f"🎯 Risk Distribution:")
print(df_risk_complete['Risk_Severity'].value_counts())
print(f"📋 Status Distribution:")
print(df_risk_complete['Status'].value_counts())
print(f"👥 Risk Owners: {df_risk_complete['Risk_Owner'].nunique()}")
