📞 Sales Call Intelligence & Coaching Agent

An Agentic AI system that analyzes sales call transcripts, extracts important sales moments, detects deal risks, generates actionable coaching, and drafts follow-up emails for sales representatives.

🚨 Problem Statement

Sales representatives conduct a large number of customer calls, but valuable information is often buried inside lengthy conversations.

Important signals such as:

Customer objections
Pricing discussions
Competitor mentions
Customer needs
Budget constraints
Decision-making authority
Purchase timeline
Next-step commitments

can be difficult and time-consuming to identify manually.

Manually reviewing every sales call also makes it difficult for sales managers to consistently identify deal risks and provide personalized coaching.

Our Goal

Build an AI-powered system that can automatically understand a sales conversation and transform it into:

Raw Sales Call → Structured Insights → Risk Analysis → Coaching → Follow-up Action

💡 Our Solution

We built Sales Call Intelligence & Coaching Agent, an Agentic AI system that takes a timestamped sales call transcript and converts it into structured, actionable insights.

The complete pipeline is:

Sales Call Transcript
        ↓
Transcript Parsing
        ↓
Conversation Metrics
        ↓
Agent 1: Sales Moment Extraction
        ↓
Extraction Validation
        ↓
Agent 2: BANT & Deal Risk Analysis
        ↓
Agent 3: Personalized Coaching
        ↓
Coaching Validation
        ↓
Follow-up Email Draft
        ↓
Interactive Dashboard
🤖 Agent 1 — Sales Moment Extraction

The extraction agent identifies important moments from the sales conversation.

It extracts categories including:

Objections
Pricing
Competitors
Buying Signals
Next Steps
Budget
Authority
Timeline
Customer Needs
Pain Points
Hesitation
Unanswered Questions

Each extracted moment contains:

Timestamp
Speaker
Category
Evidence
Status

Example:

Timestamp: 00:33
Speaker: CUSTOMER
Category: objection, need, timeline
Evidence: "My concern is really the onboarding time — we need this live in 3 weeks."
Status: unresolved
🛡️ Extraction Validation

The extracted moments are passed through a deterministic validation layer before being sent to the next agent.

The validator checks:

Valid timestamps
Valid speakers
Allowed categories
Evidence overlap with the transcript
Category-specific rules
Next-step context
Unsupported or invalid extractions

This helps reduce unsupported information and improves the reliability of the agent pipeline.

⚠️ Agent 2 — BANT & Deal Risk Analysis

The second agent analyzes the validated extraction results and conversation metrics.

It evaluates the four BANT dimensions:

Budget

Determines whether the customer's budget situation is clear.

Authority

Determines whether the person involved has decision-making authority or whether another decision-maker is required.

Need

Identifies whether the customer's requirements and problems are clearly established.

Timeline

Determines whether the customer has a defined implementation or purchase timeline.

Each BANT factor is classified as:

Strong
Partial
Unknown

The agent also identifies potential deal-risk factors and provides evidence-based reasons using timestamps from the conversation.

🎯 Agent 3 — Personalized Coaching

The coaching agent converts the extracted insights and risk analysis into actionable sales coaching.

Each coaching point contains:

What happened
Why it matters
Coaching recommendation
Suggested phrase

Example:

What happened:
The customer expressed a concern about onboarding time.

Why it matters:
The customer needs the solution live within a specific timeframe.

Coaching:
Confirm whether the proposed implementation timeline provides
enough buffer for the customer's requirement.

Suggested phrase:
"I understand you need to be live in three weeks. Our team can
have you up and running in two weeks — does that give you enough
buffer?"

The coaching agent is designed to remain grounded in the information present in the sales conversation.

📧 Follow-Up Email Draft

As a compulsory add-on, the system generates a follow-up email using the extracted next-step commitments.

The email can be:

Copied by the sales representative
Edited before sending
Used as a recap of the conversation

Example:

Subject: Follow-up on our conversation

Hi,

Thank you for taking the time to speak with me today. It was great
discussing your requirements and next steps.

As discussed, here are the next steps:

- Thursday at 2pm follow-up
- Calendar invite and pricing sheet to be shared

Looking forward to speaking with you.

Best regards,
Sales Representative

The feature reuses information already extracted from the conversation instead of requiring the representative to manually rewrite the call summary.

📊 Conversation Metrics

The system calculates quantitative conversation metrics including:

Representative word count
Customer word count
Representative talk ratio
Customer talk ratio
Overall speaking pace
Representative speaking pace
Representative filler words
Customer filler words

Example:

Representative Talk Ratio: 56.8%
Customer Talk Ratio: 43.2%
Overall Pace: 117 WPM
Representative Pace: 117.3 WPM
Representative Filler Words: 0

These metrics provide additional context for call-quality analysis and coaching.

🖥️ Interactive Dashboard

The project includes a Streamlit dashboard for visualizing sales-call intelligence.

Dashboard Sections
📊 Overview

Provides an aggregate view of:

Total calls
Average representative talk ratio
At-risk calls
Extraction performance
Calls by domain
Talk-ratio comparison
At-risk calls
Transcript search
🔍 Call Detail

Provides a detailed view of an individual call:

Transcript
Conversation metrics
BANT analysis
Deal risk
Extracted moments
Coaching feedback
Follow-up email
📈 Extraction Accuracy

Displays:

Precision
Recall
F1 score
Category-level extraction performance
Evaluation results
👤 Rep Summary

Provides aggregate performance information across different sales domains.

🧪 Validation & Evaluation

The project includes evaluation and validation components to measure extraction quality and reduce unsupported AI outputs.

Extraction performance is evaluated using:

Precision
Recall
F1 Score

The system also uses deterministic validation layers after AI generation.

LLM Output
    ↓
Validation
    ↓
Validated Output
    ↓
Next Agent

This architecture improves the reliability of the multi-agent pipeline.

🛠️ Technology Stack
Programming
Python
AI / LLM
Groq API
OpenAI GPT-OSS 120B
Data Processing
Pandas
Validation
Pydantic
Dashboard
Streamlit
Plotly
Environment & Version Control
Python-dotenv
Git
GitHub
📁 Project Structure
sales_call_intelligence/
│
├── agents/
│   ├── coaching_agent.py
│   ├── extraction_agent.py
│   ├── llm_client.py
│   └── risk_agent.py
│
├── analysis/
│   └── talk_metrics.py
│
├── data/
│   ├── transcripts/
│   └── ground_truth.json
│
├── evaluation/
│   └── score_extraction.py
│
├── validation/
│   ├── extraction_validator.py
│   └── coaching_validator.py
│
├── dashboard/
│   └── app.py
│
├── outputs/
│
├── pipeline.py
├── run_all_metrics.py
├── dataset_index.csv
├── requirements.txt
├── README.md
└── .gitignore
⚙️ Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd sales_call_intelligence

Install dependencies:

pip install -r requirements.txt

Create a .env file:

GROQ_API_KEY=your_groq_api_key
▶️ Running the Pipeline

Run the complete AI pipeline:

python pipeline.py

The pipeline performs:

Transcript loading
Conversation metric calculation
Sales-moment extraction
Extraction validation
BANT and risk analysis
Personalized coaching
Coaching validation
Follow-up email generation
🖥️ Running the Dashboard

Start the Streamlit dashboard:

streamlit run dashboard/app.py

The dashboard provides an interactive interface for exploring the generated sales intelligence.

🔐 Security

API keys and environment variables are stored in .env.

The .env file must not be committed to GitHub.

Make sure .env is included in .gitignore.

⭐ Key Features
✅ Automated sales-call transcript analysis
✅ Timestamped sales-moment extraction
✅ Objection detection
✅ Pricing detection
✅ Competitor detection
✅ Customer need identification
✅ Budget analysis
✅ Authority analysis
✅ Timeline detection
✅ Next-step extraction
✅ BANT analysis
✅ Deal-risk detection
✅ Conversation metrics
✅ Personalized coaching
✅ Suggested coaching phrases
✅ Extraction validation
✅ Coaching validation
✅ Follow-up email generation
✅ Interactive Streamlit dashboard
✅ Extraction performance evaluation
🌟 Business Impact

The system helps sales teams move from manually reviewing conversations to an automated intelligence workflow.

Instead of simply storing sales-call transcripts, the system converts conversations into:

Conversation
     ↓
Insights
     ↓
Risk Detection
     ↓
Coaching
     ↓
Actionable Follow-up

This helps sales representatives understand customer concerns, identify missing BANT information, improve their conversations, and maintain clear follow-up actions.

🔮 Future Scope

Possible future improvements include:

Audio-to-text transcription using Whisper
Real-time sales-call analysis
CRM integration
Automated email sending
Multi-language transcript support
Historical representative-performance tracking
Advanced trend analysis
Automated manager alerts
More advanced agent orchestration
Call-to-call performance comparison
👥 Team

Project: Sales Call Intelligence & Coaching Agent

Team: NexAura

Team Members
Anisha Varpre
Shruti Khobre
Kasturi Kakade
📌 Conclusion

The Sales Call Intelligence & Coaching Agent demonstrates how Agentic AI can transform raw sales conversations into structured intelligence, risk analysis, personalized coaching, and actionable follow-up.

By combining LLM-based reasoning, deterministic validation, and quantitative conversation metrics, the system creates a complete workflow for understanding and improving sales calls.
