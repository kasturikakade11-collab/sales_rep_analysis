import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

st.set_page_config(page_title="Sales Call Intelligence Dashboard", layout="wide", page_icon="📞")

st.markdown("""
<div style="padding: 4px 0 20px 0;">
    <h1 style="margin-bottom: 2px; border-bottom: 3px solid #3B82F6; padding-bottom: 10px; display: inline-block;">📞 Sales Call Intelligence</h1>
    <p style="font-size: 14px; color: #9AA1AC; margin-top: 0;">
        AI-powered call coaching, deal-risk detection, and rep performance analytics
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', -apple-system, sans-serif; }
    .main { background-color: #0B0E14; padding-top: 1rem; }
    section[data-testid="stSidebar"] {
        background-color: #10131B;
        border-right: 1px solid #252A36;
    }
    h1 { font-weight: 800; font-size: 30px; color: #E8EAED; letter-spacing: -0.5px; margin-bottom: 4px; }
    h2, h3 { font-weight: 600; color: #E8EAED; }
    p, span, label { color: #9AA1AC; }
    div[data-testid="stMetric"] {
        background: #151922;
        border: 1px solid #252A36;
        border-radius: 12px;
        padding: 18px 20px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 12px; font-weight: 600; color: #9AA1AC;
        text-transform: uppercase; letter-spacing: 0.6px;
    }
    div[data-testid="stMetricValue"] { font-size: 26px; font-weight: 700; color: #E8EAED; }
    div[data-testid="stDataFrame"] { border-radius: 10px; border: 1px solid #252A36; overflow: hidden; }
    .stAlert { border-radius: 10px; border-left: 4px solid; }
    hr { border-color: #252A36; margin: 1.6rem 0; }
    div[data-baseweb="select"] > div {
        background-color: #151922; border-color: #252A36; border-radius: 8px;
    }
    div[data-testid="stExpander"] { border: 1px solid #252A36; border-radius: 10px; }
        .stTextArea textarea { border-radius: 8px; font-family: 'Consolas', monospace; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #10131B;
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        color: #9AA1AC;
    }
    .stTabs [aria-selected="true"] {
        background-color: #151922;
        color: #E8EAED;
    }

    .card-box {
        background: #151922;
        border: 1px solid #252A36;
        border-left: 3px solid #3B82F6;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .card-box.risk-high { border-left-color: #EF4444; }
    .card-box.risk-low { border-left-color: #14B8A6; }
</style>
""", unsafe_allow_html=True)

# ---------- PATHS ----------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TALK_METRICS_PATH = os.path.join(BASE, "outputs", "talk_metrics_summary.csv")
EVAL_PATH = os.path.join(BASE, "outputs", "extraction_evaluation_summary.csv")
GROUND_TRUTH_PATH = os.path.join(BASE, "data", "ground_truth.json")
TRANSCRIPTS_DIR = os.path.join(BASE, "data", "transcripts")

CHART_COLORS = ["#3B82F6", "#14B8A6", "#6366F1", "#F59E0B"]

def theme_chart(fig):
    fig.update_layout(
        plot_bgcolor="#0B0E14",
        paper_bgcolor="#0B0E14",
        font_color="#E8EAED"
    )
    return fig


def generate_coaching_feedback(call_row, gt):
    feedback = []
    if call_row['rep_talk_ratio_pct'] > 58:
        feedback.append("⚠️ You talked significantly more than the customer — try asking more open-ended questions to let them share concerns.")
    if call_row['rep_filler_count'] > 3:
        feedback.append(f"⚠️ {call_row['rep_filler_count']} filler words detected — practice pausing instead of saying 'um'/'so'.")
    if gt.get('bant_gap', {}).get('present', False):
        feedback.append(f"⚠️ Deal risk: {gt['bant_gap'].get('reason', '')} — address this directly in your next touchpoint.")
    if not gt.get('pricing_mentions'):
        feedback.append("💡 Pricing wasn't discussed this call — confirm budget fit before the next call to avoid late-stage surprises.")
    if not feedback:
        feedback.append("✅ Solid call — good balance of talk time, no major risk signals detected.")
    return feedback


def generate_followup_email(call_row, gt):
    next_steps = gt.get('next_steps', [])
    pricing = gt.get('pricing_mentions', [])
    competitors = gt.get('competitor_mentions', [])

    lines = []
    lines.append("Subject: Great speaking with you — next steps\n")
    lines.append("Hi [Customer Name],\n")
    lines.append("Thank you for taking the time to speak with me today. Here's a quick recap of what we discussed:\n")

    if pricing:
        lines.append("Pricing discussed:")
        for p in pricing:
            lines.append(f"  - {p}")
        lines.append("")

    if competitors:
        lines.append("You mentioned you're also considering:")
        for c in competitors:
            lines.append(f"  - {c}")
        lines.append("I'd be happy to walk through how we compare whenever helpful.\n")

    if next_steps:
        lines.append("Next steps:")
        for step in next_steps:
            lines.append(f"  - {step}")
        lines.append("")
    else:
        lines.append("I'll follow up shortly with more details.\n")

    lines.append("Please let me know if you have any questions in the meantime.\n")
    lines.append("Best regards,\n[Your Name]")

    return "\n".join(lines)


@st.cache_data
def load_data():
    try:
        talk_df = pd.read_csv(TALK_METRICS_PATH)
        eval_df = pd.read_csv(EVAL_PATH)
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
        return talk_df, eval_df, ground_truth
    except FileNotFoundError as e:
        st.error(f"❌ Missing data file: {e}")
        st.stop()


with st.spinner("Loading call intelligence data..."):
    talk_df, eval_df, ground_truth = load_data()

if talk_df.empty:
    st.warning("⚠️ No call data found yet. Run the pipeline to generate calls first.")
    st.stop()


def get_domain(call_id):
    return call_id.rsplit("_", 1)[0]


talk_df["domain"] = talk_df["call_id"].apply(get_domain)


def get_risk(call_id):
    gt = ground_truth.get(call_id, {})
    return gt.get("bant_gap", {}).get("present", False)


talk_df["at_risk"] = talk_df["call_id"].apply(get_risk)

avg_f1_per_call = eval_df.groupby("call_id")["f1_score"].mean().reset_index()
avg_f1_per_call.columns = ["call_id", "avg_f1"]
talk_df = talk_df.merge(avg_f1_per_call, on="call_id", how="left")

# ---------- SIDEBAR NAV ----------
st.sidebar.title("📞 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Call Detail", "Extraction Accuracy", "Rep Summary"])

# ==========================================================
# PAGE 1: OVERVIEW
# ==========================================================
if page == "Overview":
    st.title("📊 Overview")
    st.caption("Real-time visibility into every rep's calls, risk signals, and coaching opportunities")

    min_risk_calls = st.slider("Minimum risk calls to highlight", 0, 10, 3)
    if int(talk_df["at_risk"].sum()) >= min_risk_calls:
        st.error(f"🚨 {int(talk_df['at_risk'].sum())} calls need attention this week")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📞 Total Calls", len(talk_df))
        col2.metric("🗣️ Avg Rep Talk Ratio", f"{talk_df['rep_talk_ratio_pct'].mean():.1f}%")
        col3.metric("⚠️ At-Risk Deals", int(talk_df["at_risk"].sum()))
        col4.metric("🎯 Avg Extraction F1", f"{talk_df['avg_f1'].mean():.2f}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Calls by Domain")
        domain_counts = talk_df["domain"].value_counts().reset_index()
        domain_counts.columns = ["domain", "count"]
        fig = px.bar(domain_counts, x="domain", y="count", color_discrete_sequence=CHART_COLORS)
        st.plotly_chart(theme_chart(fig), use_container_width=True)

    with c2:
        st.subheader("Talk Ratio: Rep vs Customer")
        fig2 = px.bar(talk_df, x="call_id",
                      y=["rep_talk_ratio_pct", "customer_talk_ratio_pct"],
                      barmode="stack", color_discrete_sequence=CHART_COLORS)
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(theme_chart(fig2), use_container_width=True)

    st.divider()
    st.subheader("⚠️ At-Risk Calls")
    risky = talk_df[talk_df["at_risk"] == True][["call_id", "domain", "rep_talk_ratio_pct", "avg_f1"]]
    st.dataframe(risky, use_container_width=True)

    st.divider()
    st.subheader("🔍 Search Across All Transcripts")
    search_term = st.text_input("Search for a keyword (e.g. 'budget', a competitor name, 'timeline')")
    if search_term:
        matches = []
        for call_id in talk_df["call_id"]:
            path = os.path.join(TRANSCRIPTS_DIR, f"{call_id}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    if search_term.lower() in f.read().lower():
                        matches.append(call_id)
        if matches:
            st.success(f"Found in {len(matches)} call(s): {', '.join(matches)}")
        else:
            st.info("No matches found.")

# ==========================================================
# PAGE 2: CALL DETAIL
# ==========================================================
elif page == "Call Detail":
    st.title("🔍 Call Detail View")
    st.caption("Full breakdown of one call — transcript, risk, coaching, and follow-up")

    with st.expander("🔗 Agent Pipeline Trace (click to expand)"):
        st.markdown("""
        1. **Ingestion Agent** — parsed raw transcript into speaker turns
        2. **Extraction Agent** — identified objections, pricing mentions, competitor mentions, next steps
        3. **Talk-Metrics Agent** — calculated talk ratio, pace, filler words
        4. **Risk-Detection Agent** — evaluated BANT gaps and flagged risk
        5. **Coaching Agent** — generated evidence-grounded coaching feedback
        6. **Email Agent** — drafted follow-up recap email from extracted next steps
        """)

    domain_filter = st.selectbox("Filter by domain", ["All"] + sorted(talk_df["domain"].unique().tolist()))
    if domain_filter != "All":
        filtered_calls = talk_df[talk_df["domain"] == domain_filter]["call_id"].tolist()
    else:
        filtered_calls = talk_df["call_id"].tolist()

    selected_call = st.selectbox("Choose a call", filtered_calls)

    call_row = talk_df[talk_df["call_id"] == selected_call].iloc[0]
    gt = ground_truth.get(selected_call, {})

    tab1, tab2, tab3 = st.tabs(["📄 Transcript", "🎯 Coaching & Risk", "✉️ Follow-Up"])

    with tab1:
        st.subheader("Transcript")
        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{selected_call}.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                st.text(f.read())
        else:
            st.warning("Transcript file not found.")

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rep Talk Ratio", f"{call_row['rep_talk_ratio_pct']}%")
        m2.metric("Customer Talk Ratio", f"{call_row['customer_talk_ratio_pct']}%")
        m3.metric("Rep Pace", f"{call_row['rep_pace_wpm']} wpm")
        m4.metric("Filler Words", int(call_row['rep_filler_count']))

    with tab2:
        is_risk = gt.get("bant_gap", {}).get("present", False)
        risk_class = "risk-high" if is_risk else "risk-low"
        risk_icon = "⚠️" if is_risk else "✅"
        risk_label = "At Risk" if is_risk else "Healthy"
        risk_reason = gt.get('bant_gap', {}).get('reason', '')

        st.markdown(f"""
        <div class="card-box {risk_class}">
            <strong>{risk_icon} {risk_label}</strong><br>
            <span style="font-size: 13px;">{risk_reason}</span>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🎯 Coaching Feedback")
        coaching_points = generate_coaching_feedback(call_row, gt)
        for point in coaching_points:
            st.markdown(f'<div class="card-box">{point}</div>', unsafe_allow_html=True)

        st.subheader("Extracted Moments")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("**Objections**")
            st.write(gt.get("objections", []) or "None")
            st.markdown("**Pricing Mentions**")
            st.write(gt.get("pricing_mentions", []) or "None")
        with ec2:
            st.markdown("**Competitor Mentions**")
            st.write(gt.get("competitor_mentions", []) or "None")
            st.markdown("**Next Steps**")
            st.write(gt.get("next_steps", []) or "None")

    with tab3:
        st.subheader("✉️ Follow-Up Email Draft")
        email_draft = generate_followup_email(call_row, gt)
        st.text_area("Editable draft — copy or edit as needed:", value=email_draft, height=350)

    st.divider()
    with st.expander("🔬 System Evaluation (for developers/judges)"):
        st.caption("Precision/recall/F1 scores comparing this call's AI extraction against labeled ground truth — not intended for rep-facing use.")
        call_eval = eval_df[eval_df["call_id"] == selected_call]
        if not call_eval.empty:
            st.dataframe(call_eval[["category", "precision", "recall", "f1_score"]], use_container_width=True)
        else:
            st.info("No evaluation data available for this call yet.")

# ==========================================================
# PAGE 3: EXTRACTION ACCURACY (across all calls)
# ==========================================================
elif page == "Extraction Accuracy":
    st.title("📈 Extraction Accuracy")
    st.caption("How reliable is the extraction agent across every call?")

    st.subheader("Average F1 Score by Category")
    avg_by_category = eval_df.groupby("category")["f1_score"].mean().reset_index()
    fig = px.bar(avg_by_category, x="category", y="f1_score", range_y=[0, 1],
                 color_discrete_sequence=CHART_COLORS)
    st.plotly_chart(theme_chart(fig), use_container_width=True)

    weakest = avg_by_category.loc[avg_by_category["f1_score"].idxmin()]
    st.warning(f"Weakest category: **{weakest['category']}** (F1 = {weakest['f1_score']:.2f}) — needs prompt improvement.")

    st.divider()
    st.subheader("Full Evaluation Table")
    st.dataframe(eval_df, use_container_width=True)

# ==========================================================
# PAGE 4: REP SUMMARY
# ==========================================================
elif page == "Rep Summary":
    st.title("👤 Performance Summary")
    st.caption("Aggregate view across domains — where risk concentrates and where coaching should focus")

    summary = talk_df.groupby("domain").agg(
        avg_talk_ratio=("rep_talk_ratio_pct", "mean"),
        avg_filler=("rep_filler_count", "mean"),
        at_risk_count=("at_risk", "sum"),
        calls=("call_id", "count")
    ).reset_index().sort_values("at_risk_count", ascending=False)

    st.dataframe(summary, use_container_width=True)

    fig = px.bar(summary, x="domain", y="at_risk_count",
                 title="At-Risk Calls by Domain",
                 color_discrete_sequence=CHART_COLORS)
    st.plotly_chart(theme_chart(fig), use_container_width=True)