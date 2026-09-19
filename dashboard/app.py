import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

st.set_page_config(page_title="Sales Call Intelligence Dashboard", layout="wide", page_icon="📞")

st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #1C1F26;
        border: 1px solid #2D313A;
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #A0A4AB;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- PATHS ----------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TALK_METRICS_PATH = os.path.join(BASE, "outputs", "talk_metrics_summary.csv")
EVAL_PATH = os.path.join(BASE, "outputs", "extraction_evaluation_summary.csv")
GROUND_TRUTH_PATH = os.path.join(BASE, "data", "ground_truth.json")
TRANSCRIPTS_DIR = os.path.join(BASE, "data", "transcripts")

# ---------- LOAD DATA ----------
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

# Extract domain from call_id (e.g. "saas_001" -> "saas")
def get_domain(call_id):
    return call_id.rsplit("_", 1)[0]

talk_df["domain"] = talk_df["call_id"].apply(get_domain)

# Add risk flag from ground_truth bant_gap
def get_risk(call_id):
    gt = ground_truth.get(call_id, {})
    return gt.get("bant_gap", {}).get("present", False)

talk_df["at_risk"] = talk_df["call_id"].apply(get_risk)

# Average F1 score per call (across all 4 categories)
avg_f1_per_call = eval_df.groupby("call_id")["f1_score"].mean().reset_index()
avg_f1_per_call.columns = ["call_id", "avg_f1"]
talk_df = talk_df.merge(avg_f1_per_call, on="call_id", how="left")

# ---------- SIDEBAR NAV ----------
st.sidebar.title("📞 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Call Detail", "Extraction Accuracy"])

# ==========================================================
# PAGE 1: OVERVIEW
# ==========================================================
if page == "Overview":
    st.title("📞 Sales Call Intelligence & Coaching — Overview")
    min_risk_calls = st.slider("Minimum risk calls to highlight", 0, 10, 3)
    if int(talk_df["at_risk"].sum()) >= min_risk_calls:
        st.error(f"🚨 {int(talk_df['at_risk'].sum())} calls need attention this week")


    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Calls", len(talk_df))
    col2.metric("Avg Rep Talk Ratio", f"{talk_df['rep_talk_ratio_pct'].mean():.1f}%")
    col3.metric("At-Risk Deals", int(talk_df["at_risk"].sum()))
    col4.metric("Avg Extraction F1", f"{talk_df['avg_f1'].mean():.2f}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Calls by Domain")
        domain_counts = talk_df["domain"].value_counts().reset_index()
        domain_counts.columns = ["domain", "count"]
        fig = px.bar(domain_counts, x="domain", y="count")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Talk Ratio: Rep vs Customer")
        fig2 = px.bar(talk_df, x="call_id",
                       y=["rep_talk_ratio_pct", "customer_talk_ratio_pct"],
                       barmode="stack")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("⚠️ At-Risk Calls")
    risky = talk_df[talk_df["at_risk"] == True][["call_id", "domain", "rep_talk_ratio_pct", "avg_f1"]]
    st.dataframe(risky, use_container_width=True)

# ==========================================================
# PAGE 2: CALL DETAIL
# ==========================================================
elif page == "Call Detail":
    st.title("🔍 Call Detail View")
    domain_filter = st.selectbox("Filter by domain", ["All"] + sorted(talk_df["domain"].unique().tolist()))
    if domain_filter != "All":
        filtered_calls = talk_df[talk_df["domain"] == domain_filter]["call_id"].tolist()
    else:
        filtered_calls = talk_df["call_id"].tolist()
    
    selected_call = st.selectbox("Choose a call", filtered_calls)

    call_row = talk_df[talk_df["call_id"] == selected_call].iloc[0]
    gt = ground_truth.get(selected_call, {})

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Transcript")
        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{selected_call}.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                st.text(f.read())
        else:
            st.warning("Transcript file not found.")

    with col2:
        st.subheader("Call Metrics")
        st.write(f"**Domain:** {call_row['domain']}")
        st.write(f"**Rep Talk Ratio:** {call_row['rep_talk_ratio_pct']}%")
        st.write(f"**Customer Talk Ratio:** {call_row['customer_talk_ratio_pct']}%")
        st.write(f"**Rep Pace:** {call_row['rep_pace_wpm']} wpm")
        st.write(f"**Rep Filler Words:** {call_row['rep_filler_count']}")

        st.subheader("Risk Flag")
        if gt.get("bant_gap", {}).get("present", False):
            st.error(f"⚠️ At Risk — {gt['bant_gap'].get('reason', '')}")
        else:
            st.success(f"✅ Healthy — {gt.get('bant_gap', {}).get('reason', '')}")

        st.subheader("Extracted Moments")
        st.write("**Objections:**", gt.get("objections", []) or "None")
        st.write("**Pricing Mentions:**", gt.get("pricing_mentions", []) or "None")
        st.write("**Competitor Mentions:**", gt.get("competitor_mentions", []) or "None")
        st.write("**Next Steps:**", gt.get("next_steps", []) or "None")

    st.divider()
    st.subheader("Extraction Accuracy for This Call")
    call_eval = eval_df[eval_df["call_id"] == selected_call]
    st.dataframe(call_eval[["category", "precision", "recall", "f1_score"]], use_container_width=True)

# ==========================================================
# PAGE 3: EXTRACTION ACCURACY (across all calls)
# ==========================================================
elif page == "Extraction Accuracy":
    st.title("📊 Extraction Accuracy — Across All Calls")

    st.subheader("Average F1 Score by Category")
    avg_by_category = eval_df.groupby("category")["f1_score"].mean().reset_index()
    fig = px.bar(avg_by_category, x="category", y="f1_score", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    weakest = avg_by_category.loc[avg_by_category["f1_score"].idxmin()]
    st.warning(f"Weakest category: **{weakest['category']}** (F1 = {weakest['f1_score']:.2f}) — needs prompt improvement.")

    st.divider()
    st.subheader("Full Evaluation Table")
    st.dataframe(eval_df, use_container_width=True)