"""
Carbon-Aware Data Center Job Classification and Scheduling System
Module: dashboard/app.py
Purpose: Interactive Streamlit Dashboard showcasing 7 comprehensive sections:
         1. Overview Metrics & KPI Cards
         2. Machine Learning Performance (Metrics, Confusion Matrix, Feature Importance)
         3. Grid Carbon Intensity Time-Series & Diurnal Analysis
         4. Workload Scheduling Timeline & Gantt View
         5. Emissions Comparison across Baseline, True-Label, and Predicted-Label Strategies
         6. Delay Analysis, False Negatives, and Deadline Compliance
         7. Interactive Parameter Controls and Live What-If Simulation
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# Add project root to sys.path so modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator import generate_workloads, generate_carbon_intensity
from src.preprocessing import clean_and_validate_jobs, load_and_preprocess_carbon_intensity
from src.train_model import train_and_select_model, evaluate_classifier
from src.feature_engineering import extract_features
from src.simulator import WorkloadSimulator
from src.evaluation import generate_evaluation_report


st.set_page_config(
    page_title="Carbon-Aware Data Center Workload Scheduler",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #10B981;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 6px 6px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_base_data():
    jobs_path = "data/jobs.csv"
    ci_path = "data/carbon_intensity.csv"
    
    if not os.path.exists(ci_path):
        generate_carbon_intensity(output_path=ci_path)
    if not os.path.exists(jobs_path):
        generate_workloads(n_jobs=5000, output_path=jobs_path)
        
    ci_df = load_and_preprocess_carbon_intensity(ci_path)
    jobs_df = pd.read_csv(jobs_path)
    jobs_df = clean_and_validate_jobs(jobs_df)
    return jobs_df, ci_df


@st.cache_resource(show_spinner=False)
def load_trained_model():
    model_path = "models/urgency_classifier.joblib"
    if not os.path.exists(model_path):
        train_and_select_model()
    return joblib.load(model_path)


# --- SIDEBAR: INTERACTIVE CONTROLS (Section 7) ---
st.sidebar.image("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=300&q=80", use_container_width=True)
st.sidebar.title("🎛️ Simulation Controls")

st.sidebar.markdown("### Workload & Grid Scope")
jobs_df_raw, ci_df = load_base_data()
model_package = load_trained_model()

total_available_jobs = len(jobs_df_raw)
n_jobs_selected = st.sidebar.slider(
    "Simulated Jobs Count",
    min_value=500,
    max_value=total_available_jobs,
    value=min(3000, total_available_jobs),
    step=250,
    help="Select the subset size of data center jobs to simulate."
)

st.sidebar.markdown("### Scheduling Optimization Parameters")
max_delay_hours = st.sidebar.slider(
    "Max Permissible Delay (hours)",
    min_value=1.0,
    max_value=24.0,
    value=12.0,
    step=0.5,
    help="Upper bound on scheduling delay for flexible, non-urgent workloads."
)

st.sidebar.markdown("### Machine Learning Parameters")
model_choice = st.sidebar.selectbox(
    "Classifier Architecture",
    options=["Random Forest (Primary)", "Logistic Regression (Baseline)"],
    index=0
)
model_code = "rf" if "Random Forest" in model_choice else "lr"

calibrated_default = model_package.get("calibrated_threshold", 0.40)
classification_threshold = st.sidebar.slider(
    "Urgency Decision Threshold",
    min_value=0.10,
    max_value=0.90,
    value=float(calibrated_default),
    step=0.05,
    help="Lower threshold classifies more jobs as Urgent, reducing false negatives and preventing SLA violations."
)

st.sidebar.markdown("### Safety Guardrails")
use_safety_guard = st.sidebar.checkbox(
    "Enable Deadline Slack Safety Override",
    value=True,
    help="Automatically force Urgent execution if deadline slack is under safety threshold, guarding against false negatives."
)

min_safety_slack = st.sidebar.slider(
    "Min Safety Slack (hours)",
    min_value=0.5,
    max_value=6.0,
    value=2.0,
    step=0.5,
    disabled=not use_safety_guard,
    help="Jobs with slack less than this threshold are never delayed."
)


# Filter jobs dataset to selected subset
active_jobs_df = jobs_df_raw.iloc[:n_jobs_selected].copy().reset_index(drop=True)

# Run Simulation Engine
sim = WorkloadSimulator(active_jobs_df, ci_df)
with st.spinner("Simulating Carbon-Aware Workload Execution across 3 Strategies..."):
    combined_res, summary_df, raw_strategies = sim.run_simulation(
        max_delay_hours=max_delay_hours,
        classification_threshold=classification_threshold,
        use_safety_guard=use_safety_guard,
        min_safety_slack_hours=min_safety_slack,
        model_choice=model_code
    )

baseline_row = summary_df[summary_df["Strategy"].str.contains("Baseline")].iloc[0]
true_row = summary_df[summary_df["Strategy"].str.contains("True Labels")].iloc[0]
pred_row = summary_df[summary_df["Strategy"].str.contains("Predicted Labels")].iloc[0]


# --- MAIN HEADER ---
st.markdown('<div class="main-header">🌱 Carbon-Aware Data Center Job Classification & Scheduling</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-driven workload urgency prediction and spatial-temporal emissions minimization without compromising urgent service-level agreements.</div>', unsafe_allow_html=True)


# --- SECTION 1: OVERVIEW METRICS ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(
        label="Total Jobs Processed",
        value=f"{len(combined_res):,}",
        delta=f"{int(combined_res['true_urgency'].sum())} Urgent"
    )
with col2:
    st.metric(
        label="Baseline Emissions",
        value=f"{baseline_row['Total Emissions (kg CO2)']:,.1f} kg",
        help="Emissions if all workloads execute immediately upon arrival."
    )
with col3:
    st.metric(
        label="Predicted ML Emissions",
        value=f"{pred_row['Total Emissions (kg CO2)']:,.1f} kg",
        delta=f"-{pred_row['Carbon Reduction (%)']:.1f}% Reduction",
        delta_color="normal"
    )
with col4:
    st.metric(
        label="True Oracle Emissions",
        value=f"{true_row['Total Emissions (kg CO2)']:,.1f} kg",
        delta=f"-{true_row['Carbon Reduction (%)']:.1f}% Oracle Max",
        delta_color="normal"
    )
with col5:
    urgent_delayed_count = int(pred_row["Urgent Jobs Delayed"])
    st.metric(
        label="Urgent Jobs Delayed",
        value=f"{urgent_delayed_count}",
        delta="0 SLA Target" if urgent_delayed_count == 0 else f"{urgent_delayed_count} Violations",
        delta_color="off" if urgent_delayed_count == 0 else "inverse"
    )

st.markdown("---")

# Navigation Tabs
tabs = st.tabs([
    "📊 Executive Overview",
    "🤖 ML Urgency Classifier",
    "⚡ Carbon Intensity Dynamics",
    "📅 Workload Scheduling Timeline",
    "🌍 Emissions & Savings Comparison",
    "⏱️ Delay & SLA Safety Analysis"
])


# --- TAB 1: EXECUTIVE OVERVIEW ---
with tabs[0]:
    st.subheader("Strategy Comparison Matrix")
    st.caption("Side-by-side evaluation of immediate baseline execution vs Oracle True-Label scheduler vs Production ML-Predicted scheduler.")
    
    # Styled display of summary_df
    st.dataframe(
        summary_df.style.format({
            "Total Energy (kWh)": "{:,.2f}",
            "Total Emissions (kg CO2)": "{:,.2f}",
            "Carbon Reduction (%)": "{:.2f}%",
            "Average Job Delay (hours)": "{:.2f} h",
            "Maximum Job Delay (hours)": "{:.2f} h",
            "Urgent Jobs Delayed (%)": "{:.2f}%",
            "Avg Delay of Urgent Jobs (hours)": "{:.2f} h",
            "Max Delay of Urgent Jobs (hours)": "{:.2f} h"
        }).highlight_max(subset=["Carbon Reduction (%)"], color="#D1FAE5"),
        use_container_width=True
    )
    
    st.markdown("### Core Carbon-Aware Principles Demonstrated")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "**1. Workload Urgency Independence:**\n\n"
            "Urgent workloads (APIs, high-SLA transactions) execute immediately without delay. "
            "Prioritizing urgent jobs does not reduce emissions; emissions are cut by shifting flexible jobs."
        )
    with c2:
        st.info(
            "**2. Energy Conservation Integrity:**\n\n"
            f"Total computing energy consumed is strictly invariant: **{baseline_row['Total Energy (kWh)']:,.1f} kWh** across all 3 strategies. "
            "All carbon savings result entirely from cleaner electricity grid mix timing."
        )
    with c3:
        st.info(
            "**3. Asymmetric Classification Risk:**\n\n"
            "False negatives (marking urgent jobs as flexible) can breach SLAs. The model is tuned for high recall "
            f"(threshold: {classification_threshold:.2f}) with a safety guard, keeping urgent delays at **{urgent_delayed_count}**."
        )


# --- TAB 2: MACHINE LEARNING PERFORMANCE ---
with tabs[1]:
    st.subheader("Workload Urgency Classification Engine")
    st.caption("Evaluation of supervised models predicting job urgency prior to scheduling decisions without leaking carbon-intensity signals.")
    
    feat_df = extract_features(active_jobs_df)
    X_feat = model_package["pipeline"].transform(feat_df)
    y_true = feat_df["urgency_label"].values
    
    active_clf = model_package["baseline_model"] if model_code == "lr" else model_package["model"]
    probs = active_clf.predict_proba(X_feat)[:, 1]
    
    # Evaluate at current threshold
    metrics_curr = evaluate_classifier(y_true, probs, threshold=classification_threshold, model_name=model_choice)
    
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("Accuracy", f"{metrics_curr['accuracy']*100:.1f}%")
    m_col2.metric("Precision", f"{metrics_curr['precision']*100:.1f}%")
    m_col3.metric("Recall (Urgency)", f"{metrics_curr['recall']*100:.1f}%", help="Critical metric: high recall eliminates false negatives.")
    m_col4.metric("F1-Score", f"{metrics_curr['f1_score']:.3f}")
    m_col5.metric("ROC-AUC", f"{metrics_curr['roc_auc']:.3f}")
    m_col6.metric("False Negatives", f"{metrics_curr['false_negatives']}", delta_color="inverse")
    
    row_ml_1, row_ml_2 = st.columns(2)
    
    with row_ml_1:
        st.markdown("#### Confusion Matrix")
        cm_data = [
            [metrics_curr["true_negatives"], metrics_curr["false_positives"]],
            [metrics_curr["false_negatives"], metrics_curr["true_positives"]]
        ]
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Urgency", y="True Urgency", color="Count"),
            x=["Non-Urgent (0)", "Urgent (1)"],
            y=["Non-Urgent (0)", "Urgent (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=380, margin=dict(l=40, r=40, t=30, b=30))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Note: False Negatives (bottom-left) are the highest operational risk. Low classification threshold suppresses them.")

    with row_ml_2:
        st.markdown("#### Feature Importance (Random Forest)")
        importances_df = model_package.get("feature_importances", pd.DataFrame())
        if not importances_df.empty:
            top_feats = importances_df.head(10).sort_values(by="importance", ascending=True)
            fig_imp = px.bar(
                top_feats,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Viridis",
                labels={"importance": "Gini Importance", "feature": "Feature"}
            )
            fig_imp.update_layout(height=380, showlegend=False, margin=dict(l=40, r=40, t=30, b=30))
            st.plotly_chart(fig_imp, use_container_width=True)


# --- TAB 3: CARBON INTENSITY DYNAMICS ---
with tabs[2]:
    st.subheader("Grid Electricity Carbon Intensity Time-Series")
    st.caption("Hourly grams of CO2 emitted per kWh of electricity consumed, revealing diurnal solar troughs and evening gas peaks.")
    
    ci_plot_df = ci_df.copy()
    high_threshold = ci_plot_df["carbon_intensity_gco2_kwh"].quantile(0.75)
    low_threshold = ci_plot_df["carbon_intensity_gco2_kwh"].quantile(0.25)
    
    fig_ci = go.Figure()
    
    # Main CI Line
    fig_ci.add_trace(go.Scatter(
        x=ci_plot_df["timestamp"],
        y=ci_plot_df["carbon_intensity_gco2_kwh"],
        mode="lines",
        line=dict(color="#2563EB", width=2),
        name="Grid Carbon Intensity"
    ))
    
    # High Carbon Threshold Band
    fig_ci.add_hline(
        y=high_threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"High Carbon Threshold ({high_threshold:.0f} gCO2/kWh)",
        annotation_position="top left"
    )
    
    # Low Carbon Threshold Band
    fig_ci.add_hline(
        y=low_threshold,
        line_dash="dash",
        line_color="#10B981",
        annotation_text=f"Low Carbon Threshold ({low_threshold:.0f} gCO2/kWh)",
        annotation_position="bottom left"
    )
    
    fig_ci.update_layout(
        title="Hourly Grid Carbon Intensity Profile (gCO2/kWh)",
        xaxis_title="Simulation Timestamp",
        yaxis_title="gCO2 / kWh",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified"
    )
    st.plotly_chart(fig_ci, use_container_width=True)
    
    # Hourly diurnal profile boxplot
    ci_plot_df["hour"] = ci_plot_df["timestamp"].dt.hour
    hourly_avg = ci_plot_df.groupby("hour")["carbon_intensity_gco2_kwh"].mean().reset_index()
    
    fig_diurnal = px.bar(
        hourly_avg,
        x="hour",
        y="carbon_intensity_gco2_kwh",
        color="carbon_intensity_gco2_kwh",
        color_continuous_scale="RdYlGn_r",
        labels={"hour": "Hour of Day (00:00 - 23:00)", "carbon_intensity_gco2_kwh": "Average gCO2/kWh"},
        title="Diurnal Carbon Profile (Solar Midday Trough vs Evening Peak)"
    )
    fig_diurnal.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_diurnal, use_container_width=True)


# --- TAB 4: WORKLOAD SCHEDULING TIMELINE ---
with tabs[3]:
    st.subheader("Job Scheduling Timeline (Gantt-Style)")
    st.caption("Visualizing arrival times, scheduled execution intervals, and carbon-aware shifts for a representative sample of jobs.")
    
    # Sample 30 jobs for clear timeline rendering
    timeline_sample = combined_res.head(35).copy()
    
    timeline_data = []
    for _, row in timeline_sample.iterrows():
        j_id = row["job_id"]
        is_urg = row["predicted_urgency"] == 1
        category = "Urgent (Immediate)" if is_urg else "Non-Urgent (Shifted)"
        
        timeline_data.append(dict(
            Job=f"{j_id} ({row.get('job_type', 'Job')[:14]})",
            Start=row["scheduled_start_time"],
            Finish=row["scheduled_start_time"] + pd.Timedelta(hours=row["execution_duration"]),
            Category=category,
            Delay=row["scheduling_delay_hours"],
            Duration=row["execution_duration"],
            Arrival=row["arrival_time"]
        ))
        
    df_timeline = pd.DataFrame(timeline_data)
    
    fig_gantt = px.timeline(
        df_timeline,
        x_start="Start",
        x_end="Finish",
        y="Job",
        color="Category",
        color_discrete_map={
            "Urgent (Immediate)": "#EF4444",
            "Non-Urgent (Shifted)": "#10B981"
        },
        hover_data=["Arrival", "Delay", "Duration"],
        title="Execution Windows by Job Category"
    )
    fig_gantt.update_yaxes(autorange="reversed")
    fig_gantt.update_layout(height=650, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_gantt, use_container_width=True)
    st.caption("Green bars denote flexible workloads intelligently delayed into low-emission solar windows. Red bars denote urgent jobs starting with 0 delay.")


# --- TAB 5: EMISSIONS & SAVINGS COMPARISON ---
with tabs[4]:
    st.subheader("Emissions Comparison & Carbon Reduction")
    st.caption("Comparing aggregate carbon emissions across the three strategies.")
    
    col_em1, col_em2 = st.columns([1, 1])
    
    with col_em1:
        st.markdown("#### Total Workload Carbon Emissions (kg CO2)")
        bar_data = pd.DataFrame({
            "Strategy": ["As-Run Baseline", "Predicted Labels (ML)", "True Labels (Oracle)"],
            "Emissions (kg CO2)": [
                baseline_row["Total Emissions (kg CO2)"],
                pred_row["Total Emissions (kg CO2)"],
                true_row["Total Emissions (kg CO2)"]
            ],
            "Carbon Reduction (%)": [
                0.0,
                pred_row["Carbon Reduction (%)"],
                true_row["Carbon Reduction (%)"]
            ]
        })
        
        fig_bars = px.bar(
            bar_data,
            x="Strategy",
            y="Emissions (kg CO2)",
            text=bar_data["Carbon Reduction (%)"].apply(lambda x: f"-{x:.1f}%" if x > 0 else "Baseline"),
            color="Strategy",
            color_discrete_sequence=["#64748B", "#3B82F6", "#10B981"]
        )
        fig_bars.update_traces(textposition="outside")
        fig_bars.update_layout(height=420, showlegend=False, margin=dict(l=40, r=40, t=30, b=30))
        st.plotly_chart(fig_bars, use_container_width=True)

    with col_em2:
        st.markdown("#### Cumulative Emissions Over Time")
        # Build cumulative emissions time series
        res_sorted = combined_res.sort_values(by="scheduled_start_time").copy()
        res_sorted["cum_baseline"] = res_sorted["baseline_emissions"].cumsum()
        res_sorted["cum_optimized"] = res_sorted["optimized_emissions"].cumsum()
        
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=res_sorted["scheduled_start_time"],
            y=res_sorted["cum_baseline"],
            mode="lines",
            name="Baseline Cumulative",
            line=dict(color="#64748B", dash="dash")
        ))
        fig_cum.add_trace(go.Scatter(
            x=res_sorted["scheduled_start_time"],
            y=res_sorted["cum_optimized"],
            mode="lines",
            name="Carbon-Aware Cumulative",
            line=dict(color="#10B981", width=2.5)
        ))
        fig_cum.update_layout(
            title="Cumulative Carbon Footprint Growth",
            xaxis_title="Time",
            yaxis_title="Cumulative kg CO2",
            height=420,
            margin=dict(l=40, r=40, t=30, b=30)
        )
        st.plotly_chart(fig_cum, use_container_width=True)


# --- TAB 6: DELAY & SLA SAFETY ANALYSIS ---
with tabs[5]:
    st.subheader("Scheduling Delays & SLA Violation Risk")
    st.caption("Deep-dive inspection of job waiting periods, false-negative delay impacts, and deadline compliance.")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("#### Non-Urgent Jobs Delay Distribution")
        non_urgent_delays = combined_res[combined_res["scheduling_delay_hours"] > 0]["scheduling_delay_hours"]
        
        fig_delay_hist = px.histogram(
            non_urgent_delays,
            nbins=24,
            title=f"Shift Hours Distribution (Max Delay = {max_delay_hours} hrs)",
            labels={"value": "Delay Hours"},
            color_discrete_sequence=["#3B82F6"]
        )
        fig_delay_hist.update_layout(height=380, showlegend=False, margin=dict(l=40, r=40, t=30, b=30))
        st.plotly_chart(fig_delay_hist, use_container_width=True)
        
    with col_d2:
        st.markdown("#### Urgent Workload Integrity & False Negatives")
        true_urgent_jobs = combined_res[combined_res["true_urgency"] == 1]
        false_neg_delayed = true_urgent_jobs[true_urgent_jobs["scheduling_delay_hours"] > 0]
        
        fn_count = len(false_neg_delayed)
        st.write(f"**Total True Urgent Jobs:** {len(true_urgent_jobs):,}")
        st.write(f"**Urgent Jobs Inadvertently Delayed:** {fn_count}")
        st.write(f"**Deadline Violations Incurred:** {int(combined_res['deadline_violation'].sum())}")
        
        if fn_count > 0:
            st.warning(f"⚠️ {fn_count} urgent jobs experienced delayed execution. Average delay: {false_neg_delayed['scheduling_delay_hours'].mean():.2f} hours.")
            st.dataframe(false_neg_delayed[["job_id", "job_type", "scheduling_delay_hours", "urgency_probability", "safety_override"]].head(10))
        else:
            st.success("✅ Perfect Urgent Workload Integrity: 0 true urgent jobs were delayed! All critical SLAs were met without compromise.")
            
        st.markdown("""
        **Safety Mechanisms in Effect:**
        1. **Asymmetric Decision Threshold:** Calibration favors recall over precision.
        2. **Slack Guardrail:** Automatically forces immediate execution when deadline slack is tight.
        3. **Deadline Constrained Bound:** Even delayed jobs are bounded by `deadline - duration`.
        """)

st.markdown("---")
st.caption("Carbon-Aware Data Center Workload Scheduler | Developed with Python, Scikit-learn, Streamlit & Plotly.")
