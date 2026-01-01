# ==================== IMPORTS ====================
# Core libraries
import streamlit as st
from datetime import datetime, date, timedelta, time
from typing import List

# Data processing
import pandas as pd
import plotly.graph_objects as go

# App modules
from data_handler import load_data, save_data, clean_data, add_manual_task
from data_constants import COLUMN_ORDER
from data_preprocessing import filter_by_date_range
from utils import validate_dataframe, calculate_productivity_score, filter_recent_data

# Analytics & ML
from Analytics import get_peak_hours, get_weekly_summary, assess_burnout_risk, get_workload_recommendations
from ml_models import MLModelHandler
from insights import MLInsightsGenerator
from recommendations import TaskRecommender

# UI components
from streamlit_autorefresh import st_autorefresh
from charts import show_basic_charts
from productivity_charts import show_productivity_charts
from insight_charts import show_insight_charts

# Time series forecasting
from time_series_forecast import TimeSeriesForecaster

# ==================== HEADER ====================
st.set_page_config(page_title="🧠 NeuroTrack", layout="wide", initial_sidebar_state="expanded")

# ==================== CUSTOM CSS STYLING ====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
  
    
    /* Metrics Cards Enhancement */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    [data-testid="stMetricLabel"] {
        color: #A5B4FC !important;
        font-weight: 500;
    }
    
    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricDelta"] {
        color: #34D399 !important;
    }
    
    /* Primary Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.5);
    }
    
    /* Secondary Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 50%, #A855F7 100%);
        border-radius: 10px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
        border-radius: 10px;
        border: 1px solid rgba(139, 92, 246, 0.15);
    }
    
    /* Tabs - dark text for light backgrounds */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(99, 102, 241, 0.1);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #4338CA !important;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white !important;
    }
    
    /* Selectbox styling */
    [data-baseweb="select"] {
        border-radius: 8px;
    }
    
    /* Section Headers - dark purple for light backgrounds */
    h1, h2, h3 {
        color: #4338CA !important;
        font-weight: 700;
    }
    
    /* Make markdown text dark and visible on light bg */
    .stMarkdown p, .stMarkdown li {
        color: #334155;
    }
    
    /* Ensure text in main area is readable on light bg */
    .main .block-container {
        color: #1E293B;
    }
    
    /* Metric cards text - dark for light backgrounds */
    [data-testid="stMetricLabel"] {
        color: #6366F1 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #1E293B !important;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
    }
    
    /* Success/Info/Warning/Error boxes */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(34, 211, 238, 0.1) 100%);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 10px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(251, 191, 36, 0.1) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(248, 113, 113, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 10px;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Form styling */
    [data-testid="stForm"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 12px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header with branding
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.title("🧠 NeuroTrack")
    st.markdown("**Your AI-powered productivity companion** — Track, analyze, and optimize your time with intelligent insights")

with col2:
    st.info(f"📊 **Dashboard** | Last updated: {pd.Timestamp.now().strftime('%H:%M')}")

st.markdown("---")

# ==================== DAILY MOTIVATIONAL QUOTE ====================
import random
from data_constants import PRODUCTIVITY_QUOTES

# Use today's date as seed for consistent daily quote
random.seed(date.today().toordinal())
daily_quote = random.choice(PRODUCTIVITY_QUOTES)
random.seed()  # Reset seed

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
    border-left: 4px solid #6366F1;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 20px;
">
    <p style="font-size: 1.1rem; font-style: italic; color: #374151; margin: 0;">
        💡 "{daily_quote['text']}"
    </p>
    <p style="font-size: 0.9rem; color: #6366F1; margin: 8px 0 0 0; font-weight: 500;">
        — {daily_quote['author']}
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== Initialize Components ====================
if "ml_handler" not in st.session_state:
    st.session_state.ml_handler = MLModelHandler()
    st.session_state.task_recommender = TaskRecommender()
    st.session_state.insights_generator = MLInsightsGenerator()
    st.session_state.forecaster = TimeSeriesForecaster()
    st.session_state.ml_models_trained = False
    st.session_state.timer_running = False
    st.session_state.paused = False
    st.session_state.remaining_time = 0
    st.session_state.current_task = None
    # Initialize toggle states for sections
    st.session_state.setdefault('show_statistics', True)
    st.session_state.setdefault('show_visualizations', True)
    st.session_state.setdefault('show_performance', True)
    st.session_state.setdefault('show_goals', True)
    st.session_state.setdefault('show_import', True)
    st.session_state.setdefault('show_task_management', True)
    st.session_state.setdefault('show_insights', True)
    st.session_state.setdefault('show_forecasting', True)

# Load Data
try:
    data = load_data()
    if not data.empty:
        data = clean_data(data)  # Ensure data is properly cleaned
except Exception as e:
    st.error(f"Error loading data: {e}")
    data = pd.DataFrame(columns=COLUMN_ORDER)

# Define 'today' once after data loading
today = date.today()

# ==================== SIDEBAR INFO ====================
with st.sidebar:
    st.markdown("### 📌 Session Info")
    
    # Date and Session
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        st.write(f"👤 **Active**")
    with col_date2:
        st.write(f"📅 {date.today().strftime('%b %d')}")
    
    st.divider()
    
    # Quick Stats
    st.markdown("### 📊 Quick Stats")
    if not data.empty:
        today_tasks = len(data[pd.to_datetime(data["date"], errors='coerce').dt.date == today])
        week_start = today - timedelta(days=today.weekday())
        week_tasks = len(data[pd.to_datetime(data["date"], errors='coerce') >= pd.Timestamp(week_start)])
        completed_today = len(data[(pd.to_datetime(data["date"], errors='coerce').dt.date == today) & (data["completed"] == True)])
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("📋 Today", f"{today_tasks}", f"✅ {completed_today}")
        with col_stat2:
            st.metric("📅 This Week", f"{week_tasks}", delta="tasks")
    else:
        st.write("📊 No data available yet")
    
    st.divider()
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    if st.button("➕ New Task", use_container_width=True, type="primary"):
        st.info("Scroll down to 'Add New Task' section")
    
    if st.button("📥 Import Data", use_container_width=True):
        st.info("Scroll down to 'Import Data' section")
    
    st.divider()
    
    # Performance Badge
    if not data.empty and len(data) > 0:
        st.markdown("### 🎯 Performance")
        completion_rate = (data["completed"].sum() / len(data) * 100) if len(data) > 0 else 0
        
        if completion_rate >= 80:
            st.success(f"🏆 **Excellent** | {int(completion_rate)}%")
        elif completion_rate >= 60:
            st.info(f"⭐ **Good** | {int(completion_rate)}%")
        elif completion_rate >= 40:
            st.warning(f"💪 **Fair** | {int(completion_rate)}%")
        else:
            st.error(f"⚠️ **Needs Work** | {int(completion_rate)}%")
    
    st.divider()
    
    # Toggle Controls
    st.markdown("### ⚙️ Display Settings")
    st.markdown("Toggle sections to customize your view:")
    
    st.session_state.show_statistics = st.checkbox(
        "📈 Statistics", 
        value=st.session_state.show_statistics,
        help="Show/hide task statistics dashboard"
    )
    
    st.session_state.show_visualizations = st.checkbox(
        "📊 Visualizations", 
        value=st.session_state.show_visualizations,
        help="Show/hide data visualization tabs"
    )
    
    st.session_state.show_performance = st.checkbox(
        "🎯 Performance", 
        value=st.session_state.show_performance,
        help="Show/hide performance dashboard"
    )
    
    st.session_state.show_goals = st.checkbox(
        "🎯 Goals", 
        value=st.session_state.show_goals,
        help="Show/hide productivity goals"
    )
    
    st.session_state.show_task_management = st.checkbox(
        "📋 Task Management", 
        value=st.session_state.show_task_management,
        help="Show/hide task management section"
    )
    
    st.session_state.show_insights = st.checkbox(
        "🧠 ML Insights", 
        value=st.session_state.show_insights,
        help="Show/hide ML-powered insights"
    )
    
    st.session_state.show_forecasting = st.checkbox(
        "📈 Forecasting", 
        value=st.session_state.show_forecasting,
        help="Show/hide time series forecasting"
    )
    
    st.session_state.show_import = st.checkbox(
        "📁 Import Data", 
        value=st.session_state.show_import,
        help="Show/hide data import section"
    )
    
    # Quick toggle buttons
    st.markdown("#### Quick Actions")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if st.button("✅ Show All", use_container_width=True):
            st.session_state.show_statistics = True
            st.session_state.show_visualizations = True
            st.session_state.show_performance = True
            st.session_state.show_goals = True
            st.session_state.show_task_management = True
            st.session_state.show_insights = True
            st.session_state.show_forecasting = True
            st.session_state.show_import = True
            st.rerun()
    
    with col_t2:
        if st.button("❌ Hide All", use_container_width=True):
            st.session_state.show_statistics = False
            st.session_state.show_visualizations = False
            st.session_state.show_performance = False
            st.session_state.show_goals = False
            st.session_state.show_task_management = False
            st.session_state.show_insights = False
            st.session_state.show_forecasting = False
            st.session_state.show_import = False
            st.rerun()

# ==================== DATA VALIDATION ALERTS ====================
if not data.empty:
    validation_issues = []
    
    # Check for missing critical columns
    missing_cols = [col for col in COLUMN_ORDER if col not in data.columns]
    if missing_cols:
        validation_issues.append(f"Missing columns: {', '.join(missing_cols)}")
    
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        null_counts = data.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        validation_issues.append(f"Missing values in: {', '.join(null_cols.index.tolist())}")
    
    # Check for empty or whitespace-only tasks
    if "task" in data.columns and (data["task"].str.strip() == "").any():
        validation_issues.append("Found empty or whitespace-only tasks")
    
    # Show warnings if issues found
    if validation_issues:
        with st.expander("⚠️ Data Quality Warnings", expanded=False):
            for issue in validation_issues:
                st.warning(f"• {issue}")

# Responsive layout detection
is_mobile = st.session_state.get('is_mobile', False)


# ------------------ Dashboard Metrics ------------------

score, prod_time, total_time_overall, completion_rate_overall = calculate_productivity_score(data)
# Dashboard Metrics Display (Responsive)
metric_cols = st.columns(2) if is_mobile else st.columns(4)

with metric_cols[0]:
    # Determine gauge bar color based on score
    if score >= 90:
        bar_color = "#8B5CF6"  # Violet - Excellent
    elif score >= 70:
        bar_color = "#10B981"  # Emerald - High
    elif score >= 40:
        bar_color = "#F59E0B"  # Amber - Medium
    else:
        bar_color = "#EF4444"  # Rose - Low
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "🔥 Productivity Score", 'font': {'size': 16, 'color': '#A5B4FC'}},
        number={'font': {'size': 40, 'color': '#F8FAFC'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': '#6366F1', 'tickwidth': 2},
            'bar': {'color': bar_color, 'thickness': 0.8},
            'bgcolor': 'rgba(30, 27, 75, 0.3)',
            'borderwidth': 2,
            'bordercolor': 'rgba(139, 92, 246, 0.3)',
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 90], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [90, 100], 'color': 'rgba(139, 92, 246, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#A855F7", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=200, 
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#F8FAFC'}
    )
    st.plotly_chart(fig, use_container_width=True)

with metric_cols[1]:
    st.metric("⏱️ Focus Time", f"{int(prod_time)}m", f"/{int(total_time_overall)}m total")

if not is_mobile:
    with metric_cols[2]:
        st.metric("✅ Completion Rate", f"{completion_rate_overall}%")
    with metric_cols[3]:
        burnout_risk = assess_burnout_risk(data) if not data.empty else "Low"
        risk_color = "🔴" if burnout_risk == "High" else "🟡" if burnout_risk == "Medium" else "🟢"
        st.metric("🏥 Burnout Risk", f"{risk_color} {burnout_risk}")
else:
    metric_cols2 = st.columns(2)
    with metric_cols2[0]:
        st.metric("✅ Completion Rate", f"{completion_rate_overall}%")
    with metric_cols2[1]:
        burnout_risk = assess_burnout_risk(data) if not data.empty else "Low"
        risk_color = "🔴" if burnout_risk == "High" else "🟡" if burnout_risk == "Medium" else "🟢"
        st.metric("🏥 Burnout Risk", f"{risk_color} {burnout_risk}")

# ==================== ADVANCED FILTERS & SEARCH ====================
st.markdown("### 🔍 Filter & Search")

filter_cols = st.columns(5)

with filter_cols[0]:
    date_filter = st.selectbox(
        "📅 Date Range",
        options=["All Time", "Last 7 days", "Last 30 days", "Last 3 months", "Custom"],
        key="date_filter"
    )
    
    if date_filter == "Last 7 days":
        filter_start = today - timedelta(days=7)
    elif date_filter == "Last 30 days":
        filter_start = today - timedelta(days=30)
    elif date_filter == "Last 3 months":
        filter_start = today - timedelta(days=90)
    elif date_filter == "Custom":
        filter_start = st.date_input("From date:", today - timedelta(days=30))
    else:
        filter_start = None

with filter_cols[1]:
    categories = ["All"] + (data["category"].unique().tolist() if "category" in data.columns else [])
    selected_category = st.selectbox("📂 Category", options=categories, key="cat_filter")

with filter_cols[2]:
    priorities = ["All", "🔴 High", "🟠 Medium", "🟡 Low"]
    selected_priority = st.selectbox("⚡ Priority", options=priorities, key="priority_filter")

with filter_cols[3]:
    moods = ["All", "😊 Good", "😐 Neutral", "😢 Bad"]
    selected_mood = st.selectbox("😊 Mood", options=moods, key="mood_filter")

with filter_cols[4]:
    completion = st.selectbox("✅ Status", options=["All", "Completed", "Pending"], key="completion_filter")

# Apply filters to data
filtered_data = data.copy()

if filter_start is not None and "date" in filtered_data.columns:
    filtered_data = filtered_data[pd.to_datetime(filtered_data["date"], errors='coerce') >= pd.Timestamp(filter_start)]

if selected_category != "All" and "category" in filtered_data.columns:
    filtered_data = filtered_data[filtered_data["category"] == selected_category]

if selected_priority != "All" and "priority" in filtered_data.columns:
    priority_map = {"🔴 High": "High", "🟠 Medium": "Medium", "🟡 Low": "Low"}
    filtered_data = filtered_data[filtered_data["priority"] == priority_map[selected_priority]]

if selected_mood != "All" and "mood" in filtered_data.columns:
    mood_map = {"😊 Good": "Good", "😐 Neutral": "Neutral", "😢 Bad": "Bad"}
    filtered_data = filtered_data[filtered_data["mood"] == mood_map[selected_mood]]

if completion == "Completed" and "completed" in filtered_data.columns:
    filtered_data = filtered_data[filtered_data["completed"] == True]
elif completion == "Pending" and "completed" in filtered_data.columns:
    filtered_data = filtered_data[filtered_data["completed"] == False]

st.markdown(f"**📊 Showing {len(filtered_data)} of {len(data)} tasks**")
st.divider()

# ==================== TASK STATISTICS DASHBOARD ====================
if st.session_state.show_statistics:
    st.markdown("### 📈 Task Statistics")

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    if not filtered_data.empty:
        # Category breakdown
        with stats_col1:
            if "category" in filtered_data.columns:
                cat_counts = filtered_data["category"].value_counts()
                st.metric(
                    "📂 Categories",
                    cat_counts.index[0] if len(cat_counts) > 0 else "N/A",
                    f"{len(cat_counts)} types"
                )
            else:
                st.metric("📂 Categories", "N/A")
        
        # Total time spent
        with stats_col2:
            if "time_taken" in filtered_data.columns:
                total_time = filtered_data["time_taken"].sum()
                st.metric("⏱️ Total Time", f"{int(total_time)}m", f"~{int(total_time/60)}h")
            else:
                st.metric("⏱️ Total Time", "0m")
        
        # Completion streak
        with stats_col3:
            if "completed" in filtered_data.columns:
                completed = filtered_data["completed"].sum()
                total = len(filtered_data)
                streak_pct = (completed / total * 100) if total > 0 else 0
                st.metric("🔥 Completion Streak", f"{int(streak_pct)}%", f"{int(completed)}/{int(total)}")
            else:
                st.metric("🔥 Completion Streak", "0%")
        
        # Most productive category
        with stats_col4:
            if "category" in filtered_data.columns and "completed" in filtered_data.columns:
                cat_completion = filtered_data.groupby("category")["completed"].agg(['sum', 'count'])
                cat_completion['rate'] = (cat_completion['sum'] / cat_completion['count'] * 100).fillna(0)
                best_cat = cat_completion['rate'].idxmax() if len(cat_completion) > 0 else "N/A"
                best_rate = cat_completion['rate'].max() if len(cat_completion) > 0 else 0
                st.metric("⭐ Best Category", best_cat, f"{int(best_rate)}% done")
            else:
                st.metric("⭐ Best Category", "N/A")

    # ==================== STATISTICS VISUALIZATIONS ====================

    # Category breakdown pie chart
    if not filtered_data.empty and "category" in filtered_data.columns:
        st.markdown("#### 📊 Category Distribution")
        cat_data = filtered_data["category"].value_counts()
        
        # Use modern category colors from constants
        colors = ['#8B5CF6', '#06B6D4', '#10B981', '#F59E0B', '#EC4899', '#14B8A6', '#6366F1', '#F97316']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=cat_data.index, 
            values=cat_data.values, 
            hole=0.4,
            marker=dict(colors=colors[:len(cat_data)], line=dict(color='#1E1B4B', width=2)),
            textposition='inside',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Tasks: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig_pie.update_layout(
            height=350,
            showlegend=True,
            font=dict(size=12, color='#F8FAFC'),
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Time trend analysis
    if not filtered_data.empty and "date" in filtered_data.columns:
        st.markdown("#### ⏱️ Time Trend (Last 30 Days)")
        try:
            trend_data = filtered_data.copy()
            trend_data["date"] = pd.to_datetime(trend_data["date"], errors='coerce')
            daily_time = trend_data.groupby(trend_data["date"].dt.date)["time_taken"].sum().tail(30)
            
            fig_trend = go.Figure(data=[
                go.Scatter(x=daily_time.index, y=daily_time.values, mode='lines+markers', 
                          name='Minutes', line=dict(color='#8B5CF6', width=2),
                          marker=dict(size=6, color='#8B5CF6'))
            ])
            fig_trend.update_layout(
                title={'text': "Daily Time Investment", 'font': {'color': '#A5B4FC'}},
                xaxis_title="Date",
                yaxis_title="Minutes",
                height=300,
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#F8FAFC'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate trend: {e}")

    st.divider()

st.divider()

# ==================== PERFORMANCE DASHBOARD ====================
if st.session_state.show_performance:
    st.markdown("### 🎯 Performance Dashboard")

perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.markdown("#### 📈 Daily Productivity Score Trend")
    try:
        if not data.empty and "date" in data.columns:
            daily_data = data.copy()
            daily_data["date"] = pd.to_datetime(daily_data["date"], errors='coerce')
            daily_scores = daily_data.groupby(daily_data["date"].dt.date).apply(
                lambda x: (x["completed"].sum() / len(x) * 100) if len(x) > 0 else 0
            ).tail(30)
            
            fig_score = go.Figure(data=[
                go.Scatter(x=daily_scores.index, y=daily_scores.values, fill='tozeroy', 
                          name='Score', line=dict(color='#10B981', width=2),
                          fillcolor='rgba(16, 185, 129, 0.2)')
            ])
            fig_score.update_layout(
                xaxis_title="Date",
                yaxis_title="Productivity %",
                height=300,
                hovermode='x',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#F8FAFC'}
            )
            st.plotly_chart(fig_score, use_container_width=True)
    except Exception as e:
        st.info("Insufficient data for trend analysis")

with perf_col2:
    st.markdown("#### 📊 Priority Distribution")
    try:
        if not filtered_data.empty and "priority" in filtered_data.columns:
            priority_data = filtered_data["priority"].value_counts()
            colors = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
            fig_priority = go.Figure(data=[
                go.Bar(x=priority_data.index, y=priority_data.values, 
                       marker=dict(color=[colors.get(p, '#6366F1') for p in priority_data.index]))
            ])
            fig_priority.update_layout(
                title={'text': "Tasks by Priority", 'font': {'color': '#A5B4FC'}},
                xaxis_title="Priority",
                yaxis_title="Count",
                height=300,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#F8FAFC'}
            )
            st.plotly_chart(fig_priority, use_container_width=True)
    except Exception as e:
        st.info("No priority data available")

    st.divider()

# Goal tracking
if st.session_state.show_goals:
    st.markdown("#### 🎯 Productivity Goals")
    goal_col1, goal_col2, goal_col3 = st.columns(3)

    with goal_col1:
        daily_goal = st.number_input("Daily Task Goal", min_value=1, max_value=50, value=5, step=1)
        if not data.empty and "date" in data.columns:
            today_tasks = data[pd.to_datetime(data["date"], errors='coerce').dt.date == today]
            st.metric("Today's Progress", f"{len(today_tasks)}/{daily_goal}", "tasks")

    with goal_col2:
        weekly_goal = st.number_input("Weekly Time Goal (hours)", min_value=1, max_value=100, value=40, step=5)
        if not data.empty and "date" in data.columns:
            week_start = today - timedelta(days=today.weekday())
            week_data = data[pd.to_datetime(data["date"], errors='coerce') >= pd.Timestamp(week_start)]
            week_hours = week_data["time_taken"].sum() / 60
            st.metric("Week's Progress", f"{int(week_hours)}/{weekly_goal}h", "hours")

    with goal_col3:
        target_completion = st.number_input("Target Completion %", min_value=0, max_value=100, value=80, step=5)
        if not data.empty and "completed" in data.columns:
            completion_pct = (data["completed"].sum() / len(data) * 100) if len(data) > 0 else 0
            st.metric("Completion Rate", f"{int(completion_pct)}%", f"Target: {target_completion}%")

    # Goal progress bars
    st.markdown("#### 📊 Goal Progress")
    col_progress1, col_progress2 = st.columns(2)

    with col_progress1:
        if not data.empty and "date" in data.columns:
            today_tasks = data[pd.to_datetime(data["date"], errors='coerce').dt.date == today]
            daily_progress = min(len(today_tasks) / daily_goal, 1.0)
            st.progress(daily_progress, text=f"Daily Goal: {int(daily_progress*100)}%")

    with col_progress2:
        if not data.empty and "date" in data.columns:
            week_start = today - timedelta(days=today.weekday())
            week_data = data[pd.to_datetime(data["date"], errors='coerce') >= pd.Timestamp(week_start)]
            week_hours = week_data["time_taken"].sum() / 60
            weekly_progress = min(week_hours / weekly_goal, 1.0)
            st.progress(weekly_progress, text=f"Weekly Goal: {int(weekly_progress*100)}%")

    st.divider()

st.divider()

# ==================== DATA IMPORT SECTION ====================
if st.session_state.show_import:
    st.markdown("### 📁 Import Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], help="Upload a CSV file with your task data")

    if uploaded_file is not None:
        try:
            import_data = pd.read_csv(uploaded_file)
            import_data = clean_data(import_data)  # Clean imported data
            
            st.write("Preview of imported data:")
            st.dataframe(import_data.head())
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Import Data", type="primary"):
                    try:
                        # Combine with existing data
                        data = pd.concat([data, import_data], ignore_index=True)
                        data = clean_data(data)
                        save_data(data)
                        
                        # Reset ML models to retrain with new data
                        st.session_state.ml_models_trained = False
                        
                        st.success(f"✅ Successfully imported {len(import_data)} tasks!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error importing data: {e}")
            
            with col2:
                if st.button("Cancel Import"):
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    
    st.divider()

st.divider()

# ==================== TASK CREATION SECTION ====================
st.markdown("## ➕ Add New Task")

with st.expander("Create New Task", expanded=True):
    with st.form("task_creation_form"):
        # Section 1: Core Task Info
        st.markdown("### 🎯 Basic Info")
        col1, col2 = st.columns(2)
        with col1:
            task_name = st.text_input("Task Name*", help="Required field")
            task_date = st.date_input("Date*", datetime.now().date())
            
            # Dynamic category selection
            existing_cats = data["category"].unique().tolist()
            category = st.selectbox(
                "Category*", 
                options=["New Category"] + existing_cats,
                index=0 if "New Category" in existing_cats else 1
            )
            if category == "New Category":
                category = st.text_input("Enter New Category Name*")
                
        with col2:
            time_col1, time_col2 = st.columns(2)
            with time_col1:
                start_time = st.time_input("Start Time*", time(9, 0))
            with time_col2:
                duration = st.number_input("Duration (min)*", min_value=1, max_value=480, value=30)
            
            priority = st.selectbox(
                "Priority*", 
                options=["Low", "Medium", "High"], 
                index=1
            )

        # Section 2: Advanced Metrics
        st.markdown("### 📊 Productivity Metrics")
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            mood = st.selectbox(
                "Mood", 
                options=["😊 Happy", "😐 Neutral", "😞 Tired", "😤 Frustrated", "💪 Energized"],
                index=1
            )
        with adv_col2:
            energy_level = st.slider(
                "Energy Level (1-10)", 
                min_value=1, max_value=10, value=5
            )
        with adv_col3:
            focus_level = st.slider(
                "Focus Level (1-10)", 
                min_value=1, max_value=10, value=5
            )

        # Section 3: Additional Details
        st.markdown("### 📝 Details")
        difficulty = st.slider("Difficulty (1-5)", min_value=1, max_value=5, value=3)
        intent = st.selectbox(
            "Intent", 
            options=["Complete", "Learn", "Review", "Plan","Practice","Explore"],
            index=0
        )
        
        # Tag input with chips
        tags_input = st.text_input(
            "Tags (comma separated)", 
            help="e.g. urgent,project-x,backend"
        )
        tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []
        
        notes = st.text_area("Additional Notes")

        # Form submission
        if st.form_submit_button("Add Task", type="primary"):
            try:
                data = add_manual_task(
                    df=data,
                    task_name=task_name,
                    time_taken=duration,
                    task_date=task_date,
                    start_time_obj=start_time,
                    category=category,
                    priority=priority,
                    mood=mood,
                    energy_level=energy_level,
                    focus_level=focus_level,
                    intent=intent,
                    difficulty=difficulty,
                    tags=tags,
                    notes=notes
                )
                st.success(f"Task '{task_name}' added successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error adding task: {str(e)}")

st.divider()

# ==================== AI INSIGHTS SECTION ====================
if not data.empty and len(data) > 5 and st.session_state.show_insights:
    st.markdown("## 🤖 AI Insights")
    
    insight_tabs = st.tabs(["📊 Peak Hours", "📝 Weekly Summary", "⚖️ Workload Balance", "🧠 ML Insights"])
    
    with insight_tabs[0]:
        try:
            peak_hours = get_peak_hours(data)
            if peak_hours:
                st.markdown("#### ⏰ Your Peak Performance Hours")
                for i, (hour_range, productivity) in enumerate(peak_hours[:3]):
                    emoji = "🔥" if i == 0 else "⚡" if i == 1 else "💡"
                    st.markdown(f"{emoji} **{hour_range}**: {productivity:.1f}% productivity")
                
                best_time = peak_hours[0][0] if peak_hours else "9-11 AM"
                st.info(f"💡 **Schedule important tasks during {best_time} for maximum efficiency**")
        except Exception as e:
            st.error(f"Error calculating peak hours: {e}")
    
    with insight_tabs[1]:
        try:
            weekly_summary = get_weekly_summary(data)
            if weekly_summary:
                st.markdown("#### 📈 This Week's Analysis")
                st.markdown(weekly_summary)
        except Exception as e:
            st.error(f"Error generating weekly summary: {e}")
    
    with insight_tabs[2]:
        try:
            recommendations = get_workload_recommendations(data)
            if recommendations:
                st.markdown("#### ⚖️ Workload Balance")
                for rec_type, message in recommendations.items():
                    icon = {"warning": "⚠️", "suggestion": "💡", "positive": "✅"}.get(rec_type, "ℹ️")
                    st.markdown(f"{icon} {message}")
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
    
    with insight_tabs[3]:
        try:
            ml_insights = st.session_state.insights_generator.generate_insights(data)
            if ml_insights:
                st.markdown("#### 🧠 Machine Learning Insights")
                for insight_type, message in ml_insights.items():
                    if insight_type == "status":
                        if "ready" in message.lower():
                            st.success(f"🚀 {message}")
                        elif "need more data" in message.lower():
                            st.info(f"📊 {message}")
                        else:
                            st.warning(f"⚠️ {message}")
                    else:
                        icon = {"difficulty": "🎯", "energy": "⚡", "time_pattern": "⏱️", "category": "📂"}.get(insight_type, "🔍")
                        st.markdown(f"{icon} **{insight_type.replace('_', ' ').title()}**: {message}")
        except Exception as e:
            st.error(f"Error generating ML insights: {e}")

# ==================== DATA VISUALIZATION ====================
if not data.empty and st.session_state.show_visualizations:
    st.markdown("## 📊 Data Visualization")
    tab1, tab2, tab3 = st.tabs(["📈 Basic Overview", "💪 Productivity Metrics", "🔍 Deep Insights"])
    
    with st.spinner("Rendering visualizations..."):
        with tab1:
            st.markdown("### Task Overview & Trends")
            show_basic_charts(data)
            
        with tab2:
            st.markdown("### Productivity Analysis")
            show_productivity_charts(data)
            
        with tab3:
            st.markdown("### Advanced Insights")
            show_insight_charts(data)

st.divider()

# ==================== TIME SERIES FORECASTING ====================
if not data.empty and len(data) >= 3 and st.session_state.show_forecasting:
    st.markdown("## 📈 Time Series Forecasting")
    st.markdown("Predict future productivity trends based on historical patterns")
    
    # Warning for limited data
    if len(data) < 7:
        st.warning(f"⚠️ Limited data detected ({len(data)} tasks). Forecasts will be more accurate with 7+ days of data.")
    
    # Forecast settings
    col_settings1, col_settings2 = st.columns([2, 3])
    with col_settings1:
        forecast_horizon = st.slider("Forecast Horizon (days)", min_value=3, max_value=30, value=7, step=1)
    
    with col_settings2:
        st.info(f"📅 Forecasting from {(date.today() + timedelta(days=1)).strftime('%B %d')} to {(date.today() + timedelta(days=forecast_horizon)).strftime('%B %d, %Y')}")
    
    try:
        forecaster = st.session_state.forecaster
        
        # Generate forecast summary
        with st.spinner("Generating forecasts..."):
            summary = forecaster.get_forecast_summary(data, horizon=forecast_horizon)
        
        # Display summary metrics
        if summary:
            st.markdown("### 🎯 Forecast Summary")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                if 'productivity' in summary:
                    prod = summary['productivity']
                    trend_icon = "📈" if "increasing" in prod.get('trend', '') else "📉"
                    st.metric(
                        "Avg Productivity", 
                        f"{prod.get('avg_forecast', 0):.1f}%",
                        delta=f"{prod.get('trend', 'stable')} {trend_icon}"
                    )
            
            with metric_col2:
                if 'tasks' in summary:
                    tasks = summary['tasks']
                    st.metric(
                        "Expected Tasks", 
                        f"{tasks.get('total_tasks_forecast', 0)}",
                        delta=f"{tasks.get('avg_daily_tasks', 0)} per day"
                    )
            
            with metric_col3:
                if 'workload' in summary:
                    wl = summary['workload']
                    trend_icon = "📈" if "increasing" in wl.get('trend', '') else "📉"
                    st.metric(
                        "Total Workload", 
                        f"{wl.get('total_hours_forecast', 0):.1f}h",
                        delta=f"{wl.get('avg_daily_hours', 0):.1f}h/day {trend_icon}"
                    )
            
            with metric_col4:
                if 'completion' in summary:
                    comp = summary['completion']
                    trend_icon = "✅" if "improving" in comp.get('trend', '') else "⚠️"
                    st.metric(
                        "Completion Rate", 
                        f"{comp.get('avg_rate', 0):.1f}%",
                        delta=f"{comp.get('trend', 'stable')} {trend_icon}"
                    )
        
        st.divider()
        
        # Forecast charts
        st.markdown("### 📊 Forecast Visualizations")
        
        forecast_tabs = st.tabs([
            "📊 Productivity Score", 
            "⏱️ Daily Workload", 
            "📋 Task Count",
            "✅ Completion Rate"
        ])
        
        with forecast_tabs[0]:
            st.markdown("#### Productivity Score Forecast")
            hist_score, forecast_score = forecaster.forecast_productivity_score(data, horizon=forecast_horizon)
            if hist_score is not None and forecast_score is not None:
                fig = forecaster.create_forecast_chart(
                    hist_score.tail(30), 
                    forecast_score, 
                    "Productivity Score: Historical & Forecast",
                    "Score (%)",
                    color='#10B981'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Insights
                    avg_forecast = forecast_score.mean()
                    avg_historical = hist_score.tail(7).mean()
                    change = ((avg_forecast - avg_historical) / avg_historical * 100) if avg_historical > 0 else 0
                    
                    if change > 5:
                        st.success(f"✨ Productivity expected to improve by {change:.1f}% compared to last week!")
                    elif change < -5:
                        st.warning(f"⚠️ Productivity may decline by {abs(change):.1f}%. Consider adjusting workload.")
                    else:
                        st.info(f"📊 Productivity expected to remain stable ({change:+.1f}% change)")
            else:
                st.info("Insufficient data for productivity score forecast (need at least 7 days)")
        
        with forecast_tabs[1]:
            st.markdown("#### Daily Workload Forecast")
            hist_workload, forecast_workload = forecaster.forecast_workload(data, horizon=forecast_horizon)
            if hist_workload is not None and forecast_workload is not None:
                fig = forecaster.create_forecast_chart(
                    hist_workload.tail(30), 
                    forecast_workload, 
                    "Daily Workload: Historical & Forecast",
                    "Minutes",
                    color='#8B5CF6'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Workload insights
                    total_forecast = forecast_workload.sum()
                    total_hours = total_forecast / 60
                    avg_daily = forecast_workload.mean()
                    
                    col_w1, col_w2, col_w3 = st.columns(3)
                    with col_w1:
                        st.metric("Total Forecasted Time", f"{total_hours:.1f} hours")
                    with col_w2:
                        st.metric("Average Daily", f"{avg_daily:.0f} minutes")
                    with col_w3:
                        burnout_risk = "High" if avg_daily > 480 else "Moderate" if avg_daily > 360 else "Low"
                        risk_color = "🔴" if burnout_risk == "High" else "🟡" if burnout_risk == "Moderate" else "🟢"
                        st.metric("Burnout Risk", f"{burnout_risk} {risk_color}")
            else:
                st.info("Insufficient data for workload forecast")
        
        with forecast_tabs[2]:
            st.markdown("#### Task Count Forecast")
            hist_tasks, forecast_tasks = forecaster.forecast_task_count(data, horizon=forecast_horizon)
            if hist_tasks is not None and forecast_tasks is not None:
                fig = forecaster.create_forecast_chart(
                    hist_tasks.tail(30), 
                    forecast_tasks, 
                    "Daily Task Count: Historical & Forecast",
                    "Number of Tasks",
                    color='#06B6D4'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Task count insights
                    total_tasks = int(forecast_tasks.sum())
                    avg_tasks = forecast_tasks.mean()
                    
                    if avg_tasks > hist_tasks.tail(7).mean() * 1.2:
                        st.warning(f"⚠️ Task load increasing: {total_tasks} tasks expected (avg {avg_tasks:.1f}/day)")
                    elif avg_tasks < hist_tasks.tail(7).mean() * 0.8:
                        st.info(f"📉 Task load decreasing: {total_tasks} tasks expected (avg {avg_tasks:.1f}/day)")
                    else:
                        st.success(f"✅ Steady task flow: {total_tasks} tasks expected (avg {avg_tasks:.1f}/day)")
            else:
                st.info("Insufficient data for task count forecast")
        
        with forecast_tabs[3]:
            st.markdown("#### Completion Rate Forecast")
            hist_completion, forecast_completion = forecaster.forecast_completion_rate(data, horizon=forecast_horizon)
            if hist_completion is not None and forecast_completion is not None:
                fig = forecaster.create_forecast_chart(
                    hist_completion.tail(30), 
                    forecast_completion, 
                    "Completion Rate: Historical & Forecast",
                    "Completion Rate (%)",
                    color='#F59E0B'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Completion insights
                    avg_completion = forecast_completion.mean()
                    
                    if avg_completion >= 80:
                        st.success(f"🏆 Excellent! Expected completion rate: {avg_completion:.1f}%")
                    elif avg_completion >= 60:
                        st.info(f"👍 Good completion rate expected: {avg_completion:.1f}%")
                    else:
                        st.warning(f"⚠️ Low completion rate expected: {avg_completion:.1f}%. Consider reducing task load.")
            else:
                st.info("Insufficient data for completion rate forecast")
        
        st.divider()
        
        # Recommendations based on forecast
        st.markdown("### 💡 Forecast-Based Recommendations")
        
        recommendations = []
        
        if summary:
            # Workload recommendations
            if 'avg_daily_workload' in summary:
                avg_work = summary['avg_daily_workload']
                if avg_work > 480:
                    recommendations.append("🔴 **High Workload Alert**: Expected daily workload exceeds 8 hours. Consider delegating or postponing non-critical tasks.")
                elif avg_work > 360:
                    recommendations.append("🟡 **Moderate Workload**: 6+ hours/day expected. Ensure adequate breaks and prioritize high-impact tasks.")
                else:
                    recommendations.append("🟢 **Sustainable Workload**: Workload appears manageable. Good opportunity to tackle challenging tasks.")
            
            # Productivity recommendations
            if 'productivity_trend' in summary:
                if summary['productivity_trend'] == 'decreasing':
                    recommendations.append("📉 **Productivity Declining**: Consider reviewing energy management, eliminating distractions, or adjusting task difficulty.")
                elif summary['productivity_trend'] == 'increasing':
                    recommendations.append("📈 **Positive Momentum**: Productivity is improving! Maintain current habits and consider tackling stretch goals.")
            
            # Completion rate recommendations
            if 'avg_completion_rate' in summary:
                if summary['avg_completion_rate'] < 60:
                    recommendations.append("⚠️ **Low Completion Expected**: Reduce task load or break large tasks into smaller, manageable pieces.")
                elif summary['avg_completion_rate'] > 85:
                    recommendations.append("✨ **High Performance**: Strong completion rate expected. Consider taking on more challenging or strategic work.")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.info("📊 Forecasts look stable. Continue with your current approach.")
            
    except Exception as e:
        st.error(f"Error generating forecasts: {e}")
        st.info("💡 Tip: Time series forecasting works best with 7+ days of data, but will show estimates with 3+ tasks.")

elif st.session_state.show_forecasting and not data.empty and len(data) < 3:
    st.markdown("## 📈 Time Series Forecasting")
    st.info("📊 Add at least 3 tasks to enable time series forecasting predictions")

# ==================== TASK MANAGEMENT SECTION ====================
if not data.empty and st.session_state.show_task_management:
    st.markdown("### 📋 Task Management")
    st.divider()
    
    st.markdown("#### Recent Tasks")
    filter_category = st.selectbox("Filter by Category", ["All"] + list(data["category"].unique()))
    show_completed = st.checkbox("Show completed tasks", value=True)
    
    filtered_data = data.copy()
    if filter_category != "All":
        filtered_data = filtered_data[filtered_data["category"] == filter_category]
    if not show_completed:
        filtered_data = filtered_data[filtered_data["completed"] == False]
    
    display_data = filtered_data.sort_values("date", ascending=False).head(20)
    
    if not display_data.empty:
        for idx, task in display_data.iterrows():
            col1, col2, col3 = st.columns([0.65, 0.25, 0.1])
            
            with col1:
                # Status badge
                status_emoji = "✅" if task["completed"] else "⏳"
                status_color = "green" if task["completed"] else "orange"
                
                # Priority color badge
                priority = str(task.get("priority", "Medium")).upper()
                priority_color = "🔴" if "HIGH" in priority else "🟠" if "MEDIUM" in priority else "🟡"
                
                # Display task with badges
                st.write(f"{status_emoji} **{task['task']}** {priority_color} `{task['category']}` — {int(task['time_taken'])}m")
            
            
            with col2:
                if not task["completed"]:
                    if st.button("Mark Complete", key=f"complete_{idx}"):
                        try:
                            data.at[idx, "completed"] = True
                            save_data(data)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error updating task: {e}")
            
            with col3:
                if st.button("🗑️", key=f"delete_{idx}", help="Delete task"):
                    try:
                        data = data.drop(idx).reset_index(drop=True)
                        save_data(data)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting task: {e}")
    else:
        st.info("No tasks found with current filters.")

# Smart task recommendations
if not data.empty:
    st.markdown("### 💡 Task Recommendations")
    with st.expander("Get personalized suggestions", expanded=True):
        rec_df = data.copy()
        exclude_completed = st.checkbox("Exclude completed tasks", value=True, key="rec_exclude_completed")
        if exclude_completed and "completed" in rec_df.columns:
            rec_df = rec_df[rec_df["completed"] != True]

        if len(rec_df) < 5:
            st.info("Need at least 5 tasks to compare. Add more history or broaden your filters.")
        else:
            recent_choices = rec_df.sort_values("date", ascending=False)
            context_task = st.selectbox(
                "Base recommendations on",
                recent_choices["task"].unique(),
                help="Pick a task you recently did; similar ones will be suggested"
            )

            colr1, colr2, colr3 = st.columns([0.4, 0.3, 0.3])
            with colr1:
                override_category = st.selectbox(
                    "Focus category",
                    ["Auto"] + sorted(rec_df["category"].dropna().unique().tolist())
                )
            with colr2:
                override_priority = st.selectbox(
                    "Target priority",
                    ["Auto", "Low", "Medium", "High"]
                )
            with colr3:
                top_n = st.slider("How many suggestions", min_value=3, max_value=8, value=5, step=1)

            if context_task:
                context_row = recent_choices[recent_choices["task"] == context_task].iloc[0].to_dict()
                if override_category != "Auto":
                    context_row["category"] = override_category
                if override_priority != "Auto":
                    context_row["priority"] = override_priority

                with st.form("task_recs_form"):
                    st.caption("Uses similarity across text, category, priority, mood, intent, and difficulty.")
                    generate = st.form_submit_button("Generate recommendations", type="primary")

                    if generate:
                        recs = st.session_state.task_recommender.recommend_tasks(
                            context_row,
                            rec_df,
                            top_n=top_n,
                            exclude_completed=exclude_completed
                        )

                        if recs.empty:
                            st.warning("No close matches found. Try a different context task or include completed tasks.")
                        else:
                            recs = recs.copy()
                            if "similarity_score" in recs.columns:
                                recs["similarity_score"] = recs["similarity_score"].round(3)

                            display_cols = [
                                col for col in ["task", "category", "priority", "time_taken", "similarity_score", "reason"]
                                if col in recs.columns
                            ]
                            st.dataframe(
                                recs[display_cols].rename(columns={
                                    "task": "Suggested Task",
                                    "time_taken": "Estimated Minutes",
                                    "similarity_score": "Similarity",
                                    "reason": "Why this"
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
# Focus Timer
st.markdown("## 🎯 Focus Timer")
today_tasks = data[(data["date"] == today) & (data["completed"] == False)]

if not today_tasks.empty:
    task_options = today_tasks["task"].unique().tolist()
    focus_task = st.selectbox("Pick a task to focus on:", task_options, key="selected_focus_task")

    if focus_task:
        task_row = today_tasks[today_tasks["task"] == focus_task].iloc[0]
        estimated_time = int(task_row["time_taken"])
        st.write(f"⏱️ Estimated time: {estimated_time} mins")

        if st.button("▶️ Start Timer") and not st.session_state.timer_running:
            st.session_state.timer_running = True
            st.session_state.remaining_time = estimated_time * 60
            st.session_state.current_task = focus_task

        if st.session_state.timer_running and st.session_state.current_task == focus_task:
            countdown = st_autorefresh(interval=1000, key="timer_refresh")
            
            if countdown and not st.session_state.paused:
                st.session_state.remaining_time -= 1
            
            mins = st.session_state.remaining_time // 60
            secs = st.session_state.remaining_time % 60
            
            st.progress((estimated_time * 60 - st.session_state.remaining_time) / (estimated_time * 60))
            st.markdown(f"⏳ Time Left: **{mins:02}:{secs:02}**")
            
            if st.button("⏸ Pause"):
                st.session_state.paused = True
            if st.button("🛑 Reset"):
                st.session_state.timer_running = False
                st.session_state.remaining_time = 0
            
            if st.session_state.remaining_time <= 0:
                st.session_state.timer_running = False
                st.success("🎉 Time's up!")
                task_idx = today_tasks[today_tasks["task"] == focus_task].index[0]
                data.at[task_idx, "completed"] = True
                save_data(data)
                st.rerun()
else:
    st.info("✅ All tasks for today are completed!")

st.markdown("---")

# Export Data
st.markdown("## 📤 Export Data")
if not data.empty:
    csv_data = data.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"neurotrack_data_{today.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ using Streamlit & scikit-learn</p>
    <p><small>🧠 NeuroTrack v1.0 | Your AI Productivity Partner | © 2025</small></p>
</div>
""", unsafe_allow_html=True) 