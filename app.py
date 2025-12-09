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

# ==================== HEADER ====================
st.set_page_config(page_title="🧠 NeuroTrack", layout="wide", initial_sidebar_state="expanded")

# Enhanced header with branding
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.title("🧠 NeuroTrack")
    st.markdown("**Your AI-powered productivity companion** — Track, analyze, and optimize your time with intelligent insights")

with col2:
    st.info(f"📊 **Dashboard** | Last updated: {pd.Timestamp.now().strftime('%H:%M')}")

st.markdown("---")

# ==================== Initialize Components ====================
if "ml_handler" not in st.session_state:
    st.session_state.ml_handler = MLModelHandler()
    st.session_state.task_recommender = TaskRecommender()
    st.session_state.insights_generator = MLInsightsGenerator()
    st.session_state.ml_models_trained = False
    st.session_state.timer_running = False
    st.session_state.paused = False
    st.session_state.remaining_time = 0
    st.session_state.current_task = None

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
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "🔥 Productivity Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#4e79a7"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "darkgray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score}}))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=10))
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
    
    # Define category colors for consistency
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=cat_data.index, 
        values=cat_data.values, 
        hole=0.4,
        marker=dict(colors=colors[:len(cat_data)], line=dict(color='white', width=2)),
        textposition='inside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Tasks: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig_pie.update_layout(
        height=350,
        showlegend=True,
        font=dict(size=12),
        margin=dict(l=0, r=0, t=30, b=0)
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
                      name='Minutes', line=dict(color='#4e79a7', width=2))
        ])
        fig_trend.update_layout(
            title="Daily Time Investment",
            xaxis_title="Date",
            yaxis_title="Minutes",
            height=300,
            hovermode='x unified'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate trend: {e}")

st.divider()

# ==================== PERFORMANCE DASHBOARD ====================
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
                          name='Score', line=dict(color='#2ecc71', width=2))
            ])
            fig_score.update_layout(
                xaxis_title="Date",
                yaxis_title="Productivity %",
                height=300,
                hovermode='x'
            )
            st.plotly_chart(fig_score, use_container_width=True)
    except Exception as e:
        st.info("Insufficient data for trend analysis")

with perf_col2:
    st.markdown("#### 📊 Priority Distribution")
    try:
        if not filtered_data.empty and "priority" in filtered_data.columns:
            priority_data = filtered_data["priority"].value_counts()
            colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#f1c40f"}
            fig_priority = go.Figure(data=[
                go.Bar(x=priority_data.index, y=priority_data.values, 
                       marker=dict(color=[colors.get(p, '#95a5a6') for p in priority_data.index]))
            ])
            fig_priority.update_layout(
                title="Tasks by Priority",
                xaxis_title="Priority",
                yaxis_title="Count",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_priority, use_container_width=True)
    except Exception as e:
        st.info("No priority data available")

# Goal tracking
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

# ==================== DATA IMPORT SECTION ====================
st.markdown("### 📁 Import Data")
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
if not data.empty and len(data) > 5:
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
if not data.empty:
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

# ==================== TASK MANAGEMENT SECTION ====================
if not data.empty:
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