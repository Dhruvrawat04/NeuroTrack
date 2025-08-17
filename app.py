import streamlit as st
from datetime import datetime, date, timedelta, time
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_handler import load_data, save_data, clean_data, COLUMN_ORDER, add_manual_task
from utils import validate_dataframe, calculate_productivity_score, filter_recent_data
from Analytics import get_peak_hours, get_weekly_summary, assess_burnout_risk, get_workload_recommendations
from ml_models import MLModelHandler
from insights import MLInsightsGenerator
from recommendations import TaskRecommender
from charts import show_basic_charts
from productivity_charts import show_productivity_charts
from insight_charts import show_insight_charts

# ------------------ Setup ------------------
st.set_page_config(page_title="🧠 NeuroTrack", layout="wide")
st.title("🧠 NeuroTrack")
st.markdown("**AI-powered productivity tracker** - Track, analyze, and optimize your time")

# ------------------ Initialize Components ------------------
if "ml_handler" not in st.session_state:
    st.session_state.ml_handler = MLModelHandler()
    st.session_state.task_recommender = TaskRecommender()
    st.session_state.insights_generator = MLInsightsGenerator()
    st.session_state.ml_models_trained = False

# ------------------ Load Data ------------------
try:
    data = load_data()
    if not data.empty:
        data = clean_data(data)  # Ensure data is properly cleaned
except Exception as e:
    st.error(f"Error loading data: {e}")
    data = pd.DataFrame(columns=COLUMN_ORDER)

# Define 'today' once after data loading
today = date.today()


# ------------------ Dashboard Metrics ------------------

score, prod_time, total_time_overall, completion_rate_overall = calculate_productivity_score(data)
# Dashboard Metrics Display
col1, col2, col3, col4 = st.columns(4)
with col1:
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

with col2:
    st.metric("⏱️ Focus Time", f"{int(prod_time)}m", f"/{int(total_time_overall)}m total")
with col3:
    st.metric("✅ Completion Rate", f"{completion_rate_overall}%")
with col4:
    burnout_risk = assess_burnout_risk(data) if not data.empty else "Low"
    risk_color = "🔴" if burnout_risk == "High" else "🟡" if burnout_risk == "Medium" else "🟢"
    st.metric("🏥 Burnout Risk", f"{risk_color} {burnout_risk}")

# ------------------ Data Import Section ------------------
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

#------------------- manual entry ----------------------
# Add to imports at top
from datetime import time
from typing import List

# Add this under Task Management section
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
# ------------------ AI Insights Section ------------------
if not data.empty and len(data) > 5:
    st.markdown("## 🤖 AI Insights")
    
    insight_tabs = st.tabs(["📊 Peak Hours", "📝 Weekly Summary", "⚖️ Workload Balance", "🧠 ML Insights"])
    
    with insight_tabs[0]:
        try:
            peak_hours = get_peak_hours(data)
            if peak_hours:
                st.markdown("### ⏰ Your Peak Performance Hours")
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
                st.markdown("### 📈 This Week's Analysis")
                st.markdown(weekly_summary)
        except Exception as e:
            st.error(f"Error generating weekly summary: {e}")
    
    with insight_tabs[2]:
        try:
            recommendations = get_workload_recommendations(data)
            if recommendations:
                st.markdown("### ⚖️ Workload Balance")
                for rec_type, message in recommendations.items():
                    icon = {"warning": "⚠️", "suggestion": "💡", "positive": "✅"}.get(rec_type, "ℹ️")
                    st.markdown(f"{icon} {message}")
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
    
    with insight_tabs[3]:
        try:
            ml_insights = st.session_state.insights_generator.generate_insights(data)
            if ml_insights:
                st.markdown("### 🧠 Machine Learning Insights")
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

# ------------------ Charts Section ------------------
if not data.empty:
    st.markdown("## 📊 Data Visualization")
    tab1, tab2, tab3 = st.tabs(["Basic Overview", "Productivity Metrics", "Deep Insights"])
    
    with tab1:
        show_basic_charts(data)
        
    with tab2:
        show_productivity_charts(data)
        
    with tab3:
        show_insight_charts(data)

# ------------------ Task Management Section ------------------
if not data.empty:
    st.markdown("### 📋 Task Management")
    
    col1, col2, col3, col4 = st.columns(4)
    recent_data = data[data["date"] >= (date.today() - timedelta(days=7))]
    
    with col1:
        st.metric("Total Tasks", len(data))
    with col2:
        st.metric("This Week", len(recent_data))
    with col3:
        st.metric("Total Hours", f"{data['time_taken'].sum()/60:.1f}")
    with col4:
        if len(recent_data) > 0:
            avg_daily = recent_data.groupby("date")["time_taken"].sum().mean()
            st.metric("Daily Avg", f"{avg_daily/60:.1f}h")
        else:
            st.metric("Daily Avg", "0h")
    
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
            col1, col2, col3 = st.columns([0.7, 0.2, 0.1])
            
            with col1:
                status_icon = "✅" if task["completed"] else "⏳"
                st.write(f"{status_icon} **{task['task']}** - {task['date']} ({int(task['time_taken'])}m, {task['category']})")
            
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

# Add to task management section (~line 300)
if not data.empty and st.checkbox("💡 Get task recommendations"):
    sample_task = st.selectbox("Based on:", data["task"].unique())
    
    if sample_task:
        recs = st.session_state.task_recommender.recommend_tasks(
            {"task": sample_task, "category": data[data["task"] == sample_task].iloc[0]["category"]},
            data,
            top_n=3
        )
        st.write("Try these similar tasks:")
        st.dataframe(recs[["task", "category", "time_taken", "similarity_score"]])
# ------------------ Focus Timer ------------------
st.markdown("## 🎯 Focus Timer")
today_tasks = data[(data["date"] == today) & (data["completed"] == False)]

if "timer_running" not in st.session_state:
    st.session_state.timer_running = False
if "remaining_time" not in st.session_state:
    st.session_state.remaining_time = 0
if "paused" not in st.session_state:
    st.session_state.paused = False
if "current_task" not in st.session_state:
    st.session_state.current_task = None

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

# ------------------ Data Export ------------------
if not data.empty:
    st.markdown("## 📤 Export Data")
    csv_data = data.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"neurotrack_data_{today.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | 🧠 NeuroTrack - Your AI Productivity Partner") 