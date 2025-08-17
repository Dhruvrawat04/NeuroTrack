# charts.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show_basic_charts(data: pd.DataFrame):
    st.markdown("## 📊 Basic Overview")
    
    if data.empty:
        st.info("No data available for basic charts")
        return
    
    # Clean data
    data = data.copy()
    data["time_taken"] = pd.to_numeric(data["time_taken"], errors="coerce").fillna(0)
    data = data[data["time_taken"] > 0]
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total Tasks", len(data))
    with col2:
        st.metric("⏱️ Total Hours", f"{data['time_taken'].sum()/60:.1f}")
    with col3:
        completed = data[data["completed"] == True]
        st.metric("✅ Completed", f"{len(completed)}/{len(data)}")
    with col4:
        st.metric("⭐ Avg Time", f"{data['time_taken'].mean():.0f} min")
    
    # Category distribution
    st.markdown("### Time Allocation by Category")
    cat_data = data.groupby("category").agg({
        "time_taken": "sum",
        "task": "count"
    }).reset_index()
    
    fig = px.pie(
        cat_data,
        names="category",
        values="time_taken",
        title="Time Spent by Category",
        hole=0.3,
        hover_data={"category": True, "time_taken": True, "task": True} 
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Completion status
    st.markdown("### Task Completion Status")
    completion_data = pd.DataFrame({
        "Status": ["Completed", "Pending"],
        "Count": [data["completed"].sum(), len(data) - data["completed"].sum()]
    })
    
    fig = px.bar(
        completion_data,
        x="Status",
        y="Count",
        color="Status",
        color_discrete_map={"Completed": "#2ecc71", "Pending": "#e74c3c"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.markdown("### Recent Activity (Last 14 Days)")
    recent_data = data[data["date"] >= (datetime.now().date() - timedelta(days=14))]
    
    if not recent_data.empty:
        daily_stats = recent_data.groupby("date")["time_taken"].sum().reset_index()
        
        fig = px.bar(
            daily_stats,
            x="date",
            y="time_taken",
            title="Daily Time Investment",
            labels={"time_taken": "Minutes", "date": "Date"}
        )
        st.plotly_chart(fig, use_container_width=True)