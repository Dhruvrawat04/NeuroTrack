# productivity_charts.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show_productivity_charts(data: pd.DataFrame):
    st.markdown("## 📊 Productivity Metrics")
    
    if data.empty:
        st.info("No data available for productivity charts")
        return
    
    # Clean data
    data = data.copy()
    data["time_taken"] = pd.to_numeric(data["time_taken"], errors="coerce").fillna(0)
    data = data[data["time_taken"] > 0]
    
    # Convert date column
    if pd.api.types.is_object_dtype(data["date"]):
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
    
    # Weekly productivity trend
    st.markdown("### Weekly Productivity Trend")
    weekly_data = data.copy()
    weekly_data["week"] = weekly_data["date"].apply(lambda x: x - timedelta(days=x.weekday()))
    
    weekly_stats = weekly_data.groupby("week").agg(
        total_time=("time_taken", "sum"),
        completed=("completed", lambda x: (x == True).sum()),
        task_count=("task", "count")
    ).reset_index()
    
    weekly_stats["completion_rate"] = (weekly_stats["completed"] / weekly_stats["task_count"] * 100).round(1)
    weekly_stats["avg_time_per_task"] = (weekly_stats["total_time"] / weekly_stats["task_count"]).round(1)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekly_stats["week"],
        y=weekly_stats["total_time"]/60,
        name="Total Hours",
        marker_color='#636EFA',
        hovertext=weekly_stats.apply(
            lambda row: f"Week: {row['week'].strftime('%Y-%m-%d')}<br>"
                       f"Total Hours: {row['total_time']/60:.1f}<br>"
                       f"Tasks: {row['task_count']}<br>"
                       f"Completion: {row['completion_rate']}%",
            axis=1
        ),
        hoverinfo="text"
    ))
    fig.add_trace(go.Scatter(
        x=weekly_stats["week"],
        y=weekly_stats["completion_rate"],
        name="Completion Rate (%)",
        yaxis="y2",
        line=dict(color='#EF553B', width=2),
        hovertext=weekly_stats["completion_rate"].apply(lambda x: f"{x}%"),
        hoverinfo="text"
    ))
    
    fig.update_layout(
        title="Weekly Productivity Overview",
        xaxis_title="Week Starting",
        yaxis_title="Total Hours",
        yaxis2=dict(
            title="Completion Rate (%)",
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Focus vs Energy Analysis
    if all(col in data.columns for col in ['focus_level', 'energy_level', 'category', 'task', 'date']):
        st.markdown("### Focus vs Energy Analysis")
        focus_energy = data[
            (data['focus_level'] != -1) & 
            (data['energy_level'] != -1)
        ].copy()
        
        if not focus_energy.empty:
            fig = px.scatter(
                focus_energy,
                x="energy_level",
                y="focus_level",
                color="category",
                size="time_taken",
                hover_data={
                    "task": True,
                    "date": True,
                    "time_taken": True,
                    "energy_level": True,
                    "focus_level": True
                },
                title="Task Focus vs Energy Levels",
                labels={
                    "energy_level": "Energy Level (1-10)",
                    "focus_level": "Focus Level (1-10)"
                }
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Mood Impact on Productivity
    if 'mood' in data.columns:
        st.markdown("### Mood Impact on Productivity")
        mood_data = data[data['mood'] != 'unknown'].copy()
        
        if not mood_data.empty:
            mood_stats = mood_data.groupby("mood").agg(
                avg_time=("time_taken", "mean"),
                completion_rate=("completed", "mean"),
                task_count=("task", "count")
            ).reset_index()
            
            mood_stats["avg_time"] = mood_stats["avg_time"].round(1)
            mood_stats["completion_rate"] = (mood_stats["completion_rate"] * 100).round(1)
            
            fig = px.bar(
                mood_stats,
                x="mood",
                y="avg_time",
                color="mood",
                title="Average Task Duration by Mood",
                labels={
                    "avg_time": "Avg Time (min)", 
                    "mood": "Mood",
                    "completion_rate": "Completion Rate (%)",
                    "task_count": "Task Count"
                },
                hover_data={
                    "mood": True,
                    "avg_time": ":.1f",
                    "completion_rate": ":.1f%",
                    "task_count": True
                }
            )
            st.plotly_chart(fig, use_container_width=True)