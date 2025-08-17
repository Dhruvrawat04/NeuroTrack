# insight_charts.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Custom blue-purple color scales
HEATMAP_COLORS = [
    [0.0, '#0B0B3B'],     # Deep navy
    [0.1, '#1A1A7A'],     # Royal blue
    [0.3, '#4B0082'],     # Indigo
    [0.5, '#8B0000'],     # Dark red
    [0.7, '#ef4444'],     # Red hot
    [0.9, '#FF4500'],     # Orange-red
    [1.0, '#FFD700']      # Gold/yellow (hottest)
]

PRIORITY_COLORS = {
    'High': '#ef4444',    # Red (hottest)
    'Medium': '#f59e0b',  # Amber (warm)
    'Low': '#1d4ed8'      # Blue (cool)
}

def process_heatmap_data(_df):
    """Process data for heatmap without caching for real-time updates"""
    heatmap_data = _df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(heatmap_data['start_time']):
        heatmap_data['start_time'] = pd.to_datetime(heatmap_data['start_time'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(heatmap_data['date']):
        heatmap_data['date'] = pd.to_datetime(heatmap_data['date'], errors='coerce').dt.date
    
    # Extract hour and day of week
    heatmap_data['hour'] = heatmap_data['start_time'].dt.hour
    heatmap_data['day_of_week'] = heatmap_data['date'].apply(lambda x: x.strftime('%A'))
    
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                "Friday", "Saturday", "Sunday"]
    heatmap_data['day_of_week'] = pd.Categorical(
        heatmap_data['day_of_week'],
        categories=day_order,
        ordered=True
    )
    
    # Group and fill missing hours with 0
    return (heatmap_data.groupby(['day_of_week', 'hour'])
         ['time_taken'].sum()
         .unstack()
         .reindex(columns=range(0,24), fill_value=0))  # Fixed parentheses placement

def show_insight_charts(data: pd.DataFrame):
    st.markdown("## 🔍 Deep Insights", help="Analyze your productivity patterns and trends")
    
    if data.empty:
        st.info("📭 No data available for insight charts")
        return
    
    # Data Preparation
    data = data.copy()
    try:
        # Convert numeric fields
        data["time_taken"] = pd.to_numeric(data["time_taken"], errors="coerce").fillna(0)
        data = data[data["time_taken"] > 0]
        
        # Convert date/time fields
        if pd.api.types.is_object_dtype(data["date"]):
            data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
        
        if "start_time" in data.columns and pd.api.types.is_object_dtype(data["start_time"]):
            data["start_time"] = pd.to_datetime(data["start_time"], errors="coerce")
    except Exception as e:
        st.error(f"🔧 Data preparation error: {str(e)}")
        return

    # 1. Priority Sunburst Chart
    if all(col in data.columns for col in ['priority', 'category']):
        with st.expander("⏳ Time Allocation by Priority", expanded=True):
            priority_data = data[data['priority'].isin(['Low', 'Medium', 'High'])].copy()
            
            if not priority_data.empty:
                fig = px.sunburst(
                    priority_data,
                    path=['priority', 'category'],
                    values='time_taken',
                    color='priority',
                    color_discrete_map=PRIORITY_COLORS,
                    hover_data={'time_taken': ':.0f min'},
                    height=600
                )
                fig.update_traces(
                    textinfo="label+percent parent",
                    textfont_size=14,
                    marker=dict(line=dict(color='white', width=1))
                )
                fig.update_layout(margin=dict(t=30, b=30))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid priority data available")

    # 2. Difficulty Analysis
    if 'difficulty' in data.columns:
        with st.expander("📊 Task Difficulty Analysis", expanded=True):
            difficulty_data = data[(data['difficulty'] >= 1) & (data['difficulty'] <= 5)].copy()
            
            if not difficulty_data.empty:
                difficulty_stats = difficulty_data.groupby("difficulty").agg(
                    avg_time=("time_taken", "mean"),
                    completion_rate=("completed", "mean"),
                    task_count=("task", "count")
                ).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=difficulty_stats["difficulty"],
                    y=difficulty_stats["avg_time"],
                    name="Avg Time",
                    marker_color='#4B0082',  # Indigo
                    hovertemplate="<b>Difficulty %{x}</b><br>Avg: %{y:.1f} min<extra></extra>"
                ))
                
                fig.add_trace(go.Scatter(
                    x=difficulty_stats["difficulty"],
                    y=difficulty_stats["completion_rate"]*100,
                    name="Completion %",
                    line=dict(color='#9370DB', width=3),  # Medium Purple
                    yaxis="y2",
                    mode='lines+markers',
                    marker=dict(size=10),
                    hovertemplate="<b>%{y:.1f}% completed</b><extra></extra>"
                ))
                
                fig.update_layout(
                    title="Performance by Task Difficulty",
                    xaxis_title="Difficulty Level (1-5)",
                    yaxis_title="Average Time (minutes)",
                    yaxis2=dict(
                        title="Completion Rate (%)",
                        overlaying="y",
                        side="right",
                        range=[0, 100]
                    ),
                    hovermode="x unified",
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

    # 3. Enhanced Heatmap Visualization
    with st.expander("🔥 Productivity Heatmap (24 Hours)", expanded=True):
        if "start_time" in data.columns:
            try:
                heatmap_stats = process_heatmap_data(data)
                
                fig = px.imshow(
                    heatmap_stats,
                    labels=dict(x="Hour", y="Day", color="Minutes"),
                    color_continuous_scale=HEATMAP_COLORS,
                    aspect="auto",
                    zmin=0,
                    zmax=max(heatmap_stats.max().max(), 60)  # Minimum scale of 60 mins
                )
                
                fig.update_traces(
                    hovertemplate="<b>%{y}</b><br>%{x}:00-%{x}:59<br><b>%{z:.0f} mins</b><extra></extra>",
                    hoverongaps=False
                )
                
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(0, 24, 2)),
                        ticktext=[f"{h}:00" for h in range(0, 24, 2)],
                        title="Hour of Day"
                    ),
                    yaxis=dict(title=""),
                    height=650,
                    margin=dict(t=50, b=20),
                    coloraxis_colorbar=dict(
                        title="Minutes",
                        thickness=20,
                        tickvals=np.linspace(0, heatmap_stats.max().max(), 5),
                        tickformat=".0f"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Peak hours analysis
                col1, col2 = st.columns(2)
                with col1:
                    peak_hour = heatmap_stats.sum().idxmax()
                    st.metric("⏱️ Peak Hour", 
                             f"{peak_hour}:00-{peak_hour+1}:00",
                             help="Hour with most total activity")
                
                with col2:
                    best_day = heatmap_stats.sum(axis=1).idxmax()
                    st.metric("📅 Best Day", 
                             best_day,
                             help="Day with most total activity")
                
            except Exception as e:
                st.error(f"Failed to generate heatmap: {str(e)}")  # Fixed parenthesis