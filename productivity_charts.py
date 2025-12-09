# ==================== IMPORTS ====================
# UI framework
import streamlit as st

# Data processing
import pandas as pd
from datetime import datetime, timedelta

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# Local modules
from data_preprocessing import prepare_datetime_columns

def show_productivity_charts(data: pd.DataFrame):
    """Display productivity metrics charts with error handling."""
    try:
        st.markdown("## 📊 Productivity Metrics")
        
        if data is None or data.empty:
            st.info("No data available for productivity charts")
            return
        
        # Clean data using shared utility
        data = data.copy()
        try:
            data = prepare_datetime_columns(data, copy=False)
        except Exception as e:
            st.error(f"Error processing datetime columns: {str(e)}")
            return
        
        try:
            data["time_taken"] = pd.to_numeric(data["time_taken"], errors="coerce").fillna(0)
        except Exception as e:
            st.error(f"Error processing time_taken column: {str(e)}")
            return
        
        data = data[data["time_taken"] > 0]
        if data.empty:
            st.warning("No valid time data available for charts")
            return
        
        # Weekly productivity trend
        try:
            st.markdown("### Weekly Productivity Trend")
            weekly_data = data.copy()
            
            if "date" not in weekly_data.columns:
                st.warning("Date column not found")
                return
            
            weekly_data["week"] = weekly_data["date"].apply(lambda x: x - timedelta(days=x.weekday()))
            
            weekly_stats = weekly_data.groupby("week").agg(
                total_time=("time_taken", "sum"),
                completed=("completed", lambda x: (x == True).sum() if "completed" in weekly_data.columns else 0),
                task_count=("task", "count")
            ).reset_index()
            
            if weekly_stats.empty:
                st.info("No weekly data available")
                return
            
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
        except Exception as e:
            st.error(f"Error creating weekly productivity chart: {str(e)}")
        
        # Focus vs Energy Analysis
        try:
            if all(col in data.columns for col in ['focus_level', 'energy_level', 'category', 'task', 'date']):
                st.markdown("### Focus vs Energy Analysis")
                focus_energy = data[
                    (data['focus_level'] != -1) & 
                    (data['energy_level'] != -1)
                ].copy()
                
                if focus_energy.empty:
                    st.info("No focus/energy data available")
                else:
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
        except Exception as e:
            st.error(f"Error creating focus vs energy chart: {str(e)}")
        
        # Mood Impact on Productivity
        try:
            if 'mood' in data.columns:
                st.markdown("### Mood Impact on Productivity")
                mood_data = data[data['mood'] != 'unknown'].copy()
                
                if mood_data.empty:
                    st.info("No mood data available")
                else:
                    mood_stats = mood_data.groupby("mood").agg(
                        avg_time=("time_taken", "mean"),
                        completion_rate=("completed", "mean" if "completed" in mood_data.columns else lambda x: 0),
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
        except Exception as e:
            st.error(f"Error creating mood impact chart: {str(e)}")
    
    except Exception as e:
        st.error(f"Unexpected error in productivity charts: {str(e)}")
        import traceback
        st.debug(traceback.format_exc())