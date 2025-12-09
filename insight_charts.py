# ==================== IMPORTS ====================
# UI framework
import streamlit as st

# Data processing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# Local modules
from data_constants import HEATMAP_COLORS, PRIORITY_COLORS
from data_preprocessing import prepare_datetime_columns, extract_hour_from_datetime

def process_heatmap_data(_df):
    """Process data for heatmap without caching for real-time updates."""
    try:
        if _df is None or _df.empty:
            return pd.DataFrame()
        
        heatmap_data = _df.copy()
    
        # Convert to datetime using shared utility
        try:
            heatmap_data = prepare_datetime_columns(heatmap_data, copy=False)
        except Exception as e:
            print(f"Error preparing datetime columns: {e}")
            return pd.DataFrame()
        
        # Extract hour and day of week
        if 'start_time' not in heatmap_data.columns or 'date' not in heatmap_data.columns:
            return pd.DataFrame()
        
        try:
            heatmap_data['hour'] = heatmap_data['start_time'].dt.hour
            heatmap_data['day_of_week'] = heatmap_data['date'].apply(lambda x: x.strftime('%A'))
        except Exception as e:
            print(f"Error extracting hour/day_of_week: {e}")
            return pd.DataFrame()
    
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                    "Friday", "Saturday", "Sunday"]
        heatmap_data['day_of_week'] = pd.Categorical(
            heatmap_data['day_of_week'],
            categories=day_order,
            ordered=True
        )
        
        # Group and fill missing hours with 0
        try:
            return (heatmap_data.groupby(['day_of_week', 'hour'])
                 ['time_taken'].sum()
                 .unstack()
                 .reindex(columns=range(0,24), fill_value=0))
        except Exception as e:
            print(f"Error grouping heatmap data: {e}")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Unexpected error in process_heatmap_data: {e}")
        return pd.DataFrame()

def show_insight_charts(data: pd.DataFrame):
    """Display deep insight charts with comprehensive error handling."""
    try:
        st.markdown("## 🔍 Deep Insights", help="Analyze your productivity patterns and trends")
        
        if data is None or data.empty:
            st.info("📭 No data available for insight charts")
            return
    
        # Data Preparation
        data = data.copy()
        try:
            # Convert numeric fields
            data["time_taken"] = pd.to_numeric(data["time_taken"], errors="coerce").fillna(0)
            data = data[data["time_taken"] > 0]
            
            # Convert date/time fields using shared utility
            data = prepare_datetime_columns(data, copy=False)
        except Exception as e:
            st.error(f"🔧 Data preparation error: {str(e)}")
            return
        
        if data.empty:
            st.warning("No valid data available after processing")
            return

        # Highlights strip for quick takeaways
        try:
            with st.container():
                st.markdown("### ✨ Highlights")
                col1, col2, col3 = st.columns(3)

                # Peak hour from start_time
                peak_hour_label = "Not enough data"
                if "start_time" in data.columns and not data["start_time"].isna().all():
                    hour_counts = data.copy()
                    hour_counts["hour"] = hour_counts["start_time"].dt.hour
                    hour_counts = hour_counts.groupby("hour")["time_taken"].sum().sort_values(ascending=False)
                    if not hour_counts.empty:
                        h = int(hour_counts.index[0])
                        peak_hour_label = f"{h:02d}:00 - {h+1:02d}:00"
                col1.metric("⏱️ Peak Hour", peak_hour_label)

                # Best day by total time
                best_day_label = "Not enough data"
                if "date" in data.columns and not data["date"].isna().all():
                    day_totals = data.groupby("date")["time_taken"].sum().sort_values(ascending=False)
                    if not day_totals.empty:
                        best_day_label = day_totals.index[0].strftime("%a, %b %d")
                col2.metric("📅 Best Day", best_day_label)

                # Mood with highest completion rate (if available)
                top_mood_label = "Not enough data"
                if "mood" in data.columns and "completed" in data.columns and not data["mood"].isna().all():
                    mood_stats = data.groupby("mood").agg(count=("task", "count"), comp=("completed", "mean"))
                    mood_stats = mood_stats[mood_stats["count"] >= 3]  # require some support
                    if not mood_stats.empty:
                        best_mood = mood_stats.sort_values("comp", ascending=False).index[0]
                        top_mood_label = f"{best_mood} ({mood_stats.loc[best_mood, 'comp']*100:.0f}% completion)"
                col3.metric("😊 Best Mood", top_mood_label)
        except Exception as e:
            st.warning(f"Highlights unavailable: {str(e)}")

        # 1. Priority Sunburst Chart
        try:
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
        except Exception as e:
            st.error(f"Error creating priority chart: {str(e)}")

        # 2. Difficulty Analysis
        try:
            if 'difficulty' in data.columns:
                with st.expander("📊 Task Difficulty Analysis", expanded=True):
                    difficulty_data = data[(data['difficulty'] >= 1) & (data['difficulty'] <= 5)].copy()
                    
                    if not difficulty_data.empty:
                        difficulty_stats = difficulty_data.groupby("difficulty").agg(
                            avg_time=("time_taken", "mean"),
                            completion_rate=("completed", "mean" if "completed" in difficulty_data.columns else lambda x: 0),
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
        except Exception as e:
            st.error(f"Error creating difficulty analysis chart: {str(e)}")

        # 3. Enhanced Heatmap Visualization
        try:
            with st.expander("🔥 Productivity Heatmap (24 Hours)", expanded=True):
                if "start_time" in data.columns:
                    try:
                        heatmap_stats = process_heatmap_data(data)
                        
                        if heatmap_stats.empty:
                            st.warning("No data available for heatmap")
                        else:
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
                            try:
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
                                st.warning(f"Could not calculate peak hours: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"Failed to generate heatmap: {str(e)}")
                else:
                    st.warning("Start time data not available for heatmap")
        except Exception as e:
            st.error(f"Error in heatmap section: {str(e)}")
    
    except Exception as e:
        st.error(f"Unexpected error in insight charts: {str(e)}")
        import traceback
        st.debug(traceback.format_exc())