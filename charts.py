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
from data_preprocessing import prepare_datetime_columns, filter_by_days_back

def show_basic_charts(data: pd.DataFrame):
    """Display basic productivity overview charts with error handling."""
    try:
        st.markdown("## 📊 Basic Overview")
        
        if data is None or data.empty:
            st.info("No data available for basic charts")
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
    
        # Quick metrics
        col1, col2, col3, col4 = st.columns(4)
        try:
            with col1:
                st.metric("📋 Total Tasks", len(data))
            with col2:
                st.metric("⏱️ Total Hours", f"{data['time_taken'].sum()/60:.1f}")
            with col3:
                completed = data[data["completed"] == True] if "completed" in data.columns else pd.DataFrame()
                st.metric("✅ Completed", f"{len(completed)}/{len(data)}")
            with col4:
                avg_time = data['time_taken'].mean() if len(data) > 0 else 0
                st.metric("⭐ Avg Time", f"{avg_time:.0f} min")
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
    
        # Category distribution
        try:
            st.markdown("### ⏱️ Time Allocation by Category")
            if "category" in data.columns:
                cat_data = data.groupby("category").agg({
                    "time_taken": "sum",
                    "task": "count"
                }).reset_index()
                
                if not cat_data.empty:
                    # Define category colors for consistency
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=cat_data["category"],
                        values=cat_data["time_taken"],
                        hole=0.4,
                        marker=dict(colors=colors[:len(cat_data)], line=dict(color='white', width=2)),
                        textposition='inside',
                        textinfo='label+percent',
                        hovertemplate='<b>%{label}</b><br>Time: %{value:.0f}m<br>Percentage: %{percent}<extra></extra>'
                    )])
                    
                    fig.update_layout(
                        title="Time Spent by Category",
                        height=350,
                        showlegend=True,
                        font=dict(size=12),
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No category data available")
            else:
                st.warning("Category column not found in data")
        except Exception as e:
            st.error(f"Error creating category distribution chart: {str(e)}")
    
        # Completion status
        try:
            st.markdown("### Task Completion Status")
            if "completed" in data.columns:
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
            else:
                st.warning("Completion column not found in data")
        except Exception as e:
            st.error(f"Error creating completion status chart: {str(e)}")
    
        # Recent activity
        try:
            st.markdown("### Recent Activity (Last 14 Days)")
            recent_data = filter_by_days_back(data, days=14)
            
            if recent_data is not None and not recent_data.empty:
                daily_stats = recent_data.groupby("date")["time_taken"].sum().reset_index()
                
                if not daily_stats.empty:
                    fig = px.bar(
                        daily_stats,
                        x="date",
                        y="time_taken",
                        title="Daily Time Investment",
                        labels={"time_taken": "Minutes", "date": "Date"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No activity data available for the last 14 days")
            else:
                st.info("No recent activity data available")
        except Exception as e:
            st.error(f"Error creating recent activity chart: {str(e)}")
    
    except Exception as e:
        st.error(f"Unexpected error in basic charts: {str(e)}")
        import traceback
        st.debug(traceback.format_exc())