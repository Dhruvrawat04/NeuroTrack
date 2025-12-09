# ==================== IMPORTS ====================
# Data processing
from datetime import date, timedelta
import pandas as pd

def get_peak_hours(data: pd.DataFrame):
    """Detect peak productivity hours based on completion rate and task complexity."""
    try:
        if data.empty:
            return []
        
        data = data.copy()
        data["start_time"] = pd.to_datetime(data["start_time"], errors="coerce")
        data = data.dropna(subset=["start_time"])

        if data.empty:
            return []
        
        data["hour"] = data["start_time"].dt.hour
        
        if data.empty:
            return []
        
        hour_stats = []
        for hour in range(24):
            hour_data = data[data["hour"] == hour]
            if len(hour_data) < 2:
                continue
                
            completion_rate = hour_data["completed"].mean() * 100
            avg_time = hour_data["time_taken"].mean()
            complexity_bonus = min(avg_time / 60, 2) * 10
            
            productivity_score = completion_rate + complexity_bonus
            next_hour = (hour + 1) % 24
            hour_range = f"{hour:02d}:00-{next_hour:02d}:00"
            
            hour_stats.append((hour_range, productivity_score))
        
        hour_stats.sort(key=lambda x: x[1], reverse=True)
        return hour_stats
    except Exception as e:
        print(f"Error in get_peak_hours: {e}")
        return []

def get_weekly_summary(data: pd.DataFrame) -> str:
    """Generates a summary of weekly productivity."""
    if data.empty:
        return "No data available for weekly summary."

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
    data.dropna(subset=["date"], inplace=True)

    if data.empty:
        return "No valid date data for weekly summary."

    today = date.today()
    seven_days_ago = today - timedelta(days=6)
    recent_data = data[(data["date"] >= seven_days_ago) & (data["date"] <= today)]

    if recent_data.empty:
        return "No tasks logged in the last 7 days."

    daily_summary = recent_data.groupby("date").agg(
        total_time=("time_taken", "sum"),
        completed_tasks=("completed", lambda x: x.sum()),
        total_tasks=("task", "count")
    ).reset_index()

    summary_lines = ["### Weekly Productivity Summary (Last 7 Days):"]
    for _, row in daily_summary.sort_values("date").iterrows():
        day_name = row["date"].strftime("%A, %b %d")
        total_hours = row["total_time"] / 60
        completion_rate = (row["completed_tasks"] / row["total_tasks"]) * 100 if row["total_tasks"] > 0 else 0
        
        summary_lines.append(
            f"- **{day_name}**: {total_hours:.1f} hours logged, "
            f"{int(row['completed_tasks'])}/{int(row['total_tasks'])} tasks completed ({completion_rate:.1f}%)"
        )
    
    total_weekly_time = daily_summary["total_time"].sum() / 60
    total_weekly_completed = daily_summary["completed_tasks"].sum()
    total_weekly_tasks = daily_summary["total_tasks"].sum()
    overall_completion_rate = (total_weekly_completed / total_weekly_tasks) * 100 if total_weekly_tasks > 0 else 0

    summary_lines.append(f"\n**Overall this week:**")
    summary_lines.append(f"- Total time: **{total_weekly_time:.1f} hours**")
    summary_lines.append(f"- Overall completion: **{int(total_weekly_completed)}/{int(total_weekly_tasks)} tasks** ({overall_completion_rate:.1f}%)")

    return "\n".join(summary_lines)

def assess_burnout_risk(data: pd.DataFrame) -> str:
    """Assesses burnout risk based on work patterns."""
    if data.empty:
        return "Low"

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
    data.dropna(subset=["date"], inplace=True)

    if data.empty:
        return "Low"

    risk_score = 0
    
    recent_data = data[data["date"] >= (date.today() - timedelta(days=13))]
    if not recent_data.empty:
        daily_time = recent_data.groupby("date")["time_taken"].sum()
        long_workdays = daily_time[daily_time > 480]
        if len(long_workdays) >= 3:
            risk_score += 2
        if len(long_workdays) >= 5:
            risk_score += 3

        avg_daily_hours = daily_time.mean() / 60
        if avg_daily_hours > 9:
            risk_score += 1

    all_dates = pd.date_range(end=date.today(), periods=14, freq='D').date
    logged_dates = set(data["date"].unique())
    unlogged_days_in_period = [d for d in all_dates if d not in logged_dates and d <= date.today()]
    if len(unlogged_days_in_period) >= 3:
        risk_score += 1

    if "category" in data.columns:
        break_personal_time = data[data["category"].isin(["Break", "Personal"])]["time_taken"].sum()
        total_logged_time = data["time_taken"].sum()
        if total_logged_time > 0 and (break_personal_time / total_logged_time) < 0.05:
            risk_score += 1

    if risk_score >= 4:
        return "High"
    elif risk_score >= 2:
        return "Medium"
    else:
        return "Low"

def get_workload_recommendations(data: pd.DataFrame) -> dict:
    """Provides workload recommendations based on recent activity patterns."""
    recommendations = {}
    if data.empty:
        recommendations["suggestion"] = "Start logging tasks to get personalized workload recommendations!"
        return recommendations

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
    data.dropna(subset=["date"], inplace=True)

    if data.empty:
        recommendations["suggestion"] = "No valid date data for workload recommendations."
        return recommendations

    recent_data = data[data["date"] >= (date.today() - timedelta(days=6))]
    if recent_data.empty:
        recommendations["suggestion"] = "Not enough recent data for workload recommendations. Log more tasks!"
        return recommendations

    daily_time = recent_data.groupby("date")["time_taken"].sum()
    avg_daily = daily_time.mean()
    
    if avg_daily > 480:
        recommendations["warning"] = f"High daily average ({avg_daily/60:.1f}h). Consider breaking tasks into smaller chunks or taking more breaks."
    elif avg_daily < 120:
        recommendations["suggestion"] = f"Light schedule ({avg_daily/60:.1f}h daily). Room to tackle more goals or learn something new."
    
    if "category" in recent_data.columns:
        cat_time = recent_data.groupby("category")["time_taken"].sum()
        total_time_in_cats = cat_time.sum()
        
        if total_time_in_cats > 0:
            dominant_cat_proportion = cat_time.max() / total_time_in_cats
            if dominant_cat_proportion > 0.7:
                dominant_name = cat_time.idxmax()
                recommendations["suggestion"] = f"'{dominant_name}' takes {dominant_cat_proportion*100:.0f}% of your time. Consider diversifying activities to avoid monotony."
    
    completion_rate = recent_data["completed"].mean()
    if completion_rate < 0.6:
        recommendations["warning"] = f"Low completion rate ({completion_rate*100:.0f}%). Try setting smaller, more achievable tasks to build momentum."
    elif completion_rate > 0.9:
        recommendations["positive"] = f"Excellent completion rate ({completion_rate*100:.0f}%)! You're managing tasks very effectively."

    avg_task_duration = recent_data["time_taken"].mean()
    if avg_task_duration > 90:
        recommendations["suggestion"] = "Your average task duration is quite long. Try using the Pomodoro technique or breaking tasks into 60-90 minute blocks."
    elif avg_task_duration < 15:
        recommendations["suggestion"] = "Many short tasks detected. Consider batching similar quick tasks together to reduce context switching."

    return recommendations