import pandas as pd

class MLInsightsGenerator:
    def __init__(self):
        pass
        
    def generate_insights(self, data: pd.DataFrame):
        """Generate ML-powered insights about productivity patterns."""
        insights = {}
        
        try:
            if data.empty or len(data) < 20:
                insights["status"] = "Need more data (at least 20 tasks) for ML insights"
                return insights
            
            # Completion rate by difficulty
            if 'difficulty' in data.columns:
                diff_completion = data.groupby('difficulty')['completed'].mean()
                if len(diff_completion) > 1:
                    best_diff = diff_completion.idxmax()
                    worst_diff = diff_completion.idxmin()
                    insights["difficulty"] = (
                        f"Peak completion at difficulty {best_diff} ({diff_completion[best_diff]*100:.0f}%) | "
                        f"Struggle with difficulty {worst_diff} ({diff_completion[worst_diff]*100:.0f}%)"
                    )
            
            # Energy vs Performance
            if 'energy_level' in data.columns:
                energy_completion = data.groupby('energy_level')['completed'].mean()
                if len(energy_completion) > 1:
                    best_energy = energy_completion.idxmax()
                    insights["energy"] = (
                        f"Optimal energy level: {best_energy} ({energy_completion[best_energy]*100:.0f}% completion) | "
                        f"Performance drops {((energy_completion.max() - energy_completion.min())/energy_completion.max())*100:.0f}% across energy levels"
                    )
            
            # Time patterns
            if 'time_taken' in data.columns:
                avg_time = data['time_taken'].mean()
                median_time = data['time_taken'].median()
                insights["time_pattern"] = (
                    f"Average task duration: {avg_time:.0f} mins | "
                    f"Median: {median_time:.0f} mins (suggests {'long' if avg_time > 60 else 'short'} tasks dominate)"
                )
            
            # Category performance
            if 'category' in data.columns:
                cat_stats = data.groupby('category').agg(
                    count=('task', 'count'),
                    completion_rate=('completed', 'mean'),
                    avg_time=('time_taken', 'mean')
                )
                if len(cat_stats) > 1:
                    best_cat = cat_stats['completion_rate'].idxmax()
                    worst_cat = cat_stats['completion_rate'].idxmin()
                    insights["category"] = (
                        f"Best performing: {best_cat} ({cat_stats.loc[best_cat, 'completion_rate']*100:.0f}%) | "
                        f"Needs improvement: {worst_cat} ({cat_stats.loc[worst_cat, 'completion_rate']*100:.0f}%)"
                    )
            
            # Time of day patterns
            if 'start_time' in data.columns:
                try:
                    data['hour'] = pd.to_datetime(data['start_time']).dt.hour
                    hour_stats = data.groupby('hour').agg(
                        count=('task', 'count'),
                        completion_rate=('completed', 'mean')
                    )
                    if len(hour_stats) > 1:
                        best_hour = hour_stats['completion_rate'].idxmax()
                        insights["time_of_day"] = (
                            f"Most productive hour: {best_hour}:00 ({hour_stats.loc[best_hour, 'completion_rate']*100:.0f}%) | "
                            f"Least productive: {hour_stats['completion_rate'].idxmin()}:00"
                        )
                except:
                    pass
            
            insights["status"] = "ML insights generated successfully"
            return insights
            
        except Exception as e:
            insights["status"] = f"Error generating insights: {str(e)}"
            return insights