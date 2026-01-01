# Toggle Feature Documentation 🎛️

## Overview
A comprehensive toggling feature has been added to NeuroTrack, allowing users to show/hide different sections of the dashboard to customize their viewing experience.

## Features Added

### 1. **Display Settings Section** (Sidebar)
Located in the sidebar under "⚙️ Display Settings", users can now toggle the following sections:

#### Available Toggles:
- **📈 Statistics** - Task statistics dashboard with metrics and charts
- **📊 Visualizations** - Data visualization tabs (Basic Overview, Productivity Metrics, Deep Insights)
- **🎯 Performance** - Performance dashboard with trends and priority distribution
- **🎯 Goals** - Productivity goals and progress tracking
- **📋 Task Management** - Recent tasks list with filtering options
- **🧠 ML Insights** - AI-powered insights including peak hours and ML analysis
- **📁 Import Data** - CSV data import functionality

### 2. **Quick Action Buttons**
Two convenient buttons for bulk toggling:
- **✅ Show All** - Enables all sections at once
- **❌ Hide All** - Disables all sections at once

## How It Works

### Session State Management
The feature uses Streamlit's session state to persist toggle settings across reruns:

```python
st.session_state.setdefault('show_statistics', True)
st.session_state.setdefault('show_visualizations', True)
st.session_state.setdefault('show_performance', True)
st.session_state.setdefault('show_goals', True)
st.session_state.setdefault('show_import', True)
st.session_state.setdefault('show_task_management', True)
st.session_state.setdefault('show_insights', True)
```

### Conditional Rendering
Each section is wrapped in a conditional statement:

```python
if st.session_state.show_statistics:
    # Statistics content
    ...

if st.session_state.show_visualizations:
    # Visualization content
    ...
```

## User Benefits

1. **Customized View** - Focus on specific aspects of productivity tracking
2. **Reduced Clutter** - Hide sections not currently needed
3. **Performance** - Less rendering for hidden sections may improve load times
4. **Better UX** - Cleaner interface tailored to individual workflow needs

## Usage Examples

### Example 1: Focus Mode
Hide everything except basic statistics:
1. Click "❌ Hide All"
2. Enable only "📈 Statistics"
3. View streamlined dashboard with just core metrics

### Example 2: Analysis Mode
Show only visualizations and insights:
1. Disable Statistics, Goals, Import
2. Enable Visualizations and ML Insights
3. Deep dive into data patterns and trends

### Example 3: Task Management Focus
Show only task management and goals:
1. Enable Task Management and Goals
2. Hide other sections
3. Focus on completing and tracking tasks

## Technical Implementation

### Files Modified
- `app.py` - Main application file with all toggle logic

### Key Changes
1. Added session state initialization for toggle states
2. Created sidebar toggle controls section
3. Implemented quick action buttons
4. Wrapped major sections with conditional rendering
5. Maintained proper section hierarchy and dividers

### Code Structure
```python
# Sidebar toggles
with st.sidebar:
    st.markdown("### ⚙️ Display Settings")
    st.session_state.show_statistics = st.checkbox("📈 Statistics", ...)
    st.session_state.show_visualizations = st.checkbox("📊 Visualizations", ...)
    # ... more toggles
    
    # Quick actions
    if st.button("✅ Show All"):
        # Enable all
    if st.button("❌ Hide All"):
        # Disable all

# Main content with conditionals
if st.session_state.show_statistics:
    # Statistics section
    
if st.session_state.show_visualizations:
    # Visualization section
```

## Future Enhancements

Potential improvements for the toggle feature:
1. **Save Preferences** - Persist toggle states to file/database
2. **Custom Layouts** - Pre-defined view configurations
3. **Section Reordering** - Drag and drop section arrangement
4. **Minimize/Expand** - Collapsible sections instead of complete hiding
5. **Keyboard Shortcuts** - Quick toggle via hotkeys

## Notes

- All toggles default to `True` (enabled) on first load
- Toggle states persist during the session
- Using "Show All" or "Hide All" triggers a page rerun to apply changes
- Individual toggles update immediately without rerun
- The feature is fully compatible with existing data and functionality

## Support

For issues or questions about the toggle feature:
1. Check that all sections render correctly when enabled
2. Verify session state is properly initialized
3. Ensure no conflicts with other sidebar elements
4. Test with both empty and populated data sets

---

**Version:** 1.0  
**Date:** January 1, 2026  
**Author:** NeuroTrack Development Team
