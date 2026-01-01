# NeuroTrack – Product Vision & Purpose

## 1) Value Proposition
- Be your personal productivity and wellbeing coach: a simple daily check-in that turns into tailored insights, recommendations, and gentle nudges to work better, not just more.
- Blend data you already generate (tasks, time, energy) into clear trends, risk alerts (burnout), and practical next steps.

## 2) Who It Serves
- Students and self-learners: consistent study habits, exam prep, concept retention.
- Developers/creators/freelancers: deep-work focus, timeboxing, context switching control.
- Early-career professionals: visibility into progress, momentum, and prioritization clarity.
- Neurodivergent/energy-variant users: plan around peak energy; avoid overload; track mood.

## 3) Core Jobs-To-Be-Done
- Plan today in minutes; execute with focus; end the day confident you moved forward.
- See where time really goes; reduce busywork and switch-costs.
- Catch burnout signals early; rebalance before it hurts output or wellbeing.
- Get smart suggestions: “what’s the best next task given my energy, goals, and time?”

## 4) Differentiators
- Insights-first: dashboards + burnout risk + reasoned recommendations (not just lists).
- Energy-aware planning: mood/energy signals drive task suggestions and schedules.
- Gentle habit loop: daily check-in, weekly review, streaks, and tiny experiments.
- Privacy-first local data (CSV) with export; database optional as you scale.

## 5) Initial Purposeful Additions (MVP+2 weeks)
1. Goal-focused onboarding (first run): choose focus area, set weekly targets, pick preferred schedule, confirm privacy/export options.
2. Weekly Review flow: auto-summarize wins, time shifts, completion trends; one-click commitments for next week.
3. Feedback on recommendations: thumbs up/down + “why”; use signals to adapt.
4. Reminders and streaks: light reminders inside the app; visible streak badges and progress bars.

## 6) Mid-Term Roadmap (4–8 weeks)
- Read-only calendar import (time blocks → context awareness).
- Templates for common workflows (study sprint, coding session, exam week).
- Energy-aware scheduling: suggest slots for harder tasks during peak hours.
- Email weekly digest (optional) with trends and recommended plan.

## 7) Long-Term Bets (2–3 months)
- Embedding-based recommendations with vector search; per-user personalization.
- Predictive models: time-to-complete, risk of deferral, early burnout alerts with explanations.
- Team/coach mode: share selected metrics, goals, and summaries with mentors or teams.

## 8) Success Metrics
- North Star: Weekly Active Reflectors (WAR) – users completing ≥1 check-in/week.
- Habit depth: average weekly streak length; % with defined weekly goals.
- Insight efficacy: recommendation acceptance rate; time in focus; reduction in context switches.
- Wellbeing: % of high-risk burnout weeks resolved within 7 days.

## 9) Trust & Safety
- Clear privacy copy: data stays local by default; opt-in exports.
- Explainability: “why this recommendation”; simple, editable rules.
- Escape hatches: pause tracking, private sessions, delete data.

## 10) Implementation Notes (maps to code)
- Onboarding/Goals: add first-run wizard and preferences to `app.py`; persist to CSV.
- Weekly Review: new view with summaries using `utils.py` + `data_handler.py`.
- Feedback Loop: extend `recommendations.py` to record feedback + reasons; store events.
- Reminders/Streaks: session-state counters and badges in `app.py`.
- Metrics/Logging: structured event logging (CSV/JSON) + simple telemetry toggles.
