# Demo Video Script — Bikeshare Station Risk Monitor

**Format**: 3-minute Loom screen recording with voiceover
**Target audience**: Hiring managers, technical reviewers, portfolio visitors
**Tone**: Concise, confident, technical but accessible

---

## Pre-Recording Setup Checklist

- [ ] Open the dashboard in Chrome at full width (1920x1080 recommended)
- [ ] Confirm the environment badge shows **PRODUCTION** (not STAGING)
- [ ] Select **Bike stockout** as the prediction target in the sidebar
- [ ] Set Top-N stations slider to **20**
- [ ] Set History snapshots slider to **24**
- [ ] Navigate to the **Live Ops** tab so the map is visible on screen
- [ ] Close all other browser tabs and mute desktop notifications
- [ ] Test microphone levels — speak at normal conversational volume
- [ ] Have the map zoomed to show all of Paris with station markers visible

---

## Script

### Scene 1 — Opening [0:00 – 0:20]

**Screen**: Live Ops tab with the full Paris station map visible. Status cards at the top.

> *"This is a real-time MLOps platform predicting bikeshare stockouts 30 minutes ahead for 1,400 Paris stations. End-to-end: live GBFS and weather ingestion, dbt feature engineering, XGBoost on SageMaker — all orchestrated by Airflow."*

**Action**: Point to the PRODUCTION environment badge, then gesture across the status cards (Pipeline state → Prediction target → Active endpoint → Active model → Latest prediction).

---

### Scene 2 — Live Ops Map [0:20 – 0:55]

**Action**: Switch prediction target to **Dock stockout**, then back to **Bike stockout**.

> *"The platform supports dual targets — bikes and docks each have independent models and endpoints. I can switch between them in the sidebar."*

**Screen**: The Folium map with colored station markers.

**Action**: Click on a **red** marker, then a **teal** marker.

> *"Each marker shows stockout risk — red is critical, orange is alert, teal is normal. Clicking a station shows its inventory, capacity, and predicted 30-minute probability."*

**Action**: Point to the alert banner above the tab bar, then scroll down to the top-risk table.

> *"The alert banner summarises risk across all stations. Below, the top-risk table tells an operations team exactly where to send rebalancing trucks."*

---

### Scene 3 — Station History [0:55 – 1:20]

**Action**: Click on a station marker with visible history variation, then click the **Station History** tab.

> *"Selecting a station and switching to Station History shows its 30-minute stockout probability across the last 24 snapshots. The dashed orange line is the model's alert threshold — scores above it trigger an alert."*

---

### Scene 4 — Prediction Quality [1:20 – 1:50]

**Action**: Click the **Prediction Quality** tab. Point to the PR-AUC card, then the Prediction Heartbeat card.

> *"Prediction Quality shows actual model performance against ground truth. A quality backfill runs every 15 minutes once the 30-minute label window matures. PR-AUC is the primary metric — currently [read value]. Prediction Heartbeat confirms the monitoring loop ran [read value] times in the last 24 hours."*

---

### Scene 5 — System Health [1:50 – 2:20]

**Action**: Click the **System Health** tab. Point to the ModelLatency card, then to the PSI cards.

**Screen**: Metric grid — top row: ModelLatency, Invocation5XXErrors, Invocation4XXErrors; middle row: Invocations, Prediction Heartbeat, PSI Overall; bottom row: PSI Core, PSI Weather.

> *"System Health pulls CloudWatch directly. Latency p95 is [read value] ms — within the 200 ms SLA, zero errors. PSI Core at [read value] is below the 0.20 drift threshold; weather PSI is advisory. CloudWatch alarms cover all of this and page via SNS."*

---

### Scene 6 — Data Status [2:20 – 2:45]

**Action**: Click the **Data Status** tab. Point to each row in the table.

**Screen**: Compact freshness table — four rows in order: Prediction artifact, Quality artifact, Source freshness, Feature freshness.

> *"Data Status closes the loop. Each row tracks a pipeline stage — predictions, quality, source, and feature freshness — so any break in the chain surfaces here immediately."*

---

### Scene 7 — Closing [2:45 – 3:00]

**Action**: Click back to the **Live Ops** tab.

> *"Live ingestion, automated features, ML serving, continuous monitoring — all observable from a single dashboard. Full architecture and runbook are in the repository. Thanks for watching."*

**Action**: Hold on the Live Ops map for 2–3 seconds as a clean closing frame.

---

## Post-Recording Checklist

- [ ] Trim dead air at the start and end of the recording
- [ ] Verify audio is clear and consistent throughout
- [ ] Check that all tooltips and tab transitions rendered cleanly
- [ ] Add the Loom link to `DEMO.md` (replace the `(#)` placeholder in the Demo Video section)
- [ ] Optionally add the Loom link to the GitHub repo "About" section
