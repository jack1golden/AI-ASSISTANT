
# OBW AI Safety Assistant — Floorplan (Patched)

This build keeps the pharma-style floorplan background and adds:
- Robust click resolver (safe for missing/None x/y)
- Room titles on the plan
- Accurate statuses from data (no more all-FAULT)
- Live charts with thresholds & rolling mean
- Simulation Center (Dev) to inject scenarios

## Run
```bash
pip install -r requirements.txt
streamlit run tppr_ai_demo_app.py
```
