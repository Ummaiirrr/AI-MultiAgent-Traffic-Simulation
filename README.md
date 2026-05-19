# AI Multi-Agent Traffic Simulation System
### University Lab Final Project

A complete, modular AI system that simulates city traffic using autonomous vehicle agents,
an A* informed search algorithm, and an MLP/KNN machine learning predictor trained on a
50,000-record synthetic dataset is used and modelled after the UCI Metro Interstate Traffic Volume dataset.

---

## Project Structure

```
├── data_pipeline.py       # MODULE 1 — Data Generator & Preprocessing
├── traffic_predictor.py   # MODULE 2 — ML Core Engine & Evaluation
├── simulator.py           # MODULE 3 — Multi-Agent A* Simulation Environment
├── app.py                 # MODULE 4 — Streamlit Dashboard
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit dashboard (recommended)
```bash
streamlit run app.py
```
Then use the sidebar buttons in order:
1. **📊 Generate Data** — runs Module 1, creates `traffic_clean.csv` (≈50,000 rows)
2. **🤖 Train Model**   — runs Module 2, trains MLP or KNN, prints Mean ± Std results
3. **▶ Start Sim**      — launches the real-time A* multi-agent simulation

### 3. Run modules independently
```bash
# Module 1 only
python data_pipeline.py

# Module 2 only (after Module 1 has run)
python traffic_predictor.py
# or with KNN:
MODEL_TYPE=KNN python traffic_predictor.py

# Module 3 smoke test
python simulator.py
```

---

## Academic Concepts Implemented

| Concept | Where | Details |
|---------|-------|---------|
| Data Preprocessing | `data_pipeline.py` | Median imputation, IQR clipping, LabelEncoder, OneHotEncoder, StandardScaler |
| Feature Engineering | `data_pipeline.py` | Temporal features from timestamp, congestion level discretisation |
| MLP Classifier | `traffic_predictor.py` | `MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu', solver='adam')` |
| KNN Classifier | `traffic_predictor.py` | `KNeighborsClassifier(n_neighbors=7, weights='distance')` |
| 5-Run Evaluation | `traffic_predictor.py` | 5 independent 80/20 splits; reports `Mean ± Std Dev` for Acc/Prec/Rec/F1 |
| A* Informed Search | `simulator.py` | `f(n)=g(n)+h(n)`; h = Manhattan distance + dynamic congestion penalty |
| Agent-Based Modelling | `simulator.py` | `VehicleAgent` class; autonomous navigation, red-light waiting, path replanning |
| Interactive Dashboard | `app.py` | Streamlit + Plotly; real-time heatmap, metric display, parameter sliders |

---

## Evaluation Output Format (Grading Rubric)

```
=======================================================
  EVALUATION RESULTS — MLP  (5 Independent Runs)
=======================================================
  Accuracy  : 0.XXXX ± 0.XXXX
  Precision : 0.XXXX ± 0.XXXX
  Recall    : 0.XXXX ± 0.XXXX
  F1-Score  : 0.XXXX ± 0.XXXX
=======================================================
```

---

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `Hour` | int | Hour of day (0–23) |
| `Day_of_Week` | int | 0=Monday … 6=Sunday |
| `Month` | int | 1–12 |
| `Temperature_K` | float | Kelvin, StandardScaled |
| `Rain_mm` | float | mm/hr, clipped & scaled |
| `Snow_mm` | float | mm/hr, clipped & scaled |
| `Cloud_Coverage` | float | 0–100 %, scaled |
| `Holiday_Encoded` | int | LabelEncoded holiday type |
| `Weather_Clear`, `Weather_Clouds`, … | int | One-hot encoded weather (8 columns) |
| `Congestion_Level` | int | **Target**: 0=Free Flow, 1=Moderate, 2=Heavy |

---

## A* Heuristic Details

```
f(n) = g(n) + h(n)

where:
  g(n) = actual path cost from start to n
         (BASE_MOVE_COST=1.0 per step + SIGNAL_RED_COST=5.0 through red lights)

  h(n) = Manhattan_Distance(n, goal)
         + CONGESTION_WEIGHT × Σ vehicle_count(adjacent_cells)

  CONGESTION_WEIGHT = 2.0  (tunable)
```

The heuristic is **admissible** (never over-estimates true cost) and **dynamic**
(updates each tick as vehicles move, steering agents to less-congested routes).
