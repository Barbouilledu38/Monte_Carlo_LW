"""
generate_interactive_map.py
----------------------------
Converts the ipywidgets + matplotlib interactive camera plot
into a fully self-contained HTML file using Plotly.

Usage:
    python generate_interactive_map.py

Outputs:
    camera_map_interactive.html   (open in any browser, no Python needed)
"""

import numpy as np
import xarray as xr
import plotly.graph_objects as go
import json
import os

class NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars and arrays to plain Python types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

# ── 1. PATHS ─────────────────────────────────────────────────────────────────
POS_DICT_PATH = "/home/barroisl/Transect_MC_auto/Output/guiers_250/pos_dict.npy"
TOPO_PATH     = "/home/barroisl/Transect_MC_auto/topographie/topo_params_250m.nc"
OUTPUT_HTML   = "/home/barroisl/Transect_MC_auto/Output/camera_map_interactive.html"

# ── 2. CONSTANTS ──────────────────────────────────────────────────────────────
XLIM     = (234000, 258000)
YLIM     = (5020000, 5038000)
MIN_FLUX = 280
MAX_FLUX = 400
CMAP     = "hot"
N_ROWS   = 12
N_COLS   = 12

QUANTILE_COLORS    = ["lightblue", "dodgerblue", "darkblue"]
QUANTILE_LABELS    = [90, 50, 10]
LAMARE_COLOR       = "red"
LAMARE_RADIUS      = 2000.0
THETAS             = np.linspace(0, 2 * np.pi, 300)

# ── 3. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading pos_dict …")
d_f = np.load(POS_DICT_PATH, allow_pickle=True)
pos_dict = d_f.item()

print("Loading topography …")
ds_topo = xr.open_dataset(TOPO_PATH)

# Crop topo to xlim/ylim for speed
topo_crop = ds_topo.sel(
    x=slice(XLIM[0], XLIM[1]),
    y=slice(YLIM[0], YLIM[1])
)
X_topo = topo_crop.x.values
Y_topo = topo_crop.y.values
Z_topo = topo_crop.zs.values          # shape (y, x)

# ── 4. BUILD TOPO CONTOUR (shared across all cameras) ────────────────────────
print("Building topography contour trace …")
topo_trace = go.Contour(
    x=X_topo,
    y=Y_topo,
    z=Z_topo,
    colorscale="earth",
    contours=dict(
        start=200, end=1800, size=100,
        showlabels=False,
    ),
    opacity=0.5,
    showscale=False,
    hoverinfo="skip",
    name="Topographie",
    line=dict(width=0.5),
)

# ── 5. PRE-COMPUTE ALL 144 CAMERA TRACE-SETS ─────────────────────────────────
print("Pre-computing all 144 camera plots …")

# We will store, for every camera index, a list of serialisable dicts
# ready to be injected into Plotly figure data via JS.
all_camera_data = {}   # key: str(camera_index)

for i in range(1, N_ROWS + 1):
    for j in range(1, N_COLS + 1):
        idx = (i - 1) * N_COLS + j
        data = pos_dict.get(idx)
        if data is None:
            continue

        traces = []

        # 5a. Scatter: absorption points coloured by flux
        traces.append(dict(
            type="scatter",
            x=data[:, 0].tolist(),
            y=data[:, 1].tolist(),
            mode="markers",
            marker=dict(
                color=data[:, 2].tolist(),
                colorscale=CMAP,
                cmin=MIN_FLUX,
                cmax=MAX_FLUX,
                size=4,
                colorbar=dict(
                    title="flux W.m⁻²",
                    len=0.5,
                    yanchor="middle",
                    y=0.5,
                ),
                showscale=True,
            ),
            name="Absorption",
            hovertemplate="x=%{x:.0f} m<br>y=%{y:.0f} m<br>flux=%{marker.color:.1f} W/m²<extra></extra>",
        ))

        # 5b. Camera position star marker
        cam_x = float(data[0, 3])
        cam_y = float(data[0, 4])
        tgt_x = float(data[0, 5])
        tgt_y = float(data[0, 6])

        traces.append(dict(
            type="scatter",
            x=[cam_x],
            y=[cam_y],
            mode="markers",
            marker=dict(symbol="star", size=14, color="dodgerblue"),
            name="Caméra",
            hovertemplate=f"Caméra {idx}<br>x={cam_x:.0f} m<br>y={cam_y:.0f} m<extra></extra>",
        ))

        # 5c. Arrow (camera → target) as annotation — stored separately
        arrow = dict(
            ax=cam_x, ay=cam_y,
            x=tgt_x,  y=tgt_y,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowwidth=2,
            arrowcolor="red",
        )

        # 5d. Quantile circles
        for k, (R, col, q) in enumerate(zip(
                [float(data[0, 7]), float(data[0, 8]), float(data[0, 9])],
                QUANTILE_COLORS, QUANTILE_LABELS)):
            xs = (R * np.cos(THETAS) + cam_x).tolist()
            ys = (R * np.sin(THETAS) + cam_y).tolist()
            traces.append(dict(
                type="scatter",
                x=xs, y=ys,
                mode="lines",
                line=dict(color=col, width=1.5),
                name=f"Quantile {q} ≈ {int(R)} m",
            ))

        # 5e. Lamare 2020 circle (2000 m)
        xs_l = (LAMARE_RADIUS * np.cos(THETAS) + cam_x).tolist()
        ys_l = (LAMARE_RADIUS * np.sin(THETAS) + cam_y).tolist()
        traces.append(dict(
            type="scatter",
            x=xs_l, y=ys_l,
            mode="lines",
            line=dict(color=LAMARE_COLOR, width=1.5, dash="dash"),
            name="Lamare & al 2020 = 2000 m",
        ))

        all_camera_data[str(idx)] = dict(traces=traces, arrow=arrow)

print(f"  → {len(all_camera_data)} cameras built.")

# ── 6. SERIALISE TO JSON ──────────────────────────────────────────────────────
print("Serialising to JSON …")
camera_json = json.dumps(all_camera_data, cls=NumpyEncoder)

# Topo trace as JSON for injection
topo_json = topo_trace.to_plotly_json()
topo_json_str = json.dumps(topo_json, cls=NumpyEncoder)

# ── 7. WRITE SELF-CONTAINED HTML ──────────────────────────────────────────────
print("Writing HTML …")

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Carte interactive des caméras</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 24px 16px;
  }}
  h1 {{
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 20px;
    color: #a0c4ff;
  }}
  .controls {{
    display: flex;
    gap: 40px;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    background: #16213e;
    border-radius: 12px;
    padding: 16px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }}
  .slider-group {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }}
  .slider-group label {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #a0c4ff;
  }}
  .slider-row {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  input[type=range] {{
    -webkit-appearance: none;
    width: 220px;
    height: 6px;
    border-radius: 3px;
    background: #0f3460;
    outline: none;
    cursor: pointer;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 20px; height: 20px;
    border-radius: 50%;
    background: #a0c4ff;
    cursor: pointer;
    box-shadow: 0 0 6px rgba(160,196,255,0.5);
  }}
  .val-badge {{
    background: #0f3460;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 1rem;
    font-weight: 700;
    color: #a0c4ff;
    min-width: 36px;
    text-align: center;
  }}
  .cam-badge {{
    font-size: 0.9rem;
    color: #ccc;
  }}
  #plot {{
    width: min(900px, 98vw);
    height: min(800px, 88vh);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }}
</style>
</head>
<body>
<h1>Carte des positions d'absorption — guiers 250</h1>

<div class="controls">
  <div class="slider-group">
    <label>Ligne (i)</label>
    <div class="slider-row">
      <input type="range" id="slider_i" min="1" max="{N_ROWS}" step="1" value="1">
      <span class="val-badge" id="val_i">1</span>
    </div>
  </div>
  <div class="slider-group">
    <label>Colonne (j)</label>
    <div class="slider-row">
      <input type="range" id="slider_j" min="1" max="{N_COLS}" step="1" value="1">
      <span class="val-badge" id="val_j">1</span>
    </div>
  </div>
  <div class="cam-badge">Caméra n° <span id="cam_idx">1</span></div>
</div>

<div id="plot"></div>

<script>
// ── Embedded data ──────────────────────────────────────────────────────────
const CAMERA_DATA = {camera_json};
const TOPO_TRACE  = {topo_json_str};

const XLIM = [{XLIM[0]}, {XLIM[1]}];
const YLIM = [{YLIM[0]}, {YLIM[1]}];
const N_COLS = {N_COLS};

// ── Init Plotly ────────────────────────────────────────────────────────────
const layout = {{
  paper_bgcolor: "#16213e",
  plot_bgcolor:  "#0f3460",
  font: {{ color: "#e0e0e0", size: 11 }},
  xaxis: {{
    range: XLIM,
    title: "x (m)",
    gridcolor: "#1a3a6e",
    showgrid: true,
    scaleanchor: "y",
    scaleratio: 1,
  }},
  yaxis: {{
    range: YLIM,
    title: "y (m)",
    gridcolor: "#1a3a6e",
    showgrid: true,
  }},
  title: {{
    text: "Caméra n°1",
    font: {{ size: 15, color: "#a0c4ff" }},
  }},
  legend: {{
    bgcolor: "rgba(15,52,96,0.85)",
    bordercolor: "#a0c4ff",
    borderwidth: 1,
  }},
  margin: {{ l: 60, r: 20, t: 50, b: 50 }},
  annotations: [],
}};

Plotly.newPlot("plot", [TOPO_TRACE], layout, {{
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ["select2d","lasso2d"],
}});

// ── Update function ────────────────────────────────────────────────────────
function update() {{
  const i   = parseInt(document.getElementById("slider_i").value);
  const j   = parseInt(document.getElementById("slider_j").value);
  const idx = (i - 1) * N_COLS + j;

  document.getElementById("val_i").textContent   = i;
  document.getElementById("val_j").textContent   = j;
  document.getElementById("cam_idx").textContent = idx;

  const cam = CAMERA_DATA[String(idx)];
  if (!cam) {{
    Plotly.react("plot", [TOPO_TRACE],
      Object.assign({{}}, layout, {{
        title: {{ text: `Caméra n°${{idx}} — données manquantes`, font: {{ size: 15, color: "#a0c4ff" }} }},
        annotations: [],
      }}));
    return;
  }}

  const traces = [TOPO_TRACE, ...cam.traces];
  const newLayout = Object.assign({{}}, layout, {{
    title: {{ text: `Caméra n°${{idx}}`, font: {{ size: 15, color: "#a0c4ff" }} }},
    annotations: [cam.arrow],
  }});

  Plotly.react("plot", traces, newLayout);
}}

// ── Wire sliders ───────────────────────────────────────────────────────────
document.getElementById("slider_i").addEventListener("input", update);
document.getElementById("slider_j").addEventListener("input", update);

// Init
update();
</script>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅  Done!  →  {OUTPUT_HTML}")
print("   Open this file in any browser — no Python or server needed.")
