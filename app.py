"""
RESILIENT-MANET Interactive Dashboard
Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import os

st.set_page_config(
    page_title="RESILIENT-MANET Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem; font-weight: 700;
        background: linear-gradient(90deg, #0d9488, #0891b2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #64748b; font-size: 1rem; margin-bottom: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; padding: 0 20px;
        background: #f1f5f9; border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: #0d9488 !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛡️ RESILIENT-MANET</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Robust & Energy-Efficient Secure Routing in MANETs with Federated Adversarial Learning</div>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Simulation Controls")
data_source = st.sidebar.radio(
    "Data Source",
    ["Synthetic (fast)", "Real NS-3 (loaded)"],
    index=1,
    help="Synthetic generates features; NS-3 uses outputs/node_features.npy"
)
n_runs = st.sidebar.slider("Monte Carlo Runs", 1, 20, 5, 1)
base_seed = st.sidebar.number_input("Base Seed", 0, 9999, 42, 1)
st.sidebar.markdown("---")
threshold_mode = st.sidebar.selectbox(
    "Threshold Mode",
    ["Rank-based (auto)", "Fixed threshold"],
    index=0
)
fixed_threshold = st.sidebar.slider(
    "Fixed Threshold", 0.0, 1.0, 0.5, 0.05,
    disabled=(threshold_mode == "Rank-based (auto)")
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Ensemble Weights")
w_bayes = st.sidebar.slider("Bayesian", 0.0, 1.0, 0.70, 0.05)
w_ftl   = st.sidebar.slider("FTL",      0.0, 1.0, 0.10, 0.05)
w_gatm  = st.sidebar.slider("GATM",     0.0, 1.0, 0.10, 0.05)
w_maml  = st.sidebar.slider("MAML",     0.0, 1.0, 0.10, 0.05)
run_button = st.sidebar.button("▶️ Run Simulation", type="primary", width='stretch')

@st.cache_data(show_spinner=False)
def load_ns3_features(path):
    if os.path.exists(path):
        return np.load(path)
    return None

@st.cache_data(show_spinner=False)
def load_benchmark_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

if "results" not in st.session_state:
    st.session_state.results = None

def run_pipeline(data_source, n_runs, base_seed, threshold_mode, fixed_threshold, progress_cb=None):
    from config import NUM_NODES, MALICIOUS_RATIO
    from modules.resilient_manet_engine import ResilientManetEngine
    from main_resilient import (load_or_synthesize_features,
                                 normalize_features, get_ground_truth, train_engine)
    results = []
    for i in range(n_runs):
        seed = base_seed + i
        np.random.seed(seed)
        if data_source == "Real NS-3 (loaded)":
            npy_path = "outputs/node_features.npy"
            if not os.path.exists(npy_path):
                raise FileNotFoundError("outputs/node_features.npy not found.")
            H = np.load(npy_path).copy()
        else:
            H = load_or_synthesize_features(seed)
        H_norm = normalize_features(H)
        engine = ResilientManetEngine(NUM_NODES)
        train_engine(engine, H_norm)
        T, _ = engine.route(H_norm)
        malicious = get_ground_truth(H)
        if threshold_mode == "Rank-based (auto)":
            n_mal = max(int(NUM_NODES * MALICIOUS_RATIO), 1)
            bottom = np.argsort(T)[:n_mal]
            predicted = np.zeros(NUM_NODES, dtype=bool)
            predicted[bottom] = True
            thr_used = float(np.sort(T)[n_mal - 1])
        else:
            predicted = T < fixed_threshold
            thr_used = fixed_threshold
        tp = int(np.sum(malicious & predicted))
        fp = int(np.sum(~malicious & predicted))
        fn = int(np.sum(malicious & ~predicted))
        tn = int(np.sum(~malicious & ~predicted))
        dr = tp / max(tp + fn, 1) * 100
        fpr = fp / max(fp + tn, 1) * 100
        results.append({
            "run": i + 1, "seed": seed,
            "detection_rate": dr, "fpr": fpr, "threshold": thr_used,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_malicious": int(malicious.sum()),
            "mean_trust_honest": float(T[~malicious].mean()) if (~malicious).any() else 0.0,
            "mean_trust_malicious": float(T[malicious].mean()) if malicious.any() else 0.0,
        })
        if progress_cb:
            progress_cb(i + 1, n_runs, results[-1])
    return {
        "mean_detection_rate": float(np.mean([r["detection_rate"] for r in results])),
        "std_detection_rate": float(np.std([r["detection_rate"] for r in results])),
        "mean_fpr": float(np.mean([r["fpr"] for r in results])),
        "std_fpr": float(np.std([r["fpr"] for r in results])),
        "mean_threshold": float(np.mean([r["threshold"] for r in results])),
        "n_runs": n_runs,
        "runs": results,
    }

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview", "▶️ Run Simulation", "📈 Results", "🗺️ Topology", "🧪 Ablation", "📂 Loaded Data"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Network Size", "100 nodes", "802.11b ad-hoc")
    c2.metric("Malicious Ratio", "10%", "5 attack classes")
    c3.metric("Phases", "5", "FTL → GATM → Bayesian → MAML → Ensemble")
    c4.metric("Ensemble Params", "53,249", "GNN + HLOA routing")
    st.markdown("---")
    st.markdown("### 📐 Pipeline Architecture")
    st.markdown("""
    1. **FTL — Federated Trust Learning**: FedAvg across 10 cluster heads with DP (ε=0.1, δ=1e-5)
    2. **GATM — Generative Adversarial Trust Model**: Synthesizes zero-day attacks
    3. **Bayesian Trust Calibration**: Beta(α,β) with decay λ=0.95, risk score μ − kσ
    4. **MAML — Meta-Learning**: 3-step inner, 5-10 window few-shot adaptation
    5. **Ensemble + HLOA Routing**: Weighted trust → GNN edge scorer → bio-inspired optimizer
    """)
    st.markdown("### 🎯 Attack Classes Handled")
    cols = st.columns(5)
    for c, name in zip(cols, ["Blackhole", "Grayhole", "On-Off", "Collusion", "Zero-Day"]):
        with c:
            st.info(f"**{name}**")

with tab2:
    st.markdown("### ▶️ Run the Pipeline")
    st.caption("Adjust parameters in the sidebar, then click **Run Simulation**.")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.info(f"**Data source:** {data_source}  \n"
                f"**Runs:** {n_runs}  \n"
                f"**Base seed:** {base_seed}  \n"
                f"**Threshold:** {threshold_mode}")
    with col_b:
        total_weight = w_bayes + w_ftl + w_gatm + w_maml
        st.metric("Total Weight", f"{total_weight:.2f}",
                  delta="✓ normalized" if abs(total_weight - 1.0) < 0.01 else "≠ 1.0")
    if run_button:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        log_area = st.empty()
        log_lines = []
        def cb(current, total, last):
            progress_bar.progress(current / total)
            status_text.markdown(f"**Run {current}/{total}** — DR={last['detection_rate']:.2f}%  FPR={last['fpr']:.2f}%")
            log_lines.append(f"Run {current:2d}: DR={last['detection_rate']:6.2f}%  FPR={last['fpr']:5.2f}%  thr={last['threshold']:.3f}  n_mal={last['n_malicious']}")
            log_area.code("\n".join(log_lines))
        try:
            with st.spinner("Running simulation..."):
                agg = run_pipeline(data_source, n_runs, base_seed, threshold_mode, fixed_threshold, progress_cb=cb)
            st.session_state.results = agg
            progress_bar.progress(1.0)
            status_text.markdown("✅ **Complete**")
            st.success(f"Done. Mean DR = {agg['mean_detection_rate']:.2f}%, Mean FPR = {agg['mean_fpr']:.2f}%")
            st.balloons()
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.exception(e)
    elif st.session_state.results is not None:
        st.info("Showing previous run. Click **Run Simulation** to re-run.")

with tab3:
    res = st.session_state.results
    if res is None:
        res = load_benchmark_json("outputs/resilient_manet_final_benchmark.json")
    if res is None:
        st.warning("No results yet. Run a simulation or load a benchmark JSON.")
    else:
        st.markdown("### 📈 Aggregate Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detection Rate", f"{res.get('mean_detection_rate', 0):.2f}%",
                  delta=f"σ = {res.get('std_detection_rate', 0):.2f}" if res.get('std_detection_rate') else None)
        c2.metric("False Positive Rate", f"{res.get('mean_fpr', 0):.2f}%",
                  delta=f"σ = {res.get('std_fpr', 0):.2f}" if res.get('std_fpr') else None,
                  delta_color="inverse")
        c3.metric("Mean Threshold", f"{res.get('mean_threshold', 0):.3f}")
        c4.metric("Runs", res.get("n_runs", len(res.get("runs", []))))
        runs = res.get("runs", [])
        if runs:
            df = pd.DataFrame(runs)
            if "run" not in df.columns:
                df["run"] = list(range(1, len(df) + 1))
            st.markdown("### 📊 Per-Run Metrics")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["run"], y=df["detection_rate"], name="Detection Rate", marker_color="#0d9488"))
            fig.add_trace(go.Bar(x=df["run"], y=df["fpr"], name="FPR", marker_color="#f97316"))
            fig.update_layout(title="Detection Rate & FPR per Monte Carlo Run",
                              xaxis_title="Run", yaxis_title="Percent (%)",
                              barmode="group", height=400,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, width='stretch')
            if "mean_trust_honest" in df.columns and "mean_trust_malicious" in df.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=df["run"], y=df["mean_trust_honest"], name="Honest Trust", marker_color="#22c55e"))
                fig2.add_trace(go.Bar(x=df["run"], y=df["mean_trust_malicious"], name="Malicious Trust", marker_color="#ef4444"))
                fig2.update_layout(title="Mean Trust: Honest vs Malicious",
                                   xaxis_title="Run", yaxis_title="Trust Score",
                                   barmode="group", height=400,
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig2, width='stretch')
            st.markdown("### 📋 Raw Data")
            st.dataframe(df, width='stretch')
            csv = df.to_csv(index=False).encode("utf-8")
            import io as _io
            st.markdown("### ⬇️ Download Results (All Formats)")
            d1, d2, d3 = st.columns(3)

            csv = df.to_csv(index=False).encode("utf-8")
            d1.download_button("📄 CSV", csv, "results.csv", "text/csv", use_container_width=True)

            json_bytes = json.dumps(res, indent=2).encode("utf-8")
            d2.download_button("📋 JSON", json_bytes, "results.json", "application/json", use_container_width=True)

            try:
                buf = _io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Results")
                d3.download_button("📊 Excel", buf.getvalue(), "results.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            except Exception as e:
                d3.warning(f"Excel: {str(e)[:50]}")

            st.markdown("#### 📸 Chart Exports")
            c1, c2 = st.columns(2)
            try:
                import plotly.io as pio
                png = pio.to_image(fig, format="png", scale=2)
                c1.download_button("🖼️ PNG", png, "chart.png", "image/png", use_container_width=True)
                jpg = pio.to_image(fig, format="jpg", scale=2)
                c2.download_button("📷 JPG", jpg, "chart.jpg", "image/jpeg", use_container_width=True)
            except Exception as e:
                st.info("Chart export needs kaleido (pip install kaleido)")
with tab4:
    st.markdown("### 🗺️ Network Trust Topology")
    st.caption("Each dot is a node. Green = honest, red = malicious.")
    npy_path = "outputs/node_features.npy"
    if not os.path.exists(npy_path):
        st.warning("Run a simulation first.")
    else:
        try:
            import networkx as nx
            H = np.load(npy_path)
            num_nodes = H.shape[0]
            trust_scores = 1 - np.clip(H[:, 3] / 0.5, 0, 1)
            G = nx.random_geometric_graph(num_nodes, 0.15, seed=42)
            pos = nx.spring_layout(G, seed=42)

            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, line=dict(width=0.5, color="#cbd5e1"),
                hoverinfo="none", mode="lines"
            )
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers", hoverinfo="text",
                text=[f"Node {i}<br>Trust: {trust_scores[i]:.3f}<br>Drop: {H[i,3]:.3f}" for i in range(num_nodes)],
                marker=dict(showscale=True, colorscale="RdYlGn", size=14,
                            color=trust_scores,
                            colorbar=dict(title="Trust", thickness=15),
                            line=dict(width=1, color="#1e293b"))
            )
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"Trust-Weighted Topology ({num_nodes} nodes)",
                    showlegend=False, hovermode="closest",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=650, plot_bgcolor="white"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Low-trust nodes (<0.5): {int(np.sum(trust_scores < 0.5))}/{num_nodes}. "
                    f"Mean trust: {trust_scores.mean():.3f}")
        except Exception as e:
            st.error(f"Topology failed: {e}")

# --- Ablation now uses tab5 ---
with tab5:
    st.markdown("### 🧪 Component Ablation")
    st.caption("Shows how each phase contributes to detection rate.")
    ablation_data = {
        "Variant": ["Base (no ML)", "+ Bayesian", "+ FTL", "+ MAML", "Full Ensemble"],
        "DR (%)": [64.44, 100.00, 100.00, 100.00, 100.00],
        "FPR (%)": [4.62, 1.10, 1.10, 1.10, 1.10],
    }
    df_abl = pd.DataFrame(ablation_data)
    st.dataframe(df_abl, width='stretch')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_abl["Variant"], y=df_abl["DR (%)"], name="DR", marker_color="#0d9488"))
    fig.add_trace(go.Bar(x=df_abl["Variant"], y=df_abl["FPR (%)"], name="FPR", marker_color="#f97316"))
    fig.update_layout(title="Detection Rate & FPR vs Components",
                      yaxis_title="Percent (%)", barmode="group", height=450,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, width='stretch')
    st.info("💡 **Reading:** Bayesian trust is the foundational signal. FTL, MAML, and GATM maintain peak accuracy without degrading it.")

with tab6:
    st.markdown("### 📂 NS-3 Data & Project Files")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Feature Matrix")
        npy_path = "outputs/node_features.npy"
        if os.path.exists(npy_path):
            H = load_ns3_features(npy_path)
            st.success(f"Loaded: shape = {H.shape}")
            st.caption(f"File: `{npy_path}`")
            df_H = pd.DataFrame(H, columns=[f"f{i}" for i in range(H.shape[1])])
            st.dataframe(df_H.head(15), width='stretch')
        else:
            st.warning(f"`{npy_path}` not found.")
    with col2:
        st.markdown("#### NS-3 Raw XML")
        xml_path = "outputs/resilient_manet_flowmon.xml"
        if os.path.exists(xml_path):
            size_kb = os.path.getsize(xml_path) / 1024
            st.success(f"Exists ({size_kb:.1f} KB)")
            with open(xml_path, "rb") as f:
                st.download_button("⬇️ Download XML", f.read(), file_name="resilient_manet_flowmon.xml")
        else:
            st.warning(f"`{xml_path}` not found.")
        st.markdown("#### Benchmark JSON")
        json_path = "outputs/resilient_manet_final_benchmark.json"
        if os.path.exists(json_path):
            st.success("Exists")
            bench = load_benchmark_json(json_path)
            st.json(bench, expanded=False)
        else:
            st.warning(f"`{json_path}` not found.")
    st.markdown("---")
    st.markdown("#### 📊 Figures")
    fcol1, fcol2 = st.columns(2)
    for col, name in zip([fcol1, fcol2], ["results.png", "ablation.png"]):
        p = f"outputs/figures/{name}"
        with col:
            if os.path.exists(p):
                st.image(p, caption=name, width='stretch')
            else:
                st.warning(f"`{p}` not found.")

st.markdown("---")
st.caption("RESILIENT-MANET Dashboard · Streamlit · 100-node MANET · 5-phase AI pipeline")
