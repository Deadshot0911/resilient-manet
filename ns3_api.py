"""
NS-3 Simulation API — exposes local NS-3 runs to remote clients via HTTP.
Run: python ns3_api.py
Then expose with: ngrok http 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import shutil
import numpy as np

app = FastAPI(title="RESILIENT-MANET NS-3 API", version="1.0.0")

# Allow the Streamlit cloud app to call this API from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths — adjust if your NS-3 directory differs
NS3_DIR = os.path.expanduser("~/ns-allinone-3.44/ns-3.44")
SIM_SCRIPT = "scratch/resilient_manet_ns3"
PROJECT_DIR = os.path.expanduser("~/resilient-manet")

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "RESILIENT-MANET NS-3 API",
        "ns3_dir": NS3_DIR,
        "project_dir": PROJECT_DIR,
    }

@app.get("/health")
def health():
    """Quick check that NS-3 dir exists."""
    return {
        "ns3_exists": os.path.isdir(NS3_DIR),
        "sim_script_exists": os.path.isfile(os.path.join(NS3_DIR, SIM_SCRIPT + ".cc")),
    }

@app.post("/run-simulation")
def run_simulation(seed: int = 42, nodes: int = 100, duration: float = 40.0):
    """
    Run the NS-3 C++ simulation, parse the resulting XML, and save
    the feature matrix to outputs/node_features.npy.

    Returns the parsed feature matrix as a JSON-able list so the caller
    can use the data directly without needing filesystem access.
    """
    try:
        # Call the binary directly (bypasses the ./ns3 wrapper which breaks with args)
        binary_path = os.path.join(NS3_DIR, "build/scratch/ns3.44-resilient_manet_ns3-default")

        if not os.path.isfile(binary_path):
            raise HTTPException(status_code=500, detail=f"Binary not found: {binary_path}")

        # Delete old XML first
        old_xml = os.path.join(NS3_DIR, "resilient_manet_flowmon.xml")
        if os.path.exists(old_xml):
            os.remove(old_xml)

        result = subprocess.run(
            [binary_path,
             f"--nNodes={nodes}",
             f"--duration={duration}",
             f"--seed={seed}"],
            cwd=NS3_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"NS-3 failed (code {result.returncode}): {result.stderr[-500:]}"
            )

        # NS-3 writes the XML in its own directory
        src_xml = os.path.join(NS3_DIR, "resilient_manet_flowmon.xml")
        if not os.path.isfile(src_xml):
            raise HTTPException(status_code=500, detail="NS-3 finished but XML not found.")

        # Copy XML to our project outputs
        dst_xml = os.path.join(PROJECT_DIR, "outputs", "resilient_manet_flowmon.xml")
        os.makedirs(os.path.dirname(dst_xml), exist_ok=True)
        shutil.copy(src_xml, dst_xml)

        # Parse XML into feature matrix
        import sys
        sys.path.insert(0, PROJECT_DIR)
        from ns3_scripts.ns3_trace_parser import parse_flowmon

        H = parse_flowmon(dst_xml, num_nodes=nodes)

        # Save .npy for local usage
        npy_path = os.path.join(PROJECT_DIR, "outputs", "node_features.npy")
        np.save(npy_path, H)

        # Also inject attacks so the dashboard has a ground truth to compare against
        # (same logic as your inject_attacks.py)
        H_with_attacks = H.copy()
        np.random.seed(seed)
        n_mal = max(int(nodes * 0.10), 1)
        mal_nodes = np.random.choice(nodes, n_mal, replace=False)
        H_with_attacks[mal_nodes, 1] = H_with_attacks[mal_nodes, 0] * np.random.uniform(0.05, 0.20, n_mal)
        H_with_attacks[mal_nodes, 2] = H_with_attacks[mal_nodes, 0] - H_with_attacks[mal_nodes, 1]
        H_with_attacks[mal_nodes, 3] = H_with_attacks[mal_nodes, 2] / np.maximum(H_with_attacks[mal_nodes, 0], 1)
        H_with_attacks[mal_nodes, 7] = H_with_attacks[mal_nodes, 1] / np.maximum(H_with_attacks[mal_nodes, 0], 1)
        H_with_attacks[mal_nodes, 8] = H_with_attacks[mal_nodes, 7].copy()

        np.save(npy_path, H_with_attacks)

        # Return summary + matrix (as list) so the caller can use it immediately
        return {
            "status": "success",
            "seed": seed,
            "nodes": nodes,
            "duration": duration,
            "feature_shape": list(H_with_attacks.shape),
            "malicious_count": n_mal,
            "malicious_nodes": sorted(mal_nodes.tolist()),
            "feature_matrix": H_with_attacks.tolist(),
            "xml_saved_to": dst_xml,
            "npy_saved_to": npy_path,
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Simulation timed out (>10 min).")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
