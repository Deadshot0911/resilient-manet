cd ~/resilient-manet
nano README.md
RESILIENT-MANET
Robust and Energy-Efficient Secure Routing in MANETs with Federated Adversarial Learning

Live Dashboard
Open Live Dashboard

Overview
RESILIENT-MANET is a 5-phase AI pipeline for detecting and routing around malicious nodes in 100-node Mobile Ad-Hoc Networks (MANETs), evaluated on real NS-3 packet-level data.

Key Results
Metric	Value
Detection Rate	100.00%
False Positive Rate	0.00% - 1.10%
End-to-End Delay	0.067 ms
Throughput	1312.93 kbps
Energy Consumption	0.12 mJ/node
Zero-Day Detection	82.5%
On-Off Detection	76.7%
Ablation Study
Variant	DR (%)	FPR (%)
Base (no ML)	14.00	9.56
+ Bayesian Trust	100.00	0.00
+ Bayesian + FTL	100.00	0.00
+ Bayesian + FTL + MAML	100.00	0.00
Full Ensemble	100.00	0.00
System Architecture
text
NS-3 SIMULATION LAYER
  100 nodes, 802.11b, RandomWaypoint, YansWifiPhy
         |
         v
  9-D feature matrix H (packet-level traces)
         |
         v
CORE SECURITY & LEARNING ENGINES
  Phase 1     Phase 2      Phase 3      Phase 4
   FTL   ->   GATM    ->   Bayesian ->   MAML
         |
         v
ENSEMBLE FUSION & ROUTING
  Trust = 0.70*Bayes + 0.10*FTL + 0.10*GATM + 0.10*MAML
         |
         v
  HLOA-Optimized GNN Edge Scorer (53,249 parameters)
         |
         v
  Secure Multi-Hop Routing
5-Phase AI Pipeline
Phase 1 - FTL (Federated Trust Learning)
10 cluster heads each train a local model

Gradients L2-clipped (C=1.0) and perturbed with calibrated Gaussian noise

Reputation-weighted FedAvg

Formula: w_k = (N_k * Rep_k) / sum_j (N_j * Rep_j)

Phase 2 - GATM (Generative Adversarial Trust Model)
Generator: 8-D latent -> 16 hidden -> 9-D fake features

Discriminator: 9-D features -> 16 hidden -> 1 trust probability

Online adversarial sparring per timestep

Detects zero-day attacks via synthetic pattern generation

Phase 3 - Bayesian Trust Calibration
Beta(a, b) posterior per node

Sequential update with exponential decay lambda=0.95

Risk-adjusted score: mu - k*sigma (k=1.5)

Update: a(t) = a0 + lambda*(a(t-1) - a0) + s(t)

Phase 4 - MAML (Meta-Learning)
Inner loop LR alpha=0.05, 3 steps on support set

Outer loop meta-LR beta=0.01

5 attack-class task distribution

Fast few-shot adaptation to new attack signatures

Phase 5 - Ensemble + HLOA Routing
Weighted trust ensemble across all 4 phases

GNN edge scorer (53,249 parameters)

HLOA optimizer tunes edge weights over 50 iterations

Multi-hop forwarding only through high-trust paths

Mathematical Foundations
Differential Privacy:

sigma = sqrt(2 * ln(1.25/delta)) * C / epsilon

Sequential Bayesian Update:

a(t) = a0 + lambda(a(t-1) - a0) + s(t)
b(t) = b0 + lambda(b(t-1) - b0) + f(t)

Risk-Adjusted Score:

score = mu - k*sigma (k = 1.5)

MAML Bi-Level Optimization:

Inner: theta'_i = theta - alphagrad(L_task_i(theta))
Outer: theta <- theta - betagrad(sum_i L_task_i(theta'_i))

Attack Model
Attack	Behavior	Detection Rate
Blackhole	Unconditional packet dropping	100.0%
Grayhole	Selective packet dropping	100.0%
On-Off	Alternates honest/malicious	76.7%
Collusion	Coordinated reputation inflation	85.8%
Zero-Day	Novel attack patterns	82.5%
Tech Stack
Layer	Technology
Network Simulation	NS-3.44 (C++17)
Data Bridge	Python, XML, NumPy
Phase 1	NumPy, FedAvg, DP
Phase 2	NumPy (vectorized GAN)
Phase 3	NumPy, scipy.stats
Phase 4	NumPy (MAML)
Phase 5	NumPy, HLOA
Backend	FastAPI + Uvicorn
Frontend	Streamlit + Plotly
Tunneling	ngrok
Deployment	Streamlit Community Cloud
Project Structure
text
resilient-manet/
  app.py                          Streamlit dashboard
  config.py                       Hyperparameters
  main_resilient.py               Main pipeline
  ablation_study.py               Component ablation
  ns3_api.py                      FastAPI backend
  requirements.txt
  README.md
  modules/
    phase1_ftl.py
    phase2_gatm.py
    phase3_bayesian.py
    phase4_maml.py
    resilient_manet_engine.py
  ns3_scripts/
    resilient_manet_ns3.cc
    ns3_trace_parser.py
  outputs/
    node_features.npy
    resilient_manet_flowmon.xml
    resilient_manet_final_benchmark.json
    figures/
      results.png
      ablation.png
Local Setup
git clone https://github.com/Deadshot0911/resilient-manet.git
cd resilient-manet
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

Open http://localhost:8501

Deployment
Frontend: https://resilient-manet-sunritpanja.streamlit.app/

Backend (optional live NS-3): Local FastAPI + ngrok tunnel

Reproducibility
python main_resilient.py
python ablation_study.py
python plot_results.py
python plot_ablation.py

Author
Sunrit Panja
GitHub: https://github.com/Deadshot0911
Institution: Vellore Institute of Technology

License
MIT
