"""
Parse NS-3 FlowMonitor XML into a 9-D per-node feature matrix.

NS-3 FlowMonitor XML aggregates per-flow stats (txPackets, rxPackets, etc.)
but does NOT include per-node source/destination. We aggregate totals and
distribute them across nodes with mild per-node noise, then rely on the
API's attack injection to create realistic ground truth.
"""
import xml.etree.ElementTree as ET
import numpy as np


def _parse_ns(s):
    """Parse '+3.56e+08ns' -> float nanoseconds."""
    if not s:
        return 0.0
    s = s.strip().replace('ns', '').replace('+', '')
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_flowmon(xml_path, num_nodes=100):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    total_tx = 0
    total_rx = 0
    total_lost = 0
    total_delay_ns = 0.0
    total_tx_bytes = 0
    total_rx_bytes = 0
    n_flows = 0

    for flow in root.iter('Flow'):
        n_flows += 1
        total_tx += int(flow.get('txPackets', 0))
        total_rx += int(flow.get('rxPackets', 0))
        total_lost += int(flow.get('lostPackets', 0))
        total_tx_bytes += int(flow.get('txBytes', 0))
        total_rx_bytes += int(flow.get('rxBytes', 0))
        total_delay_ns += _parse_ns(flow.get('delaySum', '0'))

    H = np.zeros((num_nodes, 9))
    if n_flows == 0:
        return H

    np.random.seed(42)

    avg_tx = max(total_tx / num_nodes, 1)
    avg_rx = max(total_rx / num_nodes, 1)
    avg_lost = max(total_lost / num_nodes, 1)
    avg_delay_ms = (total_delay_ns / max(n_flows, 1)) / 1e6
    if avg_delay_ms <= 0:
        avg_delay_ms = 0.5
    avg_rx_bytes = max(total_rx_bytes / num_nodes, 1)

    for i in range(num_nodes):
        tx = avg_tx * np.random.uniform(0.7, 1.3)
        rx = avg_rx * np.random.uniform(0.7, 1.3)
        lost = avg_lost * np.random.uniform(0.5, 1.5)

        H[i, 0] = tx                                          # packets sent
        H[i, 1] = rx                                          # packets received
        H[i, 2] = lost                                        # packets dropped
        H[i, 3] = lost / max(tx, 1)                           # drop ratio
        H[i, 4] = avg_rx_bytes * 8 / 40.0 / 1000.0            # throughput kbps
        H[i, 5] = avg_delay_ms * np.random.uniform(0.8, 1.2)  # delay ms
        H[i, 6] = 0.05 * tx + 0.001 * rx                      # energy mJ
        H[i, 7] = rx / max(tx, 1)                             # forwarding ratio
        H[i, 8] = H[i, 7]                                     # trust proxy

    return H


if __name__ == "__main__":
    H = parse_flowmon("outputs/resilient_manet_flowmon.xml")
    np.save("outputs/node_features.npy", H)
    print(f"Feature matrix shape: {H.shape}")
    print(f"Nonzero elements: {np.count_nonzero(H)}")
    print(f"Drop ratio range: {H[:,3].min():.4f} to {H[:,3].max():.4f}")
    print(f"Throughput mean: {H[:,4].mean():.2f} kbps")
