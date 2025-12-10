"""
draw_vqc.py

Simple script to visualize VQC circuits for presentations.

Usage:
    python draw_vqc.py --vqc_type basic --num_layers 2 --n_qubits 9 --output circuit.png
    python draw_vqc.py --vqc_type basic --num_layers 2 --n_qubits 9 --noise depol --noise_prob 0.01 --output circuit_noisy.png
"""

import argparse
import torch
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
import os


def draw_circuit(vqc_type, num_layers, n_qubits, output_path, 
                 noise_type=None, noise_prob=0.0):
    """
    Draw and save a VQC circuit diagram.
    
    Args:
        vqc_type: "basic" or "expressive"
        num_layers: Number of VQC layers
        n_qubits: Number of qubits
        output_path: Where to save the PNG
        noise_type: None, "bit_flip", "amp_damp", or "depol"
        noise_prob: Noise probability (0.0 - 1.0)
    """
    
    # Setup device
    use_mixed = noise_type is not None and noise_prob > 0
    dev_type = "default.mixed" if use_mixed else "default.qubit"
    dev = qml.device(dev_type, wires=n_qubits)
    
    print(f"🎨 Drawing {vqc_type} VQC circuit:")
    print(f"   Qubits: {n_qubits}")
    print(f"   Layers: {num_layers}")
    print(f"   Device: {dev_type}")
    if use_mixed:
        print(f"   Noise: {noise_type} (p={noise_prob})")
    
    # Initialize dummy weights
    if vqc_type == "basic":
        weights = torch.rand(num_layers, n_qubits) * np.pi
    elif vqc_type == "expressive":
        sq_weights = torch.rand(num_layers, n_qubits, 3) * np.pi
        ent_weights = torch.rand(num_layers, n_qubits, n_qubits) * np.pi
    else:
        raise ValueError(f"Unknown vqc_type: {vqc_type}")
    
    # Create dummy input
    dummy_x = torch.randn(2 ** n_qubits)
    dummy_x = dummy_x / torch.norm(dummy_x)
    
    # Define noise application
    def apply_noise():
        if not use_mixed:
            return
        for q in range(n_qubits):
            if noise_type == "bit_flip":
                qml.BitFlip(noise_prob, wires=q)
            elif noise_type == "amp_damp":
                qml.AmplitudeDamping(noise_prob, wires=q)
            elif noise_type == "depol":
                qml.DepolarizingChannel(noise_prob, wires=q)
    
    # Define circuit based on type
    if vqc_type == "basic":
        @qml.qnode(dev, interface="torch")
        def circuit(x, w):
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True)
            
            for layer in range(num_layers):
                # Rotation layer
                for q in range(n_qubits):
                    qml.RY(w[layer, q], wires=q)
                
                # Ring entanglement
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
                
                # Noise
                apply_noise()
            
            return qml.probs(wires=0)
        
        # Draw
        fig, ax = qml.draw_mpl(circuit)(dummy_x, weights)
    
    elif vqc_type == "expressive":
        def all_to_all_isingzz(w):
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qml.IsingZZ(w[i, j], wires=[i, j])
        
        @qml.qnode(dev, interface="torch")
        def circuit(x, sq_w, ent_w):
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), normalize=True)
            
            for layer in range(num_layers):
                # Single qubit rotations
                for q in range(n_qubits):
                    qml.U3(*sq_w[layer, q], wires=q)
                
                # All-to-all entanglement
                all_to_all_isingzz(ent_w[layer])
                
                # Noise
                apply_noise()
            
            return qml.probs(wires=0)
        
        # Draw
        fig, ax = qml.draw_mpl(circuit)(dummy_x, sq_weights, ent_weights)
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved circuit diagram to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Draw VQC circuit diagrams")
    
    parser.add_argument("--vqc_type", type=str, default="basic",
                        choices=["basic", "expressive"],
                        help="Type of VQC circuit")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of VQC layers")
    parser.add_argument("--n_qubits", type=int, default=9,
                        help="Number of qubits")
    parser.add_argument("--output", type=str, default="vqc_circuit.png",
                        help="Output path for PNG")
    parser.add_argument("--noise", type=str, default=None,
                        choices=[None, "bit_flip", "amp_damp", "depol"],
                        help="Noise type (optional)")
    parser.add_argument("--noise_prob", type=float, default=0.01,
                        help="Noise probability")
    
    args = parser.parse_args()
    
    draw_circuit(
        vqc_type=args.vqc_type,
        num_layers=args.num_layers,
        n_qubits=args.n_qubits,
        output_path=args.output,
        noise_type=args.noise,
        noise_prob=args.noise_prob
    )


if __name__ == "__main__":
    main()

# python /scratch90/chris/qc2025_final_proj/src/vqc_graphic.py --vqc_type basic --num_layers 2 --n_qubits 9 --noise depol --noise_prob 0.01 --output basic_noisy.png