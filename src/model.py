"""
model.py

Hybrid CNN-VQC model for image classification.

Pipelines:
1. P1 (Classical): CNN → Classifier
2. P2 (Quantum Feature Extractor): CNN → VQC → Features → Classifier
   - P2a: Probabilities (|αᵢ|²) - 512 features [no noise]
   - P2b: Complex amplitudes (Re, Im) - 1024 features [no noise]
   - P2c: Pauli expectations (⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit) - 27 features [no noise]
   - P2d: Full density matrix (ρ upper triangle) - 262,144 features [WITH noise]
3. P3 (Quantum Classifier): CNN → VQC → Measurement → Classes

Note: P2a/b/c run WITHOUT noise (pure states). P2d runs WITH noise (mixed states).
"""

import torch
import torch.nn as nn
from torchvision import models
import pennylane as qml
import numpy as np


class HybridModel(nn.Module):
    def __init__(self, num_classes, use_quantum=False, vqc_mode="feature_extractor",
                 n_qubits=9, num_layers=2, vqc_type="basic", shots=None, 
                 bit_flip_prob=0.0, amp_damp_prob=0.0, depol_prob=0.0, dropout=0.5,
                 feature_transform='probabilities'):
        """
        Args:
            num_classes (int): Number of output classes
            use_quantum (bool): If True, use quantum processing
            vqc_mode (str): How VQC is used:
                - "feature_extractor": VQC outputs features → classifier head
                - "classifier": VQC outputs class probabilities directly
            n_qubits (int): Number of qubits
            num_layers (int): Number of VQC layers
            vqc_type (str): Type of VQC circuit ("basic" or "expressive")
            shots (int): Number of shots for quantum measurements (None = statevector)
            bit_flip_prob, amp_damp_prob, depol_prob (float): Noise probabilities
            dropout (float): Dropout probability in classifier
            feature_transform (str): Feature extraction mode (only for vqc_mode="feature_extractor"):
                - 'probabilities': |αᵢ|² (P2a) - 512 features [no noise]
                - 'complex_amplitudes': [Re(α), Im(α)] (P2b) - 1024 features [no noise]
                - 'pauli': ⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit (P2c) - 27 features [no noise]
                - 'density_matrix': Full ρ upper triangle (P2d) - 262,144 features [WITH noise]
        """
        super(HybridModel, self).__init__()
        
        self.num_classes = num_classes
        self.use_quantum = use_quantum
        self.vqc_mode = vqc_mode
        self.n_qubits = n_qubits
        self.feature_dim = 2 ** n_qubits  # 512 for 9 qubits
        self.feature_transform = feature_transform
        
        # Validate noise requirements for density_matrix
        has_noise = any([bit_flip_prob > 0, amp_damp_prob > 0, depol_prob > 0])
        if feature_transform == 'density_matrix' and not has_noise:
            raise ValueError("P2d (density_matrix) requires noise! Use P2b for no-noise complex amplitudes.")
        if feature_transform in ['probabilities', 'complex_amplitudes', 'pauli'] and has_noise:
            print("⚠️  WARNING: P2a/b/c typically run without noise. Consider using P2d for noisy experiments.")
        
        # Frozen ResNet18 feature extractor
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        print(f"🧠 Initialized ResNet18 feature extractor (frozen)")
        
        # Determine classifier input dimension
        if use_quantum and vqc_mode == "feature_extractor":
            if feature_transform == 'probabilities':
                classifier_input_dim = 512  # |αᵢ|²
                print(f"⚛️  Quantum mode: Feature extractor (P2a - Probabilities)")
                print(f"   Features: 512 (probabilities)")
            elif feature_transform == 'complex_amplitudes':
                classifier_input_dim = 1024  # Re + Im
                print(f"⚛️  Quantum mode: Feature extractor (P2b - Complex Amplitudes)")
                print(f"   Features: 1024 (Re + Im)")
            elif feature_transform == 'pauli':
                classifier_input_dim = n_qubits * 3  # X, Y, Z per qubit
                print(f"⚛️  Quantum mode: Feature extractor (P2c - Pauli Expectations)")
                print(f"   Features: {classifier_input_dim} (⟨X⟩, ⟨Y⟩, ⟨Z⟩ × {n_qubits} qubits)")
            elif feature_transform == 'density_matrix':
                # Full density matrix: diagonal (512 real) + upper triangle (130,816 complex = 261,632 real)
                # Total: 262,144 real features
                dim = self.feature_dim  # 512
                classifier_input_dim = dim + 2 * (dim * (dim - 1) // 2)  # diagonal + 2×upper_triangle
                print(f"⚛️  Quantum mode: Feature extractor (P2d - Density Matrix)")
                print(f"   Features: {classifier_input_dim:,} (full ρ upper triangle)")
                print(f"   Breakdown: {dim} diagonal + {2*(dim*(dim-1)//2):,} off-diagonal (Re+Im)")
            else:
                raise ValueError(f"Unknown feature_transform: {feature_transform}")
            
            self._init_quantum(vqc_type, num_layers, shots, 
                             bit_flip_prob, amp_damp_prob, depol_prob, vqc_mode)
        
        elif use_quantum and vqc_mode == "classifier":
            classifier_input_dim = self.feature_dim
            print(f"⚛️  Quantum mode: Direct classifier (P3)")
            self._init_quantum(vqc_type, num_layers, shots, 
                             bit_flip_prob, amp_damp_prob, depol_prob, vqc_mode)
        
        else:
            classifier_input_dim = self.feature_dim
            print(f"🔷 Classical mode (P1)")
        
        # Classification head (only needed if not using VQC as classifier)
        if use_quantum and vqc_mode == "classifier":
            self.classifier = None
            print(f"📊 No classifier head (VQC outputs probabilities directly)")
        else:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
            print(f"📊 Classifier: {classifier_input_dim:,} → 256 (dropout={dropout}) → {num_classes}")
    
    def _init_quantum(self, vqc_type, num_layers, shots,
                     bit_flip_prob, amp_damp_prob, depol_prob, vqc_mode):
        """Initialize quantum circuit and weights."""
        
        # Determine if we need mixed state simulation
        use_mixed = any([bit_flip_prob > 0, amp_damp_prob > 0, depol_prob > 0])
        dev_type = "default.mixed" if use_mixed else "default.qubit"
        
        print(f"   Device: {dev_type}")
        print(f"   Qubits: {self.n_qubits}")
        print(f"   Layers: {num_layers}")
        print(f"   VQC Type: {vqc_type}")
        if use_mixed:
            print(f"   Noise: bit_flip={bit_flip_prob}, amp_damp={amp_damp_prob}, depol={depol_prob}")
        
        self.dev = qml.device(dev_type, wires=self.n_qubits, shots=shots)
        self.vqc_type = vqc_type
        self.num_layers = num_layers
        self.bit_flip_prob = bit_flip_prob
        self.amp_damp_prob = amp_damp_prob
        self.depol_prob = depol_prob
        
        # Initialize quantum weights based on circuit type
        if vqc_type == "basic":
            self.vqc_weights = nn.Parameter(
                torch.rand(num_layers, self.n_qubits) * np.pi
            )
        elif vqc_type == "expressive":
            self.single_qubit_weights = nn.Parameter(
                torch.rand(num_layers, self.n_qubits, 3) * np.pi
            )
            self.entangling_weights = nn.Parameter(
                torch.rand(num_layers, self.n_qubits, self.n_qubits) * np.pi
            )
        else:
            raise ValueError(f"Unknown vqc_type: {vqc_type}")
        
        # Create the quantum node
        self._create_qnode(vqc_mode)
    
    def _create_qnode(self, vqc_mode):
        """Create the PennyLane QNode based on mode."""
        
        def apply_noise():
            """Apply noise channels to all qubits."""
            for q in range(self.n_qubits):
                if self.bit_flip_prob > 0:
                    qml.BitFlip(self.bit_flip_prob, wires=q)
                if self.amp_damp_prob > 0:
                    qml.AmplitudeDamping(self.amp_damp_prob, wires=q)
                if self.depol_prob > 0:
                    qml.DepolarizingChannel(self.depol_prob, wires=q)
        
        if self.vqc_type == "basic":
            @qml.qnode(self.dev, interface="torch")
            def circuit(x, weights):
                # Encode features
                qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=True)
                
                # Apply VQC layers
                for layer in range(self.num_layers):
                    # Rotation layer
                    for q in range(self.n_qubits):
                        qml.RY(weights[layer, q], wires=q)
                    
                    # Entanglement layer (ring)
                    for q in range(self.n_qubits):
                        qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
                    
                    # Apply noise after each layer
                    apply_noise()
                
                # Return based on mode
                if vqc_mode == "classifier":
                    # P3: Direct classification via measurement
                    return qml.probs(wires=0)  # [P(|0⟩), P(|1⟩)]
                
                elif self.feature_transform == 'pauli':
                    # P2c: Pauli expectation values
                    # Order: [X₀, X₁, ..., X₈, Y₀, Y₁, ..., Y₈, Z₀, Z₁, ..., Z₈]
                    expvals = []
                    for q in range(self.n_qubits):
                        expvals.append(qml.expval(qml.PauliX(q)))
                    for q in range(self.n_qubits):
                        expvals.append(qml.expval(qml.PauliY(q)))
                    for q in range(self.n_qubits):
                        expvals.append(qml.expval(qml.PauliZ(q)))
                    return expvals  # [27] real values
                
                elif self.feature_transform == 'density_matrix':
                    # P2d: Return full density matrix for mixed states
                    return qml.density_matrix(wires=range(self.n_qubits))
                
                else:
                    # P2a, P2b: Return statevector (will extract probs or complex amps)
                    return qml.state()
        
        elif self.vqc_type == "expressive":
            def all_to_all_isingzz(weights):
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.IsingZZ(weights[i, j], wires=[i, j])
            
            @qml.qnode(self.dev, interface="torch")
            def circuit(x, sq_weights, ent_weights):
                # Encode features
                qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=True)
                
                # Apply VQC layers
                for layer in range(self.num_layers):
                    # Single qubit rotations
                    for q in range(self.n_qubits):
                        qml.U3(*sq_weights[layer, q], wires=q)
                    
                    # All-to-all entanglement
                    all_to_all_isingzz(ent_weights[layer])
                    
                    # Apply noise after each layer
                    apply_noise()
                
                # Return based on mode
                if vqc_mode == "classifier":
                    return qml.probs(wires=0)
                elif self.feature_transform == 'pauli':
                    expvals = []
                    for q in range(self.n_qubits):
                        expvals.append(qml.expval(qml.PauliX(q)))
                    for q in range(self.n_qubits):
                        expvals.append(qml.expval(qml.PauliY(q)))
                    for q in range(self.n_qubits):
                        expvals.append(qml.expval(qml.PauliZ(q)))
                    return expvals
                elif self.feature_transform == 'density_matrix':
                    return qml.density_matrix(wires=range(self.n_qubits))
                else:
                    return qml.state()
        
        self.qnode = circuit
    
    def quantum_feature_extraction(self, features):
        """
        Extract quantum features from classical features.
        Used when vqc_mode="feature_extractor".
        
        Returns different feature representations based on feature_transform:
        - 'probabilities': [512] - |αᵢ|²
        - 'complex_amplitudes': [1024] - [Re(α), Im(α)]
        - 'pauli': [27] - ⟨X⟩, ⟨Y⟩, ⟨Z⟩ per qubit
        - 'density_matrix': [262,144] - Full ρ upper triangle (Re + Im)
        
        Args:
            features: [batch_size, 512] classical features
        
        Returns:
            quantum_features: [batch_size, feature_dim] quantum features
        """
        batch_size = features.shape[0]
        quantum_features = []
        
        for i in range(batch_size):
            x = features[i].double()
            
            # Get quantum output from VQC
            if self.vqc_type == "basic":
                output = self.qnode(x, self.vqc_weights)
            elif self.vqc_type == "expressive":
                output = self.qnode(x, self.single_qubit_weights, 
                                   self.entangling_weights)
            
            # Process based on feature_transform
            if self.feature_transform == 'pauli':
                # P2c: Already returns expectation values [27]
                quantum_feat = torch.stack(output).float()
            
            elif self.feature_transform == 'density_matrix':
                # P2d: Extract full density matrix upper triangle + diagonal
                # output is [512, 512] complex Hermitian matrix
                
                # Extract diagonal (real values, these are probabilities)
                diagonal = torch.diag(output).real  # [512]
                
                # Extract upper triangle (excluding diagonal)
                # Use torch.triu to get upper triangle with offset=1
                upper_triangle_mask = torch.triu(torch.ones_like(output.real), diagonal=1).bool()
                upper_triangle = output[upper_triangle_mask]  # [130816] complex values
                
                # Separate real and imaginary parts
                upper_real = upper_triangle.real  # [130816]
                upper_imag = upper_triangle.imag  # [130816]
                
                # Concatenate: [diagonal, upper_real, upper_imag]
                quantum_feat = torch.cat([diagonal, upper_real, upper_imag]).float()
                # Total: 512 + 130816 + 130816 = 262,144 features
            
            elif self.feature_transform == 'complex_amplitudes':
                # P2b: Extract real and imaginary parts
                if torch.is_complex(output):
                    real_part = output.real  # [512]
                    imag_part = output.imag  # [512]
                    quantum_feat = torch.cat([real_part, imag_part]).float()  # [1024]
                else:
                    # If somehow we got real output, just duplicate it
                    quantum_feat = torch.cat([output, torch.zeros_like(output)]).float()
            
            else:  # 'probabilities'
                # P2a: Extract probabilities |αᵢ|²
                if torch.is_complex(output):
                    quantum_feat = torch.abs(output).pow(2).float()  # [512]
                else:
                    quantum_feat = output.float()  # Already probabilities
            
            quantum_features.append(quantum_feat)
        
        return torch.stack(quantum_features)
    
    def quantum_classification(self, features):
        """
        Use VQC as a direct classifier.
        Used when vqc_mode="classifier" (P3).
        
        Args:
            features: [batch_size, 512] classical features
        
        Returns:
            class_probs: [batch_size, 2] class probabilities
        """
        batch_size = features.shape[0]
        probs = []
        
        for i in range(batch_size):
            x = features[i].double()
            
            # Get class probabilities from VQC
            if self.vqc_type == "basic":
                prob = self.qnode(x, self.vqc_weights)  # [2] for binary
            elif self.vqc_type == "expressive":
                prob = self.qnode(x, self.single_qubit_weights, 
                                self.entangling_weights)
            
            probs.append(prob.float())
        
        return torch.stack(probs)  # [batch, 2]
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: [batch_size, 3, H, W] input images
        
        Returns:
            logits: [batch_size, num_classes] classification logits
        """
        # Extract CNN features
        features = self.feature_extractor(x)  # [batch_size, 512]
        
        # Process based on mode
        if self.use_quantum:
            if self.vqc_mode == "classifier":
                # P3: VQC directly outputs class probabilities
                probs = self.quantum_classification(features)  # [batch, 2]
                logits = torch.log(probs + 1e-8)  # Convert to log-probs
                return logits
            
            elif self.vqc_mode == "feature_extractor":
                # P2a/b/c/d: VQC outputs features, then classify
                features = self.quantum_feature_extraction(features)
                logits = self.classifier(features)
                return logits
        
        else:
            # P1: Classical - direct classification
            logits = self.classifier(features)
            return logits


def test():
    """
    Test all 6 pipelines: P1, P2a, P2b, P2c, P2d, P3
    """
    print("\n" + "="*80)
    print("TESTING ALL PIPELINES (INCLUDING P2d - DENSITY MATRIX)")
    print("="*80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}\n")
    
    # Test parameters
    batch_size = 3
    num_classes = 2
    img_size = 224
    n_qubits = 9
    
    # Create dummy batch
    dummy_images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    print(f"📥 Input images shape: {dummy_images.shape}\n")
    
    def print_feature_stats(features, name):
        """Helper to print feature statistics"""
        print(f"\n   {name}:")
        print(f"      Shape: {features.shape}")
        if features.shape[1] <= 10:
            print(f"      Sample 1: {features[0]}")
        else:
            print(f"      Sample 1 (first 5): {features[0, :5]}")
        print(f"      Mean per image: {features.mean(dim=1)}")
        print(f"      Std per image:  {features.std(dim=1)}")
        print(f"      Range: [{features.min():.4f}, {features.max():.4f}]")
    
    # ═══════════════════════════════════════════════════════════════
    # P1: CLASSICAL
    # ═══════════════════════════════════════════════════════════════
    print("="*80)
    print("P1: CLASSICAL")
    print("="*80)
    model_p1 = HybridModel(
        num_classes=num_classes,
        use_quantum=False,
        n_qubits=n_qubits,
        dropout=0.0
    ).to(device)
    
    print(f"\n🔄 Running forward pass...")
    with torch.no_grad():
        cnn_features = model_p1.feature_extractor(dummy_images)
        print_feature_stats(cnn_features, "CNN Features")
        
        logits = model_p1(dummy_images)
        print(f"\n   Logits: {logits}")
        print(f"   Predictions: {torch.argmax(logits, dim=1)}")
    
    # ═══════════════════════════════════════════════════════════════
    # P2a: QUANTUM FEATURE EXTRACTOR (PROBABILITIES)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("P2a: QUANTUM FEATURE EXTRACTOR (PROBABILITIES)")
    print("="*80)
    model_p2a = HybridModel(
        num_classes=num_classes,
        use_quantum=True,
        vqc_mode="feature_extractor",
        n_qubits=n_qubits,
        num_layers=2,
        vqc_type="basic",
        dropout=0.0,
        feature_transform='probabilities'
    ).to(device)
    
    print(f"\n🔄 Running forward pass...")
    with torch.no_grad():
        quantum_features = model_p2a.quantum_feature_extraction(model_p2a.feature_extractor(dummy_images))
        print_feature_stats(quantum_features, "Quantum Features (Probabilities)")
        
        logits = model_p2a(dummy_images)
        print(f"\n   Logits: {logits}")
        print(f"   Predictions: {torch.argmax(logits, dim=1)}")
    
    # ═══════════════════════════════════════════════════════════════
    # P2b: QUANTUM FEATURE EXTRACTOR (COMPLEX AMPLITUDES)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("P2b: QUANTUM FEATURE EXTRACTOR (COMPLEX AMPLITUDES)")
    print("="*80)
    model_p2b = HybridModel(
        num_classes=num_classes,
        use_quantum=True,
        vqc_mode="feature_extractor",
        n_qubits=n_qubits,
        num_layers=2,
        vqc_type="basic",
        dropout=0.0,
        feature_transform='complex_amplitudes'
    ).to(device)
    
    print(f"\n🔄 Running forward pass...")
    with torch.no_grad():
        quantum_features = model_p2b.quantum_feature_extraction(model_p2b.feature_extractor(dummy_images))
        print_feature_stats(quantum_features, "Quantum Features (Complex Amplitudes)")
        print(f"      First 512 = Re(α), Last 512 = Im(α)")
        
        logits = model_p2b(dummy_images)
        print(f"\n   Logits: {logits}")
        print(f"   Predictions: {torch.argmax(logits, dim=1)}")
    
    # ═══════════════════════════════════════════════════════════════
    # P2c: QUANTUM FEATURE EXTRACTOR (PAULI EXPECTATIONS)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("P2c: QUANTUM FEATURE EXTRACTOR (PAULI EXPECTATIONS)")
    print("="*80)
    model_p2c = HybridModel(
        num_classes=num_classes,
        use_quantum=True,
        vqc_mode="feature_extractor",
        n_qubits=n_qubits,
        num_layers=2,
        vqc_type="basic",
        dropout=0.0,
        feature_transform='pauli'
    ).to(device)
    
    print(f"\n🔄 Running forward pass...")
    with torch.no_grad():
        quantum_features = model_p2c.quantum_feature_extraction(model_p2c.feature_extractor(dummy_images))
        print_feature_stats(quantum_features, "Quantum Features (Pauli Expectations)")
        print(f"      Format: [X₀...X₈, Y₀...Y₈, Z₀...Z₈]")
        
        logits = model_p2c(dummy_images)
        print(f"\n   Logits: {logits}")
        print(f"   Predictions: {torch.argmax(logits, dim=1)}")
    
    # ═══════════════════════════════════════════════════════════════
    # P2d: QUANTUM FEATURE EXTRACTOR (DENSITY MATRIX - WITH NOISE)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("P2d: QUANTUM FEATURE EXTRACTOR (DENSITY MATRIX - WITH NOISE)")
    print("="*80)
    model_p2d = HybridModel(
        num_classes=num_classes,
        use_quantum=True,
        vqc_mode="feature_extractor",
        n_qubits=n_qubits,
        num_layers=2,
        vqc_type="basic",
        dropout=0.0,
        feature_transform='density_matrix',
        depol_prob=0.01  # Add noise to test density matrix
    ).to(device)
    
    print(f"\n🔄 Running forward pass...")
    with torch.no_grad():
        quantum_features = model_p2d.quantum_feature_extraction(model_p2d.feature_extractor(dummy_images))
        print_feature_stats(quantum_features, "Quantum Features (Density Matrix)")
        print(f"      Format: [diagonal(512), upper_real(130816), upper_imag(130816)]")
        print(f"      Total: 262,144 features from full ρ upper triangle")
        
        logits = model_p2d(dummy_images)
        print(f"\n   Logits: {logits}")
        print(f"   Predictions: {torch.argmax(logits, dim=1)}")
    
    # ═══════════════════════════════════════════════════════════════
    # P3: QUANTUM CLASSIFIER
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("P3: QUANTUM CLASSIFIER")
    print("="*80)
    model_p3 = HybridModel(
        num_classes=num_classes,
        use_quantum=True,
        vqc_mode="classifier",
        n_qubits=n_qubits,
        num_layers=2,
        vqc_type="basic"
    ).to(device)
    
    print(f"\n🔄 Running forward pass...")
    with torch.no_grad():
        probs = model_p3.quantum_classification(model_p3.feature_extractor(dummy_images))
        print(f"\n   VQC Probabilities:")
        print(f"      {probs}")
        
        logits = model_p3(dummy_images)
        print(f"\n   Logits (log probs): {logits}")
        print(f"   Predictions: {torch.argmax(logits, dim=1)}")
    
    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ P1 (Classical):                       512 features → Classifier")
    print(f"✅ P2a (Probabilities):                  512 features → Classifier")
    print(f"✅ P2b (Complex Amplitudes):           1,024 features → Classifier")
    print(f"✅ P2c (Pauli Expectations):              27 features → Classifier")
    print(f"✅ P2d (Density Matrix):             262,144 features → Classifier")
    print(f"✅ P3 (Quantum Classifier):                2 outputs (direct)")
    print("="*80 + "\n")


if __name__ == "__main__":
    test()