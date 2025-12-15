#!/bin/bash

# Flexible experimental suite launcher with command-line options
# Can run all pipelines or select specific groups
# NOW SUPPORTS MULTIPLE DATASETS: retinamnist, breastmnist, pneumoniamnist
# NOW WITH DUAL LEARNING RATES for P2 pipelines!
#
# Pipelines:
# - P1: Classical + Dropout (lr=0.001)
# - P2a/b/c: Quantum Features, no noise (lr_quantum=0.1, lr_classical=0.001)
# - P2d: Density Matrix Features, with noise (lr_quantum=0.1, lr_classical=0.001)
# - P3: Quantum Classifier + Noise (lr=0.1)

# === PARSE ARGUMENTS ===
RUN_P1=false
RUN_P2=false
RUN_P3=false
RUN_ALL=false
DATASET="retinamnist"  # Default dataset

while [[ $# -gt 0 ]]; do
    case $1 in
        --p1)
            RUN_P1=true
            shift
            ;;
        --p2)
            RUN_P2=true
            shift
            ;;
        --p3)
            RUN_P3=true
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --p1              Run Pipeline 1 (Classical + Dropout)"
            echo "  --p2              Run Pipeline 2 (Quantum Feature Extractor variants)"
            echo "                    • P2a/b/c: No noise (DUAL LR: quantum=0.1, classical=0.001)"
            echo "                    • P2d: Density matrix with noise (DUAL LR)"
            echo "  --p3              Run Pipeline 3 (Quantum Classifier + Noise, lr=0.1)"
            echo "  --all             Run all pipelines"
            echo "  --dataset NAME    Choose dataset: retinamnist, breastmnist, or pneumoniamnist"
            echo "                    (default: retinamnist)"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --p2 --dataset retinamnist       # Run P2 on RetinaMNIST"
            echo "  $0 --p2 --dataset breastmnist       # Run P2 on BreastMNIST"
            echo "  $0 --p2 --dataset pneumoniamnist    # Run P2 on PneumoniaMNIST"
            echo "  $0 --all --dataset retinamnist      # Run everything on RetinaMNIST"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate dataset choice
if [[ "$DATASET" != "retinamnist" && "$DATASET" != "breastmnist" && "$DATASET" != "pneumoniamnist" ]]; then
    echo "❌ Error: Invalid dataset '$DATASET'"
    echo "   Valid options: retinamnist, breastmnist, pneumoniamnist"
    exit 1
fi

# If --all is set, enable all pipelines
if [ "$RUN_ALL" = true ]; then
    RUN_P1=true
    RUN_P2=true
    RUN_P3=true
fi

# If nothing selected, show error
if [ "$RUN_P1" = false ] && [ "$RUN_P2" = false ] && [ "$RUN_P3" = false ]; then
    echo "❌ Error: No pipelines selected!"
    echo ""
    echo "Please specify which pipelines to run:"
    echo "  --p1       Pipeline 1 (Classical)"
    echo "  --p2       Pipeline 2 (Quantum Features with DUAL LR)"
    echo "  --p3       Pipeline 3 (Quantum Classifier)"
    echo "  --all      All pipelines"
    echo ""
    echo "Use --help for more information"
    exit 1
fi

# === CONFIG ===
BASE_DIR="/scratch90/chris/qc2025_final_proj/results_${DATASET}_n10_dual_lr"
SCRIPT_PATH="/scratch90/chris/qc2025_final_proj/src/train.py"

# Dataset-specific training size
# BreastMNIST only has ~294 balanced samples, so use all of them
if [ "$DATASET" = "breastmnist" ]; then
    N_TRAIN=-1  # Use all available samples
else
    N_TRAIN=1000
fi

BATCH_SIZE=4
N_EPOCHS=25
LEARNING_RATE_CLASSICAL=0.001
LEARNING_RATE_QUANTUM=0.1
PATIENCE=5
N_QUBITS=9
NUM_LAYERS=2
VQC_TYPE="basic"

# Experiment parameters
DROPOUT_LEVELS=(0.0 0.1 0.2 0.3 0.4 0.5 0.9)
QUANTUM_NOISE_LEVELS=(0.01 0.03 0.05 0.1 0.2 0.3)
N_RUNS=10  # Number of runs per configuration

# Create base results directory
mkdir -p "$BASE_DIR"

# === DISPLAY LAUNCH PLAN ===
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   QUANTUM-CLASSICAL HYBRID EXPERIMENTS (DUAL LR)          ║"
echo "║                    N_RUNS = $N_RUNS                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Results directory: $BASE_DIR"
echo "📊 Dataset: ${DATASET^^}"
if [ "$DATASET" = "breastmnist" ]; then
    echo "   ⚠️  BreastMNIST: Using all ~294 samples (small dataset)"
else
    echo "   Training samples: $N_TRAIN per experiment"
fi
echo ""
echo "⚙️  Learning Rates:"
echo "   P1 (Classical):           lr = $LEARNING_RATE_CLASSICAL"
echo "   P2 (Quantum Features):    lr_quantum = $LEARNING_RATE_QUANTUM, lr_classical = $LEARNING_RATE_CLASSICAL ✨"
echo "   P3 (Quantum Classifier):  lr = $LEARNING_RATE_QUANTUM"
echo ""
echo "🎯 Selected pipelines:"
[ "$RUN_P1" = true ] && echo "   ✅ P1: Classical + Dropout ($(( 7 * N_RUNS )) experiments)"
[ "$RUN_P2" = true ] && echo "   ✅ P2: Quantum Feature Extractor ($(( 3 * N_RUNS + 3 * 6 * N_RUNS )) experiments)"
[ "$RUN_P2" = true ] && echo "      • P2a/b/c: No noise ($(( 3 * N_RUNS )) experiments)"
[ "$RUN_P2" = true ] && echo "      • P2d: Density matrix + noise ($(( 3 * 6 * N_RUNS )) experiments)"
[ "$RUN_P3" = true ] && echo "   ✅ P3: Quantum Classifier + Noise ($(( (1 + 3 * 6) * N_RUNS )) experiments)"
echo ""

# Count total experiments
TOTAL_EXPERIMENTS=0
TOTAL_STREAMS=0
[ "$RUN_P1" = true ] && ((TOTAL_EXPERIMENTS+=$(( 7 * N_RUNS )))) && ((TOTAL_STREAMS+=1))
[ "$RUN_P2" = true ] && ((TOTAL_EXPERIMENTS+=$(( 3 * N_RUNS + 3 * 6 * N_RUNS )))) && ((TOTAL_STREAMS+=6))
[ "$RUN_P3" = true ] && ((TOTAL_EXPERIMENTS+=$(( (1 + 3 * 6) * N_RUNS )))) && ((TOTAL_STREAMS+=4))

echo "📊 Total: $TOTAL_EXPERIMENTS experiments in $TOTAL_STREAMS parallel streams"
echo ""
echo "⏳ Launching in 3 seconds..."
sleep 3

# ═══════════════════════════════════════════════════════════════
# STREAM 1: PIPELINE 1 - CLASSICAL
# ═══════════════════════════════════════════════════════════════

if [ "$RUN_P1" = true ]; then
    screen -dmS p1_classical_${DATASET} bash -c '
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         PIPELINE 1: Classical + Dropout                   ║"
    echo "║              Dataset: '"${DATASET^^}"'                             ║"
    echo "║              7 dropout × '"$N_RUNS"' runs = '"$(( 7 * N_RUNS ))"'                    ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    total=0
    for dropout in 0.0 0.1 0.2 0.3 0.4 0.5 0.9; do
        for run in $(seq 1 '"$N_RUNS"'); do
            ((total++))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🔷 P1 Classical [$total/'"$(( 7 * N_RUNS ))"']: dropout=$dropout, run=$run"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            python '"$SCRIPT_PATH"' \
                --dataset '"$DATASET"' \
                --binary \
                --balance_classes \
                --dropout $dropout \
                --save_dir "'"$BASE_DIR"'/p1_classical_dropout_${dropout}/run_${run}" \
                --n_train '"$N_TRAIN"' \
                --batch_size '"$BATCH_SIZE"' \
                --n_epochs '"$N_EPOCHS"' \
                --learning_rate '"$LEARNING_RATE_CLASSICAL"' \
                --patience '"$PATIENCE"' \
                --run_id $run
        done
    done
    
    echo ""
    echo "✅ PIPELINE 1 COMPLETE!"
    exec bash
    '
    
    echo "✅ Launched: p1_classical_${DATASET}"
fi

# ═══════════════════════════════════════════════════════════════
# STREAM 2: PIPELINE 2A - PROBABILITIES (NO NOISE) - DUAL LR
# ═══════════════════════════════════════════════════════════════

if [ "$RUN_P2" = true ]; then
    screen -dmS p2a_prob_${DATASET} bash -c '
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  PIPELINE 2A: Quantum Features (Probabilities) DUAL LR    ║"
    echo "║              Dataset: '"${DATASET^^}"'                             ║"
    echo "║              lr_quantum=0.1, lr_classical=0.001           ║"
    echo "║                      '"$N_RUNS"' runs                               ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    for run in $(seq 1 '"$N_RUNS"'); do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "⚛️  P2A (Probabilities) [$run/'"$N_RUNS"']"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python '"$SCRIPT_PATH"' \
            --dataset '"$DATASET"' \
            --binary \
            --balance_classes \
            --use_quantum \
            --vqc_mode feature_extractor \
            --feature_transform probabilities \
            --dropout 0.0 \
            --n_qubits '"$N_QUBITS"' \
            --num_layers '"$NUM_LAYERS"' \
            --vqc_type '"$VQC_TYPE"' \
            --save_dir "'"$BASE_DIR"'/p2a_probabilities/run_${run}" \
            --n_train '"$N_TRAIN"' \
            --batch_size '"$BATCH_SIZE"' \
            --n_epochs '"$N_EPOCHS"' \
            --learning_rate_quantum '"$LEARNING_RATE_QUANTUM"' \
            --learning_rate_classical '"$LEARNING_RATE_CLASSICAL"' \
            --patience '"$PATIENCE"' \
            --run_id $run
    done
    
    echo ""
    echo "✅ P2A (PROBABILITIES) COMPLETE!"
    exec bash
    '
    
    echo "✅ Launched: p2a_prob_${DATASET} (DUAL LR)"
fi

# ═══════════════════════════════════════════════════════════════
# STREAM 3: PIPELINE 2B - COMPLEX AMPLITUDES (NO NOISE) - DUAL LR
# ═══════════════════════════════════════════════════════════════

if [ "$RUN_P2" = true ]; then
    screen -dmS p2b_complex_${DATASET} bash -c '
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ PIPELINE 2B: Quantum Features (Complex Amplitudes) DUAL LR║"
    echo "║              Dataset: '"${DATASET^^}"'                             ║"
    echo "║              lr_quantum=0.1, lr_classical=0.001           ║"
    echo "║                      '"$N_RUNS"' runs                               ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    for run in $(seq 1 '"$N_RUNS"'); do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "⚛️  P2B (Complex Amplitudes) [$run/'"$N_RUNS"']"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python '"$SCRIPT_PATH"' \
            --dataset '"$DATASET"' \
            --binary \
            --balance_classes \
            --use_quantum \
            --vqc_mode feature_extractor \
            --feature_transform complex_amplitudes \
            --dropout 0.0 \
            --n_qubits '"$N_QUBITS"' \
            --num_layers '"$NUM_LAYERS"' \
            --vqc_type '"$VQC_TYPE"' \
            --save_dir "'"$BASE_DIR"'/p2b_complex_amplitudes/run_${run}" \
            --n_train '"$N_TRAIN"' \
            --batch_size '"$BATCH_SIZE"' \
            --n_epochs '"$N_EPOCHS"' \
            --learning_rate_quantum '"$LEARNING_RATE_QUANTUM"' \
            --learning_rate_classical '"$LEARNING_RATE_CLASSICAL"' \
            --patience '"$PATIENCE"' \
            --run_id $run
    done
    
    echo ""
    echo "✅ P2B (COMPLEX AMPLITUDES) COMPLETE!"
    exec bash
    '
    
    echo "✅ Launched: p2b_complex_${DATASET} (DUAL LR)"
fi

# ═══════════════════════════════════════════════════════════════
# STREAM 4: PIPELINE 2C - PAULI EXPECTATIONS (NO NOISE) - DUAL LR
# ═══════════════════════════════════════════════════════════════

if [ "$RUN_P2" = true ]; then
    screen -dmS p2c_pauli_${DATASET} bash -c '
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ PIPELINE 2C: Quantum Features (Pauli Expectations) DUAL LR║"
    echo "║              Dataset: '"${DATASET^^}"'                             ║"
    echo "║              lr_quantum=0.1, lr_classical=0.001           ║"
    echo "║                      '"$N_RUNS"' runs                               ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    for run in $(seq 1 '"$N_RUNS"'); do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "⚛️  P2C (Pauli Expectations) [$run/'"$N_RUNS"']"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python '"$SCRIPT_PATH"' \
            --dataset '"$DATASET"' \
            --binary \
            --balance_classes \
            --use_quantum \
            --vqc_mode feature_extractor \
            --feature_transform pauli \
            --dropout 0.0 \
            --n_qubits '"$N_QUBITS"' \
            --num_layers '"$NUM_LAYERS"' \
            --vqc_type '"$VQC_TYPE"' \
            --save_dir "'"$BASE_DIR"'/p2c_pauli_expectations/run_${run}" \
            --n_train '"$N_TRAIN"' \
            --batch_size '"$BATCH_SIZE"' \
            --n_epochs '"$N_EPOCHS"' \
            --learning_rate_quantum '"$LEARNING_RATE_QUANTUM"' \
            --learning_rate_classical '"$LEARNING_RATE_CLASSICAL"' \
            --patience '"$PATIENCE"' \
            --run_id $run
    done
    
    echo ""
    echo "✅ P2C (PAULI EXPECTATIONS) COMPLETE!"
    exec bash
    '
    
    echo "✅ Launched: p2c_pauli_${DATASET} (DUAL LR)"
fi

# ═══════════════════════════════════════════════════════════════
# STREAM 5: PIPELINE 2D - DENSITY MATRIX (BIT FLIP) - DUAL LR
# ═══════════════════════════════════════════════════════════════

if [ "$RUN_P2" = true ]; then
    screen -dmS p2d_bitflip_${DATASET} bash -c '
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  PIPELINE 2D: Density Matrix Features (Bit Flip) DUAL LR  ║"
    echo "║              Dataset: '"${DATASET^^}"'                             ║"
    echo "║              lr_quantum=0.1, lr_classical=0.001           ║"
    echo "║            6 noise levels × '"$N_RUNS"' runs = '"$(( 6 * N_RUNS ))"'                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    total=0
    for noise in 0.01 0.03 0.05 0.1 0.2 0.3; do
        for run in $(seq 1 '"$N_RUNS"'); do
            ((total++))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "⚛️  P2D (Density Matrix, Bit Flip) [$total/'"$(( 6 * N_RUNS ))"']: p=$noise, run=$run"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            python '"$SCRIPT_PATH"' \
                --dataset '"$DATASET"' \
                --binary \
                --balance_classes \
                --use_quantum \
                --vqc_mode feature_extractor \
                --feature_transform density_matrix \
                --dropout 0.0 \
                --n_qubits '"$N_QUBITS"' \
                --num_layers '"$NUM_LAYERS"' \
                --vqc_type '"$VQC_TYPE"' \
                --bit_flip_prob $noise \
                --save_dir "'"$BASE_DIR"'/p2d_density_matrix_bit_flip_${noise}/run_${run}" \
                --n_train '"$N_TRAIN"' \
                --batch_size '"$BATCH_SIZE"' \
                --n_epochs '"$N_EPOCHS"' \
                --learning_rate_quantum '"$LEARNING_RATE_QUANTUM"' \
                --learning_rate_classical '"$LEARNING_RATE_CLASSICAL"' \
                --patience '"$PATIENCE"' \
                --run_id $run
        done
    done
    
    echo ""
    echo "✅ P2D (BIT FLIP) COMPLETE!"
    exec bash
    '
    
    echo "✅ Launched: p2d_bitflip_${DATASET} (DUAL LR)"
fi

# ═══════════════════════════════════════════════════════════════
# STREAM 6: PIPELINE 2D - DENSITY MATRIX (AMP DAMP) - DUAL LR
# ═══════════════════════════════════════════════════════════════

if [ "$RUN_P2" = true ]; then
    screen -dmS p2d_ampdamp_${DATASET} bash -c '
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ PIPELINE 2D: Density Matrix Features (Amp Damping) DUAL LR║"
    echo "║              Dataset: '"${DATASET^^}"'                             ║"
    echo "║              lr_quantum=0.1, lr_classical=0.001           ║"
    echo "║            6 noise levels × '"$N_RUNS"' runs = '"$(( 6 * N_RUNS ))"'                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    total=0
    for noise in 0.01 0.03 0.05 0.1 0.2 0.3; do
        for run in $(seq 1 '"$N_RUNS"'); do
            ((total++))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "⚛️  P2D (Density Matrix, Amp Damp) [$total/'"$(( 6 * N_RUNS ))"']: p=$noise, run=$run"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            python '"$SCRIPT_PATH"' \
                --dataset '"$DATASET"' \
                --binary \
                --balance_classes \
                --use_quantum \
                --vqc_mode feature_extractor \
                --feature_transform density_matrix \
                --dropout 0.0 \
                --n_qubits '"$N_QUBITS"' \
                --num_layers '"$NUM_LAYERS"' \
                --vqc_type '"$VQC_TYPE"' \
                --amp_damp_prob $noise \
                --save_dir "'"$BASE_DIR"'/p2d_density_matrix_amp_damp_${noise}/run_${run}" \
                --n_train '"$N_TRAIN"' \
                --batch_size '"$BATCH_SIZE"' \
                --n_epochs '"$N_EPOCHS"' \
                --learning_rate_quantum '"$LEARNING_RATE_QUANTUM"' \
                --learning_rate_classical '"$LEARNING_RATE_CLASSICAL"' \
                --patience '"$PATIENCE"' \
                --run_id $run
        done
    done
    
    echo ""
    echo "✅ P2D (AMP DAMP) COMPLETE!"
    exec bash
    '
    
    echo "✅ Launched: p2d_ampdamp_${DATASET} (DUAL LR)"
fi

# ═══════════════════════════════════════════════════════════════
# STREAM 7: PIPELINE 2D - DENSITY MATRIX (DEPOLARIZING) - DUAL LR
# ═══════════════════════════════════════════════════════════════

if [ "$RUN_P2" = true ]; then
    screen -dmS p2d_depol_${DATASET} bash -c '
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ PIPELINE 2D: Density Matrix Features (Depolarizing) DUAL LR║"
    echo "║              Dataset: '"${DATASET^^}"'                             ║"
    echo "║              lr_quantum=0.1, lr_classical=0.001           ║"
    echo "║            6 noise levels × '"$N_RUNS"' runs = '"$(( 6 * N_RUNS ))"'                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    total=0
    for noise in 0.01 0.03 0.05 0.1 0.2 0.3; do
        for run in $(seq 1 '"$N_RUNS"'); do
            ((total++))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "⚛️  P2D (Density Matrix, Depol) [$total/'"$(( 6 * N_RUNS ))"']: p=$noise, run=$run"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            python '"$SCRIPT_PATH"' \
                --dataset '"$DATASET"' \
                --binary \
                --balance_classes \
                --use_quantum \
                --vqc_mode feature_extractor \
                --feature_transform density_matrix \
                --dropout 0.0 \
                --n_qubits '"$N_QUBITS"' \
                --num_layers '"$NUM_LAYERS"' \
                --vqc_type '"$VQC_TYPE"' \
                --depol_prob $noise \
                --save_dir "'"$BASE_DIR"'/p2d_density_matrix_depol_${noise}/run_${run}" \
                --n_train '"$N_TRAIN"' \
                --batch_size '"$BATCH_SIZE"' \
                --n_epochs '"$N_EPOCHS"' \
                --learning_rate_quantum '"$LEARNING_RATE_QUANTUM"' \
                --learning_rate_classical '"$LEARNING_RATE_CLASSICAL"' \
                --patience '"$PATIENCE"' \
                --run_id $run
        done
    done
    
    echo ""
    echo "✅ P2D (DEPOL) COMPLETE!"
    exec bash
    '
    
    echo "✅ Launched: p2d_depol_${DATASET} (DUAL LR)"
fi

# ═══════════════════════════════════════════════════════════════
# STREAM 8: PIPELINE 3 - CLASSIFIER (NO NOISE)
# ═══════════════════════════════════════════════════════════════

if [ "$RUN_P3" = true ]; then
    screen -dmS p3_no_noise_${DATASET} bash -c '
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║       PIPELINE 3: Quantum Classifier (No Noise)           ║"
    echo "║              Dataset: '"${DATASET^^}"'                             ║"
    echo "║              lr=0.1 (quantum only)                        ║"
    echo "║                      '"$N_RUNS"' runs                               ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    for run in $(seq 1 '"$N_RUNS"'); do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "⚛️  P3 Classifier (No Noise) [$run/'"$N_RUNS"']"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        python '"$SCRIPT_PATH"' \
            --dataset '"$DATASET"' \
            --binary \
            --balance_classes \
            --use_quantum \
            --vqc_mode classifier \
            --n_qubits '"$N_QUBITS"' \
            --num_layers '"$NUM_LAYERS"' \
            --vqc_type '"$VQC_TYPE"' \
            --save_dir "'"$BASE_DIR"'/p3_no_noise/run_${run}" \
            --n_train '"$N_TRAIN"' \
            --batch_size '"$BATCH_SIZE"' \
            --n_epochs '"$N_EPOCHS"' \
            --learning_rate_quantum '"$LEARNING_RATE_QUANTUM"' \
            --patience '"$PATIENCE"' \
            --run_id $run
    done
    
    echo ""
    echo "✅ P3 (NO NOISE) COMPLETE!"
    exec bash
    '
    
    echo "✅ Launched: p3_no_noise_${DATASET}"
fi

# ═══════════════════════════════════════════════════════════════
# STREAM 9: PIPELINE 3 - CLASSIFIER (BIT FLIP)
# ═══════════════════════════════════════════════════════════════

if [ "$RUN_P3" = true ]; then
    screen -dmS p3_bitflip_${DATASET} bash -c '
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║       PIPELINE 3: Quantum Classifier (Bit Flip)           ║"
    echo "║              Dataset: '"${DATASET^^}"'                             ║"
    echo "║            6 noise levels × '"$N_RUNS"' runs = '"$(( 6 * N_RUNS ))"'                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    total=0
    for noise in 0.01 0.03 0.05 0.1 0.2 0.3; do
        for run in $(seq 1 '"$N_RUNS"'); do
            ((total++))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "⚛️  P3 Classifier (Bit Flip) [$total/'"$(( 6 * N_RUNS ))"']: p=$noise, run=$run"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            python '"$SCRIPT_PATH"' \
                --dataset '"$DATASET"' \
                --binary \
                --balance_classes \
                --use_quantum \
                --vqc_mode classifier \
                --n_qubits '"$N_QUBITS"' \
                --num_layers '"$NUM_LAYERS"' \
                --vqc_type '"$VQC_TYPE"' \
                --bit_flip_prob $noise \
                --save_dir "'"$BASE_DIR"'/p3_bit_flip_${noise}/run_${run}" \
                --n_train '"$N_TRAIN"' \
                --batch_size '"$BATCH_SIZE"' \
                --n_epochs '"$N_EPOCHS"' \
                --learning_rate_quantum '"$LEARNING_RATE_QUANTUM"' \
                --patience '"$PATIENCE"' \
                --run_id $run
        done
    done
    
    echo ""
    echo "✅ P3 (BIT FLIP) COMPLETE!"
    exec bash
    '
    
    echo "✅ Launched: p3_bitflip_${DATASET}"
fi

# ═══════════════════════════════════════════════════════════════
# STREAM 10: PIPELINE 3 - CLASSIFIER (AMP DAMP)
# ═══════════════════════════════════════════════════════════════

if [ "$RUN_P3" = true ]; then
    screen -dmS p3_ampdamp_${DATASET} bash -c '
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     PIPELINE 3: Quantum Classifier (Amp Damping)          ║"
    echo "║              Dataset: '"${DATASET^^}"'                             ║"
    echo "║            6 noise levels × '"$N_RUNS"' runs = '"$(( 6 * N_RUNS ))"'                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    total=0
    for noise in 0.01 0.03 0.05 0.1 0.2 0.3; do
        for run in $(seq 1 '"$N_RUNS"'); do
            ((total++))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "⚛️  P3 Classifier (Amp Damp) [$total/'"$(( 6 * N_RUNS ))"']: p=$noise, run=$run"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            python '"$SCRIPT_PATH"' \
                --dataset '"$DATASET"' \
                --binary \
                --balance_classes \
                --use_quantum \
                --vqc_mode classifier \
                --n_qubits '"$N_QUBITS"' \
                --num_layers '"$NUM_LAYERS"' \
                --vqc_type '"$VQC_TYPE"' \
                --amp_damp_prob $noise \
                --save_dir "'"$BASE_DIR"'/p3_amp_damp_${noise}/run_${run}" \
                --n_train '"$N_TRAIN"' \
                --batch_size '"$BATCH_SIZE"' \
                --n_epochs '"$N_EPOCHS"' \
                --learning_rate_quantum '"$LEARNING_RATE_QUANTUM"' \
                --patience '"$PATIENCE"' \
                --run_id $run
        done
    done
    
    echo ""
    echo "✅ P3 (AMP DAMP) COMPLETE!"
    exec bash
    '
    
    echo "✅ Launched: p3_ampdamp_${DATASET}"
fi

# ═══════════════════════════════════════════════════════════════
# STREAM 11: PIPELINE 3 - CLASSIFIER (DEPOLARIZING)
# ═══════════════════════════════════════════════════════════════

if [ "$RUN_P3" = true ]; then
    screen -dmS p3_depol_${DATASET} bash -c '
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     PIPELINE 3: Quantum Classifier (Depolarizing)         ║"
    echo "║              Dataset: '"${DATASET^^}"'                             ║"
    echo "║            6 noise levels × '"$N_RUNS"' runs = '"$(( 6 * N_RUNS ))"'                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    total=0
    for noise in 0.01 0.03 0.05 0.1 0.2 0.3; do
        for run in $(seq 1 '"$N_RUNS"'); do
            ((total++))
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "⚛️  P3 Classifier (Depol) [$total/'"$(( 6 * N_RUNS ))"']: p=$noise, run=$run"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            python '"$SCRIPT_PATH"' \
                --dataset '"$DATASET"' \
                --binary \
                --balance_classes \
                --use_quantum \
                --vqc_mode classifier \
                --n_qubits '"$N_QUBITS"' \
                --num_layers '"$NUM_LAYERS"' \
                --vqc_type '"$VQC_TYPE"' \
                --depol_prob $noise \
                --save_dir "'"$BASE_DIR"'/p3_depol_${noise}/run_${run}" \
                --n_train '"$N_TRAIN"' \
                --batch_size '"$BATCH_SIZE"' \
                --n_epochs '"$N_EPOCHS"' \
                --learning_rate_quantum '"$LEARNING_RATE_QUANTUM"' \
                --patience '"$PATIENCE"' \
                --run_id $run
        done
    done
    
    echo ""
    echo "✅ P3 (DEPOL) COMPLETE!"
    exec bash
    '
    
    echo "✅ Launched: p3_depol_${DATASET}"
fi

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🚀 All selected streams launched (DUAL LR)!              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Dataset: ${DATASET^^}"
echo ""
echo "🔍 Monitor sessions:"
echo "   screen -ls                                   # List all sessions"
[ "$RUN_P1" = true ] && echo "   screen -r p1_classical_${DATASET}            # Attach to P1"
[ "$RUN_P2" = true ] && echo "   screen -r p2a_prob_${DATASET}                # Attach to P2a (DUAL LR)"
[ "$RUN_P2" = true ] && echo "   screen -r p2b_complex_${DATASET}             # Attach to P2b (DUAL LR)"
[ "$RUN_P2" = true ] && echo "   screen -r p2c_pauli_${DATASET}               # Attach to P2c (DUAL LR)"
[ "$RUN_P2" = true ] && echo "   screen -r p2d_bitflip_${DATASET}             # Attach to P2d (bit flip, DUAL LR)"
[ "$RUN_P2" = true ] && echo "   screen -r p2d_ampdamp_${DATASET}             # Attach to P2d (amp damp, DUAL LR)"
[ "$RUN_P2" = true ] && echo "   screen -r p2d_depol_${DATASET}               # Attach to P2d (depol, DUAL LR)"
[ "$RUN_P3" = true ] && echo "   screen -r p3_no_noise_${DATASET}             # Attach to P3 (no noise)"
[ "$RUN_P3" = true ] && echo "   screen -r p3_bitflip_${DATASET}              # Attach to P3 (bit flip)"
echo ""
echo "   Inside a session:"
echo "   • Ctrl+A then D                              # Detach (keep running)"
echo "   • Ctrl+C                                     # Kill session"
echo ""
echo "📁 Results directory: $BASE_DIR"
echo "   ⚠️  Note: New directory with '_dual_lr' suffix!"
echo ""