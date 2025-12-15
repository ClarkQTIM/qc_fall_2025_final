# Statistical Analysis Summary
================================================================================

## Test 1

### Test 1a: No Noise Comparison
- P1 (dropout_0.0): 0.9375 ± 0.0048
- P3 (No Noise): 0.6654 ± 0.0709
- Cohen's d: 5.4166

### Test 1b: Best Configuration
- P1 Best (dropout_0.0): 0.9375 ± 0.0048
- P3 Best (bit_flip_0.01): 0.7198 ± 0.0660
- Cohen's d: 4.6564

## Test 2
**Result:** ✓ SIGNIFICANT (p=9.7394e-03)

**Summary:**
- P2a (Probabilities): 0.8553 ± 0.0686
- P2b (Complex Amplitudes): 0.9187 ± 0.0103

## Test 3

**Optimal noise levels:**
- amp_damp: p=0.03, AUC=0.9398
- bit_flip: p=0.01, AUC=0.9398
- depol: p=0.01, AUC=0.9421

## Test 4
**Result:** ✓ SIGNIFICANT (p=4.3574e-21)
- Optimal dropout: 0.0
- AUC: 0.9375

## Test 5
**Result:** ✓ SIGNIFICANT (p=1.5609e-09)
- Best P2: P2b (0.9187 ± 0.0103)
- P3: 0.6654 ± 0.0709
- Cohen's d: 5.0003

## Test 6
**Result:** ✗ Not significant (p=6.4337e-01)

**Optimal configurations:**
- amp_damp: p=0.03, AUC=0.9398
- bit_flip: p=0.01, AUC=0.9398
- depol: p=0.01, AUC=0.9421

## Test 7
**Result:** ✗ Not significant (p=5.8821e-02)
- Best Classical: P1 (dropout_0.0) - 0.9375 ± 0.0048
- Best Quantum: P2d (depol_0.01) - 0.9421 ± 0.0052
- Cohen's d: -0.9022
- **🏆 WINNER: Quantum**
