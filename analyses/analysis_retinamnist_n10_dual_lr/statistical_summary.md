# Statistical Analysis Summary
================================================================================

## Test 1

### Test 1a: No Noise Comparison
- P1 (dropout_0.0): 0.8730 ± 0.0059
- P3 (No Noise): 0.7824 ± 0.0630
- Cohen's d: 2.0234

### Test 1b: Best Configuration
- P1 Best (dropout_0.0): 0.8730 ± 0.0059
- P3 Best (depol_0.2): 0.8064 ± 0.0926
- Cohen's d: 2.2317

## Test 2
**Result:** ✓ SIGNIFICANT (p=2.3140e-02)

**Summary:**
- P2a (Probabilities): 0.7764 ± 0.0794
- P2b (Complex Amplitudes): 0.8424 ± 0.0278

## Test 3

**Optimal noise levels:**
- amp_damp: p=0.05, AUC=0.8702
- bit_flip: p=0.03, AUC=0.8674
- depol: p=0.1, AUC=0.8695

## Test 4
**Result:** ✓ SIGNIFICANT (p=2.1861e-11)
- Optimal dropout: 0.0
- AUC: 0.8730

## Test 5
**Result:** ✓ SIGNIFICANT (p=1.3012e-02)
- Best P2: P2b (0.8424 ± 0.0278)
- P3: 0.7824 ± 0.0630
- Cohen's d: 1.2324

## Test 6
**Result:** ✗ Not significant (p=8.9529e-01)

**Optimal configurations:**
- amp_damp: p=0.05, AUC=0.8702
- bit_flip: p=0.03, AUC=0.8674
- depol: p=0.1, AUC=0.8695

## Test 7
**Result:** ✗ Not significant (p=5.8836e-01)
- Best Classical: P1 (dropout_0.0) - 0.8730 ± 0.0059
- Best Quantum: P2d (amp_damp_0.05) - 0.8702 ± 0.0145
- Cohen's d: 0.2464
- **🏆 WINNER: Classical**
