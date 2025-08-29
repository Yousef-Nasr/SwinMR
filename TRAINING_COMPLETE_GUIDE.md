# SwinMR Complete Training Guide

## 📊 Training Output Explanation

When you see this line during training:
```
[       1] Epoch:  0 | LR:2.00e-04 | Loss:0.4523 | Img:0.3215 | Freq:0.0892 | Perc:0.0416
```

**Each value means:**
- **`[1]`**: Current iteration/batch number
- **`Epoch: 0`**: Current epoch (full pass through dataset)
- **`LR:2.00e-04`**: Learning rate (0.0002)
- **`Loss:0.4523`**: **Total combined loss**
- **`Img:0.3215`**: **Image reconstruction loss** (main component)
- **`Freq:0.0892`**: **Frequency domain loss** (FFT-based)
- **`Perc:0.0416`**: **Perceptual loss** (LPIPS-based)

**Loss Formula:**
```
Total Loss = 15×Img + 0.1×Freq + 0.0025×Perc
```
So: 0.4523 ≈ 15×0.3215 + 0.1×0.0892 + 0.0025×0.0416

## 🔄 Testing & Saving Schedule

### Current Configuration (from JSON):
- **Testing**: Every **1,000 iterations** (`checkpoint_test: 1000`)
- **Model Saving**: Every **10,000 iterations** (`checkpoint_save: 10000`)
- **Detailed Logging**: Every **200 iterations** (`checkpoint_print: 200`)

### What Happens During Testing:
1. **Tests on ALL test data** (calculates metrics on full dataset)
2. **Saves 20 merged comparison images** showing:
   - Ground Truth | Noisy Input | Predicted Output
3. **Calculates and displays:**
   - Average PSNR (Peak Signal-to-Noise Ratio)
   - Average SSIM (Structural Similarity)
   - Average LPIPS (Perceptual similarity)
   - FID Score (Fréchet Inception Distance)

### Example Testing Output:
```
================================================================================
VALIDATION PHASE - Iteration    1,000
================================================================================
Processing 25 validation samples...
Saving merged comparisons for first 20 samples...

Testing sample   1/25 (  4.0%) - Processing...
Testing sample   2/25 (  8.0%) - Processing...
...
Testing sample  25/25 (100.0%) - Processing...

--------------------------------------------------------------------------------
VALIDATION RESULTS
--------------------------------------------------------------------------------
Average PSNR     :  32.4567 dB
Average SSIM     : 0.945123
Average LPIPS    : 0.067543
FID Score        :  15.2341
================================================================================
```

## 💾 Model Saving & Resumption

### Automatic Saving:
- **Model weights** saved every 10,000 iterations
- **Optimizer state** saved (for proper resumption)
- **Scheduler state** saved (maintains learning rate schedule)

### Resumption:
The system **automatically resumes** from the last checkpoint if you stop and restart training:

```
================================================================================
TRAINING INITIALIZATION
================================================================================
Model Type       : swinmr_npi
Task             : swinmr_enhanced_noise
Batch Size       : 1
Workers          : 0
Learning Rate    : 2.00e-04
Total Samples    : 15,307
Iterations/Epoch : 15,307
Distributed      : No
Test Every       : 1,000 iterations
Save Every       : 10,000 iterations
Resuming from    : Iteration 12,500 (continuing training)  ← Shows resumption
--------------------------------------------------------------------------------
```

## 🖼️ Saved Images Structure

### Directory Structure:
```
results/
├── merged_comparisons/           ← New merged images (20 per test)
│   ├── comparison_01000_sample_000.png
│   ├── comparison_01000_sample_001.png
│   └── ...
├── tempH/                       ← GT images for FID
├── tempE/                       ← Predicted images for FID
├── tempL/                       ← Noisy images for FID
└── individual_samples/          ← Individual sample folders
    ├── sample_001/
    │   ├── GT_01000.png
    │   ├── Recon_01000.png
    │   └── ZF_01000.png
    └── ...
```

### Merged Images:
Each merged image shows **side-by-side comparison**:
```
[Ground Truth] | [Noisy Input] | [Predicted Output]
```
- **20 samples saved** each testing cycle
- **Labeled** (if PIL available)
- **Easy visual comparison** of reconstruction quality

## 🚀 Training Process Flow

1. **Training starts** → Shows configuration summary
2. **Every iteration** → Inline progress display
3. **Every 1,000 iterations** → Detailed milestone + Testing
4. **Every 10,000 iterations** → Model saving
5. **Automatic resumption** if stopped/restarted

### Complete Training Session Example:
```
🚀 Training started! Iteration updates will appear below:

[       1] Epoch:  0 | LR:2.00e-04 | Loss:0.4523 | Img:0.3215 | Freq:0.0892 | Perc:0.0416
[       2] Epoch:  0 | LR:2.00e-04 | Loss:0.4501 | Img:0.3201 | Freq:0.0885 | Perc:0.0415
...
[     999] Epoch:  0 | LR:2.00e-04 | Loss:0.4102 | Img:0.2890 | Freq:0.0801 | Perc:0.0411

================================================================================
MILESTONE - Iteration    1,000 | Epoch   0
================================================================================
Learning Rate    : 2.000000e-04
Total Loss       : 0.410200
Speed            : 15.67 iter/sec (63.8s/1k iters)
================================================================================

================================================================================
VALIDATION PHASE - Iteration    1,000
================================================================================
[Testing happens here - 20 merged images saved]

💾 Saving model at iteration 10,000...
✅ Model saved successfully!
```

## 🎯 Key Features Implemented

✅ **Real-time progress** for every iteration
✅ **Scheduled testing** based on JSON configuration
✅ **All test data evaluation** for accurate metrics
✅ **20 merged comparison images** saved each test
✅ **Automatic model saving** with resumption capability
✅ **Visual progress indicators** and clear status messages
✅ **Robust error handling** with graceful recovery

## 📝 Configuration Options

You can adjust these in your JSON config:
```json
{
  "train": {
    "checkpoint_test": 1000,    // Test every N iterations
    "checkpoint_save": 10000,   // Save model every N iterations
    "checkpoint_print": 200     // Detailed logging every N iterations
  }
}
```

The system is now fully automated and will run continuously with regular testing, saving, and the ability to resume from any interruption! 🎉