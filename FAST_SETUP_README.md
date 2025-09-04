# 🚀 Fast Sign Language Recognition Setup

**Get your system running in 15 minutes!**

## ⚡ Super Quick Start (Recommended)

```bash
python quick_start.py
```

This single command will:
- Install dependencies
- Create sample data
- Train models quickly
- Start real-time recognition

## 🏃‍♂️ Manual Fast Track

### 1. Data Collection (5 minutes)
```bash
python -m src.mediapipe_hand_tracker
```

**For Static Signs (A, B, C, etc.):**
- Press `s` and enter gesture name (e.g., "A")
- Hold the pose steady
- Press `SPACE` to capture 5-10 samples
- Press `ENTER` to save

**For Dynamic Gestures (wave, hello, etc.):**
- Press `d` and enter gesture name (e.g., "wave")
- Press `r` to start recording
- Perform the gesture (2 seconds max)
- Recording auto-stops

### 2. Train Models (5 minutes)
```bash
python -m src.gesture_trainer
```
Choose option 3 to train both static and dynamic models.

### 3. Real-Time Recognition (immediately)
```bash
python -m src.unified_predictor
```

## 🎯 What You Get

**Static Recognition:** Letters A, B, C, etc.
- Uses simple neural network
- Instant prediction when hand is still
- 90%+ accuracy with good data

**Dynamic Recognition:** Gestures like wave, hello
- Uses LSTM for sequence learning  
- Detects movement automatically
- Works with 2-second gesture sequences

## 📁 File Structure

```
src/
├── mediapipe_hand_tracker.py  # Data collection
├── gesture_trainer.py         # Model training
├── unified_predictor.py       # Real-time recognition
└── ...

gesture_data/
├── static/     # Static gesture samples
└── dynamic/    # Dynamic gesture sequences

gesture_models/
├── static_model_*.h5    # Trained static model
└── dynamic_model_*.h5   # Trained dynamic model
```

## 🔧 Key Features for Speed

- **MediaPipe**: Accurate hand tracking without setup
- **Fast Training**: 10-30 epochs for quick results
- **Auto Movement Detection**: Switches between static/dynamic automatically  
- **Real-time Prediction**: Live camera feed with smooth predictions
- **Simple Data Format**: JSON files for easy debugging

## 🐛 Quick Troubleshooting

**No camera detected:**
- Check camera permissions
- Try different camera index in code

**Low accuracy:**
- Collect more training samples (20+ per gesture)
- Ensure good lighting
- Keep hand clearly visible

**Slow performance:**
- Reduce model epochs in trainer
- Use lightweight models
- Close other applications

## 🚀 Production Tips

For a real deployment:
1. Collect 50-100 samples per gesture
2. Train for more epochs (50-100)
3. Add data augmentation
4. Test in different lighting conditions

**You now have both static AND dynamic sign recognition working together!**