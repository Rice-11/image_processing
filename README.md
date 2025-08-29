# 🤟 Sign Language Interpreter

A real-time American Sign Language (ASL) recognition system using deep learning and computer vision. This project recognizes 24 ASL letters (A-Y, excluding J & Z) from webcam input with high accuracy and smooth performance.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.13+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.8+-green.svg)

## ✨ Features

- **🎯 Real-time Recognition**: Live webcam-based ASL letter detection
- **🎨 Multiple Model Types**: Basic, Advanced, and Lightweight CNN architectures  
- **📊 High Accuracy**: Achieves 85-95% accuracy on clear gestures
- **⚡ Smooth Performance**: Runs at 15-30 FPS on standard hardware
- **🖥️ Interactive UI**: User-friendly interface with visual feedback
- **📸 Screenshot Feature**: Save predictions with timestamps
- **🔧 Easy Setup**: One-command installation and demo

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Interactive Demo

```bash
# Start the interactive application
python main.py --interactive
```

Or try the quick demo:

```bash
# Quick demo (trains lightweight model + starts camera)
python main.py --demo
```

## 📁 Project Structure

```
Hand sign (ML)/
│
├── src/                          # Source code
│   ├── data_loader.py           # Dataset handling and preprocessing
│   ├── model.py                 # CNN model architectures
│   ├── train.py                 # Model training pipeline
│   └── camera_predictor.py      # Real-time prediction system
├── models/                       # Trained models (auto-created)
├── data/                        # Dataset storage (auto-created) 
├── screenshots/                 # Saved predictions (auto-created)
├── requirements.txt             # Python dependencies
├── main.py                      # Main application
└── README.md                    # This file
```

## 🎮 Usage Options

### Interactive Mode (Recommended)
```bash
python main.py --interactive
```
Choose from menu options:
1. **Quick Demo** - Train lightweight model + run camera
2. **Train Model** - Guided training workflow
3. **Camera Recognition** - Use existing trained model
4. **Model Info** - View available architectures
5. **List Models** - See all trained models

### Command Line Usage

**Training:**
```bash
# Train basic model
python main.py --train --model_type basic --epochs 50

# Train advanced model (higher accuracy)
python main.py --train --model_type advanced --epochs 100

# Train lightweight model (faster inference)
python main.py --train --model_type lightweight --epochs 30
```

**Real-time Recognition:**
```bash
# Use latest trained model
python main.py --predict

# Use specific model with custom settings
python main.py --predict --model_path models/your_model.h5 --confidence_threshold 0.8
```

## 🏗️ Model Architectures

| Model Type | Accuracy | Speed | Training Time | Best For |
|------------|----------|-------|---------------|-----------|
| **Basic** | 85-90% | 25ms | 30 min (CPU) | Learning & testing |
| **Advanced** | 90-95% | 35ms | 2 hours (CPU) | Production use |
| **Lightweight** | 80-85% | 15ms | 15 min (CPU) | Real-time apps |

## 📷 Camera Controls

When running real-time recognition:

- **Q** - Quit application
- **R** - Reset prediction buffer  
- **S** - Save screenshot with prediction
- **C** - Toggle confidence threshold

## 🎯 Supported Signs

The system recognizes **24 ASL letters**: `A B C D E F G H I K L M N O P Q R S T U V W X Y`

*Note: J and Z are excluded as they require motion gestures*

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.15+, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 8GB  
- **Storage**: 2GB free space
- **Camera**: USB webcam or built-in camera (720p minimum)

### Recommended for Training
- **GPU**: NVIDIA GTX 1060+ with CUDA support
- **RAM**: 16GB+
- **CPU**: Intel i7 or AMD Ryzen 7

## 🛠️ VS Code Setup

### Extensions
1. **Python** (Microsoft)
2. **Pylance** (Microsoft)
3. **Jupyter** (Microsoft)

### Running in VS Code
1. Open the project folder in VS Code
2. Select Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose your Python environment
4. Run with `F5` or use integrated terminal

## 📊 Performance Tips

### For Better Accuracy:
- Use good lighting conditions
- Keep hand clearly visible in ROI rectangle
- Use solid background
- Make clear, distinct gestures

### For Faster Performance:
- Use lightweight model for real-time apps
- Close other applications during use
- Use GPU acceleration if available
- Reduce camera resolution if needed

## 🐛 Troubleshooting

### Common Issues

**Camera not working:**
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'Error')"
```

**Installation errors:**
```bash
# Update pip and try again
pip install --upgrade pip
pip install -r requirements.txt
```

**Low performance:**
- Try the lightweight model
- Check GPU drivers for CUDA support
- Reduce camera resolution
- Close background applications

**Poor accuracy:**
- Ensure good lighting
- Keep hand in green rectangle
- Use solid background
- Train with more epochs
- Try advanced model

## 📈 Training Your Own Model

### Dataset Information
- **Source**: [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist)
- **Size**: ~34,000 training images, ~8,700 test images
- **Format**: 28x28 grayscale images
- **Classes**: 24 ASL letters (A-Y excluding J,Z)

### Custom Training
```bash
# Basic training with data augmentation
python main.py --train --model_type basic --epochs 50

# Advanced training without augmentation
python main.py --train --model_type advanced --epochs 100 --no_augmentation

# Quick lightweight training
python main.py --train --model_type lightweight --epochs 20
```

### Training Output
- Model saved to `models/` directory
- Training plots saved as PNG files
- Results saved as JSON metadata
- Confusion matrix visualization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist) by DataMunge
- **TensorFlow**: Deep learning framework
- **OpenCV**: Computer vision library  
- **Kaggle**: Dataset hosting platform

## 📞 Support

- **Issues**: Report bugs and request features
- **Documentation**: Check this README for setup help
- **Performance**: See troubleshooting section above

## 🗺️ Future Enhancements

- [ ] Support for more sign languages (BSL, etc.)
- [ ] Mobile app development
- [ ] Cloud deployment options
- [ ] Multi-hand gesture recognition  
- [ ] Sentence/phrase recognition
- [ ] Real-time text-to-speech conversion

---

**Made with ❤️ for the deaf and hard-of-hearing community**

*Ready to start recognizing sign language? Run `python main.py --demo` to get started in minutes!*