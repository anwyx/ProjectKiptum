# ProjectKiptum
<div align="center">
<img width="200" height="200" alt="App Logo" src="https://github.com/user-attachments/assets/dab70112-1c42-4ef0-a576-7d7d9277216a" />
</div>

PrivyLens, a privacy-first image redaction agent using LLM that automatically detects sensitive information in photos and instantly blurs it on-device.

---

## 📁 Repository Structure


```
ProjectKiptum/
├── test.py               # Example/test script for running the app or features
├── app.py                # Dash app initialization and layout
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation (this file)
├── assets/               # Static assets for Dash (CSS, images, etc.)
│   ├── App Logo          # App logo image(s)
│   └── style.css         # Custom styles for the app
├── config/               # Configuration files
│   └── model_config.py   # Model and app configuration settings
├── data/                 # Data storage
│   ├── raw/              # Raw, unprocessed images
│   └── processed/        # Processed/blurred images
├── models/               # Model code and weights
│   ├── prompts.py        # Prompt templates for LLMs
│   ├── qwen.py           # Qwen model interface/utilities
│   └── qwen_model.py     # Qwen model loading and logic
├── pages/                # Dash multi-page app modules
│   ├── gallery.py        # Gallery page (image display, blur, unlock)
│   └── home.py           # Home/landing page
└── utils/                # Utility modules
    ├── annotations.py    # Annotation helpers for images
    ├── detection.py      # Detection logic (e.g., face, object)
    └── image_utils.py    # Image processing utilities
```

---

## 🔧 Installation

1. Clone the repo
   ```bash
   git clone https://github.com/anwyx/ProjectKiptum
   cd ProjectKiptum
   ```

2. Install dependencies
   ```bash
    pip install -r requirements.txt
   ```

   > Adjust the above list to match the actual packages your code requires.

---

## 🚀 Quick Start


```bash
python test.py
```





---

## 📚 References

- [Zero-Shot Object Detection with Qwen2.5-VL](https://github.com/roboflow/notebooks/blob/main/notebooks/zero-shot-object-detection-with-qwen2-5-vl.ipynb)  

---


## 🤝 Contributing

Feel free to open issues or submit pull requests.

