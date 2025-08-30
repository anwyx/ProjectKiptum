# ProjectKiptum
<div align="center">
<img width="200" height="200" alt="App Logo" src="https://github.com/user-attachments/assets/dab70112-1c42-4ef0-a576-7d7d9277216a" />
</div>

PrivyLens, a privacy-first image redaction agent using LLM that automatically detects sensitive information in photos and instantly blurs it on-device.

---

## 📁 Repository Structure

```
ProjectKiptum/
├── test.py
├── main.py
├── app.py
├── requirements.txt        # dependencies
├── README.md               # ← this file
├── .gitignore
├── assets/ 
|       ├── App Logo
|       ├── style.css
├── config/ 
|       ├── model_config.py
├── data/
│       └── raw/          
│       ├── processed/
├── models/
│       └── __pycache__/
|       ├── prompts.py
|       ├── qwen.py
|       ├── qwen_model.py
├── pages/
│       └── __pycache__/
|       ├── gallery.py
|       ├── home.py
└── utils/
        ├── annotations.py
        ├── detection.py
        └── image_utils.py               
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

You should see output like:



---

## 📚 References

- [Zero-Shot Object Detection with Qwen2.5-VL](https://github.com/roboflow/notebooks/blob/main/notebooks/zero-shot-object-detection-with-qwen2-5-vl.ipynb)  

---


## 🤝 Contributing

Feel free to open issues or submit pull requests.

