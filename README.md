# ProjectKiptum
<img width="512" height="512" alt="App Logo" src="https://github.com/user-attachments/assets/dab70112-1c42-4ef0-a576-7d7d9277216a" />

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

Run the example script to instantiate the TAMP environment, inspect the registry, and perform a pick/place action:

```bash
python test.py
```

You should see output like:

```
=== Testing Pick Action ===
checking object at True
Preconditions met for pick
Create subtask for pick
Executing pick action...
Pick action completed successfully (simulated)
Step finished for action: pick
Pick result: reward=10, done={'success': tensor(False)}, info={'success': True}

=== Testing Place Action ===
Preconditions met for place
Create subtask for place
Executing place action...
Pose([-0.0840995, -0.0953381, 0.0586828], [-0.000479358, 0.888979, 0.457947, -0.000480696])
Place action completed successfully (simulated)
Step finished for action: place
Place result: reward=10, done={'success': tensor(True)}, info={'success': True}
```
First done measn the overall task is finished, the info return whether subtask (symbolic action) finished

---

## 📚 References

- [Zero-Shot Object Detection with Qwen2.5-VL](https://github.com/roboflow/notebooks/blob/main/notebooks/zero-shot-object-detection-with-qwen2-5-vl.ipynb)  

---


## 🤝 Contributing

Feel free to open issues or submit pull requests.

