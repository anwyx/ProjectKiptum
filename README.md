# ProjectKiptum

A framework for Task-and-Motion Planning (TAMP) with ManiSkill and PDDL.  
Defines a dynamic PDDL‑to‑ManiSkill converter and a high‑level test harness.

---

## 📁 Repository Structure

```
TAMPbench/
├── setup.py
├── README.md               # ← this file
├── .gitignore
├── scripts/                # scripts list
├── src/
│   └── tampbench/          # Installable Python package
│       ├── __init__.py
│       ├── env/
│       │   ├── __init__.py
│       │   ├── pddl_maniskill_converter.py  # Convert pddl to maniskill tasks
│       │   ├── action.py       # PDDLAction implementation
│       │   ├── tamp_env.py     # Environment glue code
|       |   └── utils.py        # Env utils
│       ├── datasets/           # datasets download scripts from huggingface
|       ├── motion_planner/     # motion_planner for translate pddl action to motion (only used in data collection+ task planning only)
|       |   └── motion_solver.py
|       ├──policy/              # policy warpper 
|── benchmark/
│   └── table-top/  # table on tasks
|       ├── super_long_horizon/ #task type
|       ├── clutter/
├── baselines/ 
|        ├── tamp_baselines
|        └── vla_baselines  
└── examples/
    └── test.py                # Example: example scripts for how to use environment. Future include notebook for easy start.
```

---

## 🔧 Installation

1. Clone the repo
   ```bash
   git clone https://github.com/HaoranZhangumich/TAMPBench
   cd TAMPbench
   ```

2. Create & activate a virtual environment
   ```bash
    conda env create -f environment.yaml
   ```

3. Install dependencies
   ```bash
   pip install -e .
   ```
   > Adjust the above list to match the actual packages your code requires.

---

## 🚀 Quick Start

Run the example script to instantiate the TAMP environment, inspect the registry, and perform a pick/place action:

```bash
python example/test.py
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
## 📅 Project Roadmap & Schedule

| Milestone                         | Target Date     | Status       |
|-----------------------------------|-----------------|--------------|
| PDDL ↔ ManiSkill conversion   | July 15, 2025   | Finished  |


#TODO

---

## 📝 Usage

### Importing your environment

```python
from tampbench.env.tamp_env import TAMPEnvironment

# Create:
env = TAMPEnvironment(
    domain_file="path/to/domain.pddl",
    problem_file="path/to/problem.pddl"
)

# Reset & Step:
obs = env.reset()
obs, reward, done, info = env.step("pick", {"r": "panda_wristcam", "o": "cubeA", "p": "pos1"})
```

### Key Components

- **`action.py`**  
  Wraps PDDL actions into executable operators.

- **`pddl_maniskill_converter.py`**  
  Converts PDDL problems into a ManiSkill task, with:
  - `_load_scene`: builds cubes, table, robot  
  - `_initialize_episode`: places cubes with randomized sampler  
  - `evaluate()`: checks goal predicates and returns a 0-D tensor success flag

- **`tamp_env.py`** 
  OpenAI gym-like environment wrapper for TAMP 
  


---

## ⚙️ Configuration

- TODO


---

## 📚 References

- [ManiSkill Documentation](https://maniskill.readthedocs.io/en/latest/user_guide/)  
- [PDDL tutorial](https://fareskalaboud.github.io/LearnPDDL/)  

---

## 🛠️ Development

- TODO

---

## 🤝 Contributing

Feel free to open issues or submit pull requests. Please follow the [Contributor Guidelines](./CONTRIBUTING.md) (TBD). (TODO)

---

## 📄 License

TODO
