# Efficient Graph Summarization Using Structural and Degree-Based Similarity

## Course
CSI3020 – Advanced Graph Algorithms  

## Author
- Name: B. Bhuvanesh  
- Reg No: 23MID0384  

---

## Overview
This project implements a graph summarization framework to reduce the size of large graphs while preserving important structural properties.

The system combines:
- Structural similarity (Jaccard)
- Degree-based similarity
- Adaptive threshold selection
- Hierarchical multi-level summarization
- Comparison with Louvain community detection

---

## Features
- Hybrid Graph Summarization (Degree + Jaccard)
- Adaptive Threshold Selection (data-driven)
- Hierarchical Multi-Level Summarization
- Louvain Algorithm Comparison
- Visualization of graphs and metrics
- Multi-dataset experimentation

---

## Project Structure
```
project/
│
├── main.py
├── graph_summarizer.py
├── adaptive_summarizer.py
├── hierarchical_summarizer.py
├── louvain_comparator.py
├── dataset_loader.py
├── visualizer.py
├── comparison_visualizer.py
├── adaptive_visualizer.py
├── hierarchical_visualizer.py
│
└── outputs/
```

---

## Requirements
Make sure Python 3.8 or above is installed.

Install required libraries:
```bash
pip install networkx numpy matplotlib
```

---

## How to Run
1. Open terminal / command prompt  
2. Navigate to the project folder:
```bash
cd path/to/project
```
3. Run:
```bash
python main.py
```

---

## What Happens When You Run

### Loads datasets:
- Karate Club  
- Petersen Graph  
- Erdos–Rényi Graph  
- Barabasi–Albert Graph  

### Performs:
- Graph summarization  
- Threshold sensitivity analysis  
- Comparison with Louvain algorithm  
- Adaptive vs Standard comparison  
- Hierarchical summarization  

---

## Output
All results are automatically generated and saved in:
```
outputs/
```

Generated outputs include:
- Graph comparison images  
- Degree distribution plots  
- Compression metrics charts  
- Node and edge reduction plots  
- Modularity comparison graphs  
- Runtime comparison  
- Adaptive analysis plots  
- Hierarchical analysis plots  

---

## Important Notes
- Uses non-interactive backend (`matplotlib Agg`)  
- Graphs will NOT open in a window  
- All plots are saved directly in the `outputs/` folder  

---

## Troubleshooting
- Missing module → install required packages again  
- No outputs → check if `outputs/` folder is created  
- Errors → ensure all `.py` files are in the same directory  
```


