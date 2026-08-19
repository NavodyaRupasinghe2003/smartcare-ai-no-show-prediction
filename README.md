OGBN-Arxiv Node Classification with Graph Neural Networks
A scalable Graph Neural Network pipeline for node classification on the OGBN-Arxiv citation network, combining graph analytics, data preprocessing, GCN, GraphSAGE and GAT architectures, hyperparameter optimization, model evaluation, graph explainability, embedding analysis, and an interactive Streamlit dashboard.

The system uses both paper-level content features and citation relationships to predict the research subject category of each paper.

Overview
Research papers form a naturally structured graph:

Nodes represent research papers.
Edges represent citation relationships.
Node features represent paper content.
Labels represent research subject categories.
Unlike conventional tabular classification, this system learns from both the individual paper representation and the surrounding citation network.

Problem Definition
Given a paper (v), its feature vector (x_v), and the citation graph (G=(V,E)), the objective is to predict the subject category:

y
^
v
=
f
(
X
,
G
)

where:

X
 = node feature matrix
G
 = citation graph
y
^
v
 = predicted subject category for node 
v
The OGBN-Arxiv dataset contains approximately 169K papers, 1.1M citation links, and 128 features per paper. The classification task contains 40 subject categories.

Key Capabilities
Tensor Operations

Tensor creation
Indexing and slicing
Reshaping and dimension manipulation
Matrix multiplication
Broadcasting
Aggregation operations
CPU/GPU device handling
Graph Analytics

Edge-list representation
Sample subgraph visualization
Degree distribution analysis
Graph density analysis
Connected-component analysis
Subject-category distribution
Data Preparation

OGBN-Arxiv feature and label loading
Official train/validation/test splits
Training-only feature normalization
Leakage-aware preprocessing
Graph Neural Networks

Graph Convolutional Network (GCN)
GraphSAGE
Graph Attention Network (GAT)
Training & Optimization

Cross-entropy classification loss
Adam optimization
Learning-rate tuning
Hidden-dimension tuning
Dropout tuning
Weight decay
Validation monitoring
Early stopping
Model checkpointing
Evaluation

Accuracy
Precision
Recall
Macro F1-score
Validation/test comparison
Cross-model performance comparison
Explainability

PCA embedding visualization
t-SNE embedding visualization
GAT attention-weight analysis
Feature ablation / feature importance
Neighborhood influence analysis
Interactive Dashboard

Graph statistics
Model performance
Interactive node classification
Prediction confidence
Top-5 class probabilities
Embedding visualization
Explainability visualizations
System Architecture
                         OGBN-Arxiv Dataset
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Data Acquisition       │
                    │   OGB / PyG              │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Graph Analysis           │
                    │ • Nodes / Edges          │
                    │ • Degree Distribution     │
                    │ • Density                │
                    │ • Components             │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Data Preparation         │
                    │ • Features               │
                    │ • Labels                 │
                    │ • Train/Val/Test Split   │
                    │ • Normalization          │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┼────────────────┐
                 │               │                │
                 ▼               ▼                ▼
          ┌───────────┐   ┌────────────┐   ┌───────────┐
          │    GCN    │   │ GraphSAGE  │   │    GAT    │
          └─────┬─────┘   └──────┬─────┘   └─────┬─────┘
                │                │                │
                └────────────────┼────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ Training & Optimization │
                    │ • Adam                  │
                    │ • Cross Entropy         │
                    │ • Hyperparameter Tuning  │
                    │ • Early Stopping        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Model Evaluation        │
                    │ Accuracy / Precision    │
                    │ Recall / Macro F1       │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
          ┌──────────────────┐      ┌──────────────────┐
          │ Explainability   │      │ Model Artifacts  │
          │ • PCA / t-SNE    │      │ • Checkpoints    │
          │ • GAT Attention  │      │ • Metrics        │
          │ • Feature Ablation│     │ • Configurations │
          │ • Neighborhood   │      └────────┬─────────┘
          └────────┬─────────┘               │
                   └──────────────┬──────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │ Streamlit Dashboard     │
                    │ • Graph Intelligence    │
                    │ • Predictions           │
                    │ • Performance            │
                    │ • Embeddings             │
                    │ • Explainability         │
                    └─────────────────────────┘
Model Architectures
1. Graph Convolutional Network — GCN
The GCN is used as the primary graph-convolution baseline.

Architecture
Input Features (128)
        │
        ▼
GCNConv
        │
BatchNorm
        │
ReLU
        │
Dropout
        │
        ▼
GCNConv
        │
BatchNorm
        │
ReLU
        │
Dropout
        │
        ▼
GCNConv
        │
        ▼
40 Class Logits
The implementation uses three graph-convolution layers, BatchNorm after the hidden convolution layers, ReLU activations, and dropout regularization. Hidden representations can also be extracted before the final classification layer for embedding analysis.

2. GraphSAGE
GraphSAGE provides a different neighborhood aggregation mechanism from GCN.

Architecture
Input Features (128)
        │
        ▼
SAGEConv
        │
BatchNorm
        │
ReLU
        │
Dropout
        │
        ▼
SAGEConv
        │
BatchNorm
        │
ReLU
        │
Dropout
        │
        ▼
SAGEConv
        │
        ▼
40 Class Logits
GraphSAGE learns how neighborhood information is aggregated and is designed to support inductive and scalable graph learning.

3. Graph Attention Network — GAT
GAT extends the system with learnable attention over graph neighborhoods.

Architecture
Input Features
      │
      ▼
Multi-Head GATConv
      │
BatchNorm
      │
ELU
      │
Dropout
      │
      ▼
Single-Head GATConv
      │
      ▼
40 Class Logits
The first GAT layer uses multi-head attention, while the output layer uses a single attention head with concatenation disabled.

The implementation also exposes attention coefficients for interpretability analysis. The GAT implementation explicitly supports extracting the learned attention weights associated with graph edges.
Model Configuration
The model implementations support configurable:

Input dimension
Hidden dimension
Output dimension
Dropout
GAT attention heads
The project also provides a model registry and configuration-based model construction, allowing trained architectures to be reconstructed from saved configuration files.

Dataset
OGBN-Arxiv
The project uses the OGBN-Arxiv dataset from the Open Graph Benchmark.

Property	Value
Dataset	OGBN-Arxiv
Task	Node Classification
Nodes	~169,000
Citation Edges	~1.1 million
Node Features	128
Target Classes	40
Domain	Scientific research / Computer Science
Each node represents a research paper, while edges represent citation relationships. The node features are generated from paper titles and abstracts.

Dataset documentation:

https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv

Raw dataset files are intentionally excluded from version control and are downloaded/generated as part of the data pipeline.

Data Preparation
The preprocessing pipeline includes:

Loading the OGBN-Arxiv dataset.
Extracting node features.
Extracting target labels.
Loading the official train/validation/test split.
Applying feature normalization where appropriate.
Saving processed data for downstream training and dashboard use.
Feature normalization is performed using the training portion to avoid information leakage from validation or test data.

The resulting processed artifact is consumed by both the training pipeline and Streamlit dashboard.

Training Pipeline
The training workflow includes:

Processed Graph
      │
      ▼
Model Initialization
      │
      ▼
Forward Pass
      │
      ▼
Cross-Entropy Loss
      │
      ▼
Backpropagation
      │
      ▼
Adam Optimizer
      │
      ▼
Validation Monitoring
      │
      ├── Improvement → Save Best Model
      │
      └── No Improvement → Early Stopping
Hyperparameters explored include:

Learning rate
Hidden dimension
Dropout
Weight decay
GAT attention heads
Number of layers / architecture configuration
Training outputs include model checkpoints, configuration files, training logs and evaluation results.

Evaluation
The models are evaluated using:

Accuracy
Precision
Recall
Macro F1-score
Evaluation is performed on both validation and test data.

The system also compares model performance across GCN, GraphSAGE and GAT.

The dashboard surfaces these metrics interactively and highlights the strongest model according to accuracy and macro F1.

Explainability
The project includes multiple complementary interpretability techniques.

PCA
Learned node representations are projected into two dimensions using Principal Component Analysis.

t-SNE
t-SNE is used to explore nonlinear structure and potential class clustering within learned embeddings.

GAT Attention Analysis
GAT attention coefficients are extracted to investigate how different neighboring papers contribute to predictions.

Feature Importance
Feature ablation is used to measure how prediction confidence changes when individual input features are removed.

Neighborhood Influence
Predictions are compared under different graph-neighborhood conditions to investigate the contribution of citation relationships.

These analyses allow the system to move beyond simply reporting accuracy and provide insight into why graph models make particular predictions.

Dashboard
The project includes an interactive Streamlit application.

Run:

streamlit run dashboard/app.py
The dashboard contains the following sections:

Overview
High-level dataset and model summary.

Graph Statistics
Displays:

Number of papers
Citation links
Subject categories
Feature dimensions
Degree distribution
Graph density
Category distribution
Model Performance
Displays:

GCN performance
GraphSAGE performance
GAT performance
Accuracy
Precision
Recall
F1-score
Validation vs. test performance
Classify a Paper
Allows users to:

Select a test-set paper
Select a trained model
Generate a prediction
View predicted category
View confidence
Compare against the true label
Inspect top-5 predicted categories
Embedding Explorer
Displays PCA and t-SNE representations of learned embeddings.

Explainability
Provides:

GAT attention visualization
Feature importance
Neighborhood influence
The dashboard loads processed data, model configurations, trained checkpoints, evaluation results and generated analysis artifacts from the project repository.

Technology Stack
Category	Technology
Language	Python
Deep Learning	PyTorch
Graph Deep Learning	PyTorch Geometric
Dataset	Open Graph Benchmark (OGB)
Data Processing	NumPy, Pandas
Machine Learning Utilities	Scikit-learn
Graph Analysis	NetworkX
Visualization	Matplotlib, Plotly
Interactive Dashboard	Streamlit
Experiment Tracking	Weights & Biases
Notebook Environment	Jupyter Notebook
The selected libraries align with the project's graph-learning, analysis, visualization and dashboard requirements.

Requirements
Recommended Environment
The GNN development notebooks were executed using:

Python 3.10
PyTorch 2.3.1+cpu
The project was also tested in CPU-only environments.

Important: The repository currently contains notebook metadata from more than one Python/PyTorch environment. Before publishing the repository, standardize the environment and pin the final versions in requirements.txt. The GNN/model notebook reports PyTorch 2.3.1+cpu, while another tensor notebook reports a different PyTorch version.

Core Dependencies
The project requires:

Python 3.10
PyTorch
PyTorch Geometric
OGB
NumPy
Pandas
Scikit-learn
NetworkX
Matplotlib
Plotly
Streamlit
Jupyter
Install the exact versions specified in the repository's requirements.txt.

For reproducible execution, it is recommended to use the same Python version and dependency versions used to generate the saved model checkpoints.

Installation
1. Clone the Repository
git clone https://github.com/Thrithwaka/ogbn-arxiv-gnn.git
cd ogbn-arxiv-gnn
2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Upgrade pip
python -m pip install --upgrade pip
4. Install Dependencies
pip install -r requirements.txt
If PyTorch Geometric requires a platform-specific installation for your PyTorch/CUDA configuration, follow the corresponding PyTorch Geometric installation instructions before running the notebooks.

Running the Project Locally
Option 1 — Launch the Dashboard
After installing dependencies and ensuring the processed data and trained model artifacts are available:

streamlit run dashboard/app.py
The application should open automatically in your browser.

If it does not, open:

http://localhost:8501
Option 2 — Run the Notebooks
Launch Jupyter:

jupyter notebook
or:

jupyter lab
Then execute the notebooks in logical pipeline order:

01_tensor_fundamentals.ipynb
        ↓
02_03_graph_analysis_and_preparation.ipynb
        ↓
04_gnn_models.ipynb
        ↓
05_06_training_and_evaluation.ipynb
        ↓
07_Explainability.ipynb
The notebooks cover the complete workflow from tensor operations and graph analysis through GNN development, training, evaluation and explainability.

Running the Dashboard with Existing Artifacts
The dashboard expects project artifacts such as:

data/processed/processed_data.pkl
models_checkpoints/model_config.json
models_checkpoints/gcn_final.pt
models_checkpoints/graphsage_final.pt
models_checkpoints/gat_final.pt
reports/model_evaluation_results.csv
reports/embedding_visualization.png
reports/gat_attention_weights.png
reports/feature_importance.png
reports/neighborhood_influence.png
The dashboard includes defensive checks and displays informative messages when expected artifacts are unavailable.

Project Structure
ogbn-arxiv-gnn/
│
├── data/
│   ├── raw/
│   │   └── OGBN-Arxiv raw dataset
│   │
│   └── processed/
│       └── processed_data.pkl
│
├── notebooks/
│   ├── 01_tensor_fundamentals.ipynb
│   ├── 02_03_graph_analysis_and_preparation.ipynb
│   ├── 04_gnn_models.ipynb
│   ├── 05_06_training_and_evaluation.ipynb
│   └── 07_Explainability.ipynb
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── gnn_models.py
│   │
│   ├── data/
│   │   └── data processing modules
│   │
│   └── utils/
│       └── metrics, plotting and analysis utilities
│
├── dashboard/
│   └── app.py
│
├── models_checkpoints/
│   ├── model_config.json
│   ├── gcn_final.pt
│   ├── graphsage_final.pt
│   └── gat_final.pt
│
├── reports/
│   ├── model_evaluation_results.csv
│   ├── embedding_visualization.png
│   ├── gat_attention_weights.png
│   ├── feature_importance.png
│   └── neighborhood_influence.png
│
├── requirements.txt
├── README.md
└── CONTRIBUTORS.md
Large datasets, generated artifacts and model files should remain excluded from Git when appropriate through .gitignore and/or Git LFS.

Model Outputs
The trained models produce logits for each paper across the 40 target categories.

For a graph containing (N) nodes:

Model Output Shape = (N, 40)
The final prediction is obtained using the class with the highest predicted probability.

Reproducibility
For reproducible experiments:

Use the specified Python version.
Create a clean virtual environment.
Install dependencies from requirements.txt.
Use the same OGBN-Arxiv dataset.
Use the saved preprocessing configuration.
Use the same train/validation/test split.
Record model hyperparameters.
Preserve random seeds where applicable.
Keep trained checkpoints and configuration files together.
Record evaluation metrics alongside model versions.
The model package also supports configuration-driven reconstruction through the model registry and build_model_from_config() utility.

Performance Summary
The current experiments compare three GNN architectures:

Model	Role
GCN	Primary graph convolution baseline
GraphSAGE	Alternative neighborhood aggregation architecture
GAT	Attention-based advanced architecture
The current test results from the project experiments are approximately:

Model	Accuracy	Precision	Recall	Macro F1
GCN	71.50%	57.28%	49.62%	51.54%
GraphSAGE	71.36%	57.09%	50.12%	51.22%
GAT	68.72%	57.28%	43.76%	45.32%
These values should always be regenerated from the project's evaluation artifacts if the models or preprocessing pipeline are changed.

Design Considerations
Why GCN?
GCN provides a strong and relatively efficient graph-convolution baseline for learning from citation neighborhoods.

Why GraphSAGE?
GraphSAGE provides a different neighborhood aggregation strategy and offers a useful comparison against GCN.

Why GAT?
GAT introduces learnable attention weights, allowing the model to assign different importance to neighboring papers. It also enables direct attention-based graph explainability.

The GAT implementation therefore serves both as an additional model architecture and as the basis for attention-based interpretability.

Interpretability Strategy
The system uses multiple levels of interpretability:

Global Representation
        │
        ├── PCA
        └── t-SNE
                │
                ▼
       Learned Embedding Space

Local Graph Explanation
        │
        ├── GAT Attention
        └── Neighborhood Influence
                │
                ▼
       Neighbor Contribution

Input-Level Analysis
        │
        └── Feature Ablation
                │
                ▼
       Feature Contribution
This combination provides complementary views of model behavior rather than relying on a single explanation method.

Limitations
Current limitations include:

Full-graph inference can be computationally expensive on CPU-only environments.
The interactive dashboard currently exposes a subset of test-set nodes for direct selection.
Embedding visualizations are generated from sampled nodes rather than the entire graph.
The dataset contains class imbalance, so accuracy alone does not fully represent performance.
Saved model checkpoints depend on compatible PyTorch/PyTorch Geometric versions.
Reproducing exact training results may require matching the original environment and random seeds.
Future Improvements
Potential extensions include:

Graph Transformer architectures
Mini-batch / neighbor-sampling training
GPU acceleration
Automated experiment tracking
Improved hyperparameter search
Confusion-matrix analysis
Per-class performance dashboards
Searchable paper-level prediction
More efficient inference caching
Additional graph explainability methods
Model calibration and uncertainty estimation
Deployment through a production API
Containerized deployment with Docker
CI/CD for automated testing and validation
Development Workflow
The repository follows a modular development structure:

Feature Branch
     │
     ▼
Implementation
     │
     ▼
Notebook / Unit Validation
     │
     ▼
Experiment
     │
     ▼
Evaluation
     │
     ▼
Pull Request
     │
     ▼
Review
     │
     ▼
Main Branch
Recommended branch naming:

feature/<short-name>
Examples:

feature/gcn-model
feature/gat-explainability
feature/dashboard-performance
feature/data-pipeline
Team
Contributor	Primary Area
Aadhil	Tensor operations & foundations
Thilani	Graph analysis & data pipeline
Thrithwaka	GNN model development
Navodya	Training, optimization & evaluation
Kaveesha	Explainability & dashboard
See CONTRIBUTORS.md for additional contribution details.

Contributing
Create a feature branch.
Keep changes focused and modular.
Test notebooks and Python modules before opening a pull request.
Update documentation when functionality changes.
Keep generated datasets and large artifacts out of Git unless explicitly required.
Use clear, descriptive commit messages.
Open a pull request for review before merging.
Troubleshooting
ModuleNotFoundError: torch_geometric
Install PyTorch Geometric and verify that its version is compatible with the installed PyTorch version.

python -c "import torch; print(torch.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
Dashboard cannot find processed data
Ensure:

data/processed/processed_data.pkl
exists.

The dashboard checks for this file before enabling live graph statistics.

Dashboard cannot load models
Ensure the following files exist:

models_checkpoints/gcn_final.pt
models_checkpoints/graphsage_final.pt
models_checkpoints/gat_final.pt
models_checkpoints/model_config.json
Streamlit does not start
Verify the virtual environment is activated:

python --version
pip --version
Then run:

streamlit run dashboard/app.py
CUDA unavailable
The project supports CPU execution. If CUDA is available and the environment is configured correctly, the training workflow can be adapted to use the GPU.

Dataset Citation
If this project is used in another research or engineering context, cite the Open Graph Benchmark and the OGBN-Arxiv dataset according to the official OGB documentation:

https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
