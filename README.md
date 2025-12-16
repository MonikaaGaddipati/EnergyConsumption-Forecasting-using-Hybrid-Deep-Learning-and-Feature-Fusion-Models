# Energy Consumption Forecasting using Hybrid Deep Learning and Feature Fusion Models

A comprehensive deep learning framework for forecasting household energy consumption using multiple neural network architectures including GRU, LSTM, CNN-GRU-Attention, and Hybrid models. This project provides end-to-end workflows for data preprocessing, model training, evaluation, and interactive visualization.

## 🎯 Overview

Accurate energy consumption forecasting is essential for:
- **Smart Grid Management**: Optimizing energy distribution and load balancing
- **Cost Optimization**: Reducing energy waste and operational costs
- **Demand Planning**: Anticipating peak usage periods
- **Renewable Energy Integration**: Balancing supply and demand efficiently

This project implements and compares multiple deep learning architectures with automated preprocessing pipelines, comprehensive evaluation metrics, and an interactive dashboard for result visualization.

## 🏗️ Architecture

### Implemented Models

1. **GRU (Gated Recurrent Unit)**
   - Lightweight recurrent architecture
   - Efficient for sequential time series data
   - Configurable hidden layers and dropout

2. **LSTM (Long Short-Term Memory)**
   - Advanced RNN with memory cells
   - Handles long-term dependencies
   - Optional bidirectional and attention mechanisms

3. **CNN-GRU with Attention**
   - Convolutional layers for feature extraction
   - GRU for temporal modeling
   - Attention mechanism for focusing on important time steps

4. **Hybrid Deep Learning Model**
   - Combines CNN and GRU architectures
   - Feature fusion capabilities
   - Attention-based temporal weighting

5. **ARIMA (Baseline)**
   - Classical statistical time series model
   - Serves as baseline for comparison

## ✨ Features

### Data Processing
- **Automated Preprocessing**: Data cleaning, resampling, and normalization
- **Feature Engineering**: Time-based features, lag variables, rolling statistics
- **Robust Scaling**: RobustScaler or MinMaxScaler options
- **Train/Test Splitting**: Configurable split ratios

### Model Training
- **Unified Training Pipeline**: Single command to train any model
- **Early Stopping**: Prevents overfitting with configurable patience
- **Learning Rate Scheduling**: Optional step or plateau-based scheduling
- **Checkpoint Management**: Automatic saving of best models

### Evaluation & Metrics
- **Comprehensive Metrics**: MAE, RMSE, MAPE, SMAPE
- **Visual Analysis**: 
  - Training/validation loss curves
  - Forecast vs actual comparisons
  - Residual analysis
  - Error distributions
  - QQ plots for normality testing
  - Attention heatmaps (for attention-based models)

### Interactive Dashboard
- **Real-time Visualization**: Streamlit-based web dashboard
- **Model Comparison**: Side-by-side performance metrics
- **Interactive Charts**: Explore training results visually
- **Multi-model Support**: View results for all trained models

## 📋 Requirements

- **Python**: 3.10 or higher
- **PyTorch**: For deep learning models
- **Core Libraries**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: streamlit
- **Time Series**: statsmodels (for ARIMA)

All dependencies are listed in `requirements.txt`.

## 🚀 Installation

### Step 1: Clone the Repository

```bash
cd EnergyConsumption-Forecasting-using-Hybrid-Deep-Learning-and-Feature-Fusion-Models
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📖 Usage

### Training Models

Train any model using the unified training workflow:

```bash
# Train GRU model
python -m experiments.run_training_workflow --model gru

# Train LSTM model
python -m experiments.run_training_workflow --model lstm

# Train CNN-GRU-Attention model
python -m experiments.run_training_workflow --model cnn_gru_attn

# Train Hybrid model
python -m experiments.run_training_workflow --model hybrid
```

**Options:**
- `--model`: Model type (gru, lstm, cnn_gru_attn, hybrid)
- `--config`: Path to config file (default: `config/config.yaml`)
- `--device`: Device to use (cuda/cpu, auto-detected if not specified)

### Configuration

Edit `config/config.yaml` to customize:

```yaml
preprocessing:
  window_size: 168      # Sequence length
  horizon: 1            # Prediction horizon
  test_size: 0.2        # Test set ratio

training:
  batch_size: 32
  epochs: 50
  lr: 0.001
  early_stop_patience: 8

models:
  gru:
    hidden: 128
    num_layers: 1
    dropout: 0.1
```

### Running the Dashboard

Launch the interactive dashboard to visualize results:

```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501` and display:
- Model performance metrics
- Comparison charts
- Training visualizations
- Interactive model selection

### Data Location

The workflow automatically handles data:
- **Raw Data**: `data/raw/` or downloads from UCI ML Repository
- **Processed Data**: `experiments/data/processed/` or `data/processed/`
- **Results**: `results/checkpoints/` (models), `results/figures/` (plots), `results/*_metrics.csv` (metrics)

## 📁 Project Structure

```
EnergyConsumption-Forecasting-using-Hybrid-Deep-Learning-and-Feature-Fusion-Models/
├── config/
│   └── config.yaml              # Configuration file
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── experiments/
│   ├── __init__.py
│   ├── run_training_workflow.py  # Main training script
│   └── data/
│       └── processed/            # Processed datasets
├── models/                       # Legacy model implementations
├── src/
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Model architectures
│   ├── training/                 # Training and evaluation
│   └── utils/                    # Utilities and config
├── results/
│   ├── checkpoints/              # Saved model weights
│   ├── figures/                  # Generated visualizations
│   └── *_metrics.csv            # Performance metrics
├── requirements.txt
└── README.md
```

## 📊 Output Files

After training, the following files are generated:

### Model Checkpoints
- `results/checkpoints/{model}_best.pth` - Best model weights

### Metrics
- `results/{model}_metrics.csv` - Performance metrics (MAE, RMSE, MAPE, SMAPE)

### Visualizations
- `results/figures/{model}/loss_{model}.png` - Training loss curves
- `results/figures/{model}/forecast_{model}.png` - Forecast vs actual
- `results/figures/{model}/residuals_{model}.png` - Residual analysis
- `results/figures/{model}/error_hist_{model}.png` - Error distribution
- `results/figures/{model}/pred_vs_actual_{model}.png` - Scatter plot
- `results/figures/{model}/qq_{model}.png` - QQ plot
- `results/figures/{model}/error_over_time_{model}.png` - Error timeline
- `results/figures/{model}/attention_{model}.png` - Attention weights (if applicable)

## 🔧 Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'experiments'**
- Make sure you're in the project root directory
- Ensure `experiments/__init__.py` exists

**File not found errors**
- Run training first to generate visualizations
- Check that data files exist in `data/processed/` or `experiments/data/processed/`

**CUDA out of memory**
- Reduce `batch_size` in config
- Use CPU: `--device cpu`

**Dashboard shows "No metrics found"**
- Train at least one model first
- Check that `results/*_metrics.csv` files exist

## 📈 Model Performance

Models are evaluated using:
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **MAPE** (Mean Absolute Percentage Error): Percentage-based error
- **SMAPE** (Symmetric MAPE): Balanced percentage error

## 🤝 Contributing

This project was developed as part of CS5820 AI Final Project by Monikaa Gaddipati.

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **Dataset**: UCI Machine Learning Repository - Individual Household Electric Power Consumption Dataset
- **Frameworks**: PyTorch, Streamlit, scikit-learn
- **Libraries**: pandas, numpy, matplotlib, seaborn, plotly

---

**Built with ❤️ for Energy Forecasting**
