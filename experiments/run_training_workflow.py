#!/usr/bin/env python3
"""
Training workflow script for energy consumption forecasting models.
Supports: gru, lstm, cnn_gru_attn, hybrid
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset
import joblib

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.loader import load_raw
from src.data.preprocessor import run_full_preprocessing
from src.data.dataset import EnergyDataset
from src.training.trainer import train_model
from src.training.evaluator import (
    evaluate_model_on_loader,
    compute_metrics,
    plot_loss,
    plot_forecast,
    plot_attention,
    plot_residuals,
    plot_error_histogram,
    plot_pred_vs_actual_scatter,
    plot_error_qq,
    plot_error_over_time,
    save_metrics_csv,
)
from src.models.gru import GRUModel
from src.models.lstm import LSTMModel
from src.models.cnn_gru_attn import CNN_GRU_Attn
from src.models.hybrid import HybridModel


def load_or_preprocess_data(cfg_path: str = "config/config.yaml"):
    """Load processed data if available, otherwise preprocess from raw."""
    cfg = load_config(cfg_path)
    processed_dir = cfg.get("data", {}).get("processed_dir", "data/processed")
    # Check both standard location and experiments location
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        alt_dir = Path("experiments/data/processed")
        if alt_dir.exists():
            processed_dir = alt_dir
    
    train_path = processed_dir / "trains.csv"
    test_path = processed_dir / "tests.csv"
    scaler_path = processed_dir / "scaler.joblib"
    
    # Check if processed data exists
    if train_path.exists() and test_path.exists() and scaler_path.exists():
        print(f"✅ Loading preprocessed data from {processed_dir}")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Set timestamp as index if present
        if "timestamp" in train_df.columns:
            train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
            train_df = train_df.set_index("timestamp")
        if "timestamp" in test_df.columns:
            test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
            test_df = test_df.set_index("timestamp")
        
        scaler = joblib.load(scaler_path)
        return train_df, test_df, scaler
    else:
        print("⚠️  Preprocessed data not found. Running preprocessing...")
        raw_dir = cfg.get("data", {}).get("raw_dir", "data/raw")
        energy_csv = cfg.get("data", {}).get("energy_csv", "household_power_consumption.txt")
        
        # Load raw data
        try:
            raw_df = load_raw(use_ucimlrepo=True, raw_dir=raw_dir, filename=energy_csv)
        except Exception as e:
            print(f"Failed to load via ucimlrepo: {e}")
            raw_df = load_raw(use_ucimlrepo=False, raw_dir=raw_dir, filename=energy_csv)
        
        # Preprocess
        preprocess_cfg = cfg.get("preprocessing", {})
        train_df, test_df, scaler = run_full_preprocessing(
            raw_df=raw_df,
            processed_dir=str(processed_dir),
            resample_freq=preprocess_cfg.get("resample_freq", "h"),
            resample_agg=preprocess_cfg.get("resample_agg", "mean"),
            test_size=preprocess_cfg.get("test_size", 0.2),
            scaler_type=preprocess_cfg.get("scaler_type", "robust"),
            interpolate_limit=preprocess_cfg.get("interpolate_limit", 24),
        )
        return train_df, test_df, scaler


def create_model(model_type: str, input_dim: int, static_dim: int, cfg: dict):
    """Create model based on type."""
    model_type = model_type.lower()
    model_cfg = cfg.get("models", {}).get(model_type, {})
    
    if model_type == "gru":
        return GRUModel(
            n_features=input_dim,
            n_static=static_dim,
            hidden=model_cfg.get("hidden", 128),
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.1),
        )
    elif model_type == "lstm":
        return LSTMModel(
            input_dim=input_dim,
            static_dim=static_dim,
            lstm_hidden=model_cfg.get("hidden", 128),
            lstm_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.1),
            bidirectional=model_cfg.get("bidirectional", False),
            use_attention=model_cfg.get("use_attention", False),
        )
    elif model_type == "cnn_gru_attn":
        return CNN_GRU_Attn(
            input_dim=input_dim,
            cnn_channels=model_cfg.get("cnn_channels", 32),
            hidden_size=model_cfg.get("hidden", 128),
            num_layers=model_cfg.get("num_layers", 1),
            attn_dim=model_cfg.get("attn_dim", 32),
        )
    elif model_type == "hybrid":
        return HybridModel(
            input_dim=input_dim,
            static_dim=static_dim,
            hidden_dim=model_cfg.get("hidden", 128),
            cnn_channels=model_cfg.get("cnn_channels", 32),
            kernel_size=model_cfg.get("kernel_size", 3),
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.2),
            use_attention=model_cfg.get("use_attention", True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: gru, lstm, cnn_gru_attn, hybrid")


def prepare_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict):
    """Create train/val/test datasets."""
    window = cfg.get("preprocessing", {}).get("window_size", 168)
    horizon = cfg.get("preprocessing", {}).get("horizon", 1)
    
    # Identify feature columns (exclude target)
    target_col = "energy_consumption"
    all_cols = [c for c in train_df.columns if c != target_col]
    
    # Separate sequence and static features
    # Static features: binary flags, time features that don't change much
    static_cols = [c for c in all_cols if c in ["is_weekend", "is_holiday"]]
    seq_cols = [c for c in all_cols if c not in static_cols]
    
    # Create datasets
    train_dataset = EnergyDataset(train_df, seq_cols, static_cols, target_col, window, horizon)
    test_dataset = EnergyDataset(test_df, seq_cols, static_cols, target_col, window, horizon)
    
    # Split train into train/val (80/20)
    n_train = len(train_dataset)
    n_val = int(n_train * 0.2)
    n_train_final = n_train - n_val
    
    # Create validation split
    val_dataset = Subset(train_dataset, range(n_train_final, n_train))
    train_dataset_final = Subset(train_dataset, range(n_train_final))
    
    print(f"📊 Dataset sizes: Train={n_train_final}, Val={n_val}, Test={len(test_dataset)}")
    print(f"📊 Features: {len(seq_cols)} sequence, {len(static_cols)} static")
    
    return train_dataset_final, val_dataset, test_dataset, len(seq_cols), len(static_cols)


def main():
    parser = argparse.ArgumentParser(description="Train energy consumption forecasting models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gru", "lstm", "cnn_gru_attn", "hybrid"],
        help="Model type to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified.",
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting training workflow for {args.model.upper()} model")
    print(f"📁 Config: {args.config}")
    
    # Load config
    cfg = load_config(args.config)
    
    # Load/preprocess data
    train_df, test_df, scaler = load_or_preprocess_data(args.config)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset, input_dim, static_dim = prepare_datasets(
        train_df, test_df, cfg
    )
    
    # Create model
    print(f"🏗️  Creating {args.model.upper()} model...")
    model = create_model(args.model, input_dim, static_dim, cfg)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training config
    train_cfg = cfg.get("training", {})
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Checkpoint path
    ckpt_dir = cfg.get("results", {}).get("ckpt_dir", "results/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.model}_best.pth")
    
    # Train
    print(f"🎯 Starting training...")
    history, saved_path = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=train_cfg,
        device=device,
        save_path=ckpt_path,
        seed=42,
    )
    
    print(f"✅ Training complete! Best model saved to: {saved_path}")
    
    # Evaluate on test set
    print(f"📊 Evaluating on test set...")
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(test_dataset, batch_size=train_cfg.get("batch_size", 32), shuffle=False)
    y_true, y_pred, attn_weights, _, _ = evaluate_model_on_loader(
        model, test_loader, device=device
    )
    
    # Flatten arrays to ensure 1D shape
    y_true = y_true.flatten() if y_true.ndim > 1 else y_true
    y_pred = y_pred.flatten() if y_pred.ndim > 1 else y_pred
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    print(f"\n📈 Test Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # Save metrics
    results_dir = cfg.get("results", {}).get("out_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, f"{args.model}_metrics.csv")
    save_metrics_csv(metrics, metrics_path)
    
    # Create visualizations
    print(f"📊 Generating visualizations...")
    figures_dir = os.path.join(results_dir, "figures", args.model)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot loss
    plot_loss(history, args.model, save_path=os.path.join(figures_dir, f"loss_{args.model}.png"), show=False)
    
    # Plot forecast
    plot_forecast(
        y_true, y_pred, args.model,
        save_path=os.path.join(figures_dir, f"forecast_{args.model}.png"),
        show=False
    )
    
    # Plot residuals
    plot_residuals(
        y_true, y_pred, args.model,
        save_path=os.path.join(figures_dir, f"residuals_{args.model}.png"),
        show=False
    )
    
    # Plot error histogram
    plot_error_histogram(
        y_true, y_pred, args.model,
        save_path=os.path.join(figures_dir, f"error_hist_{args.model}.png"),
        show=False
    )
    
    # Plot pred vs actual
    plot_pred_vs_actual_scatter(
        y_true, y_pred, args.model,
        save_path=os.path.join(figures_dir, f"pred_vs_actual_{args.model}.png"),
        show=False
    )
    
    # Plot QQ plot
    plot_error_qq(
        y_true, y_pred, args.model,
        save_path=os.path.join(figures_dir, f"qq_{args.model}.png"),
        show=False
    )
    
    # Plot error over time
    plot_error_over_time(
        y_true, y_pred, args.model,
        save_path=os.path.join(figures_dir, f"error_over_time_{args.model}.png"),
        show=False
    )
    
    # Plot attention if available
    if attn_weights is not None:
        plot_attention(
            attn_weights, args.model,
            save_path=os.path.join(figures_dir, f"attention_{args.model}.png"),
            show=False
        )
    
    print(f"\n✅ All done! Results saved to {results_dir}")
    print(f"   - Model checkpoint: {saved_path}")
    print(f"   - Metrics: {metrics_path}")
    print(f"   - Figures: {figures_dir}")


if __name__ == "__main__":
    main()

