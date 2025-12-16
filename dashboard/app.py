import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")

st.title("⚡ Energy Consumption Forecasting Dashboard")
st.markdown("This dashboard visualizes **actual vs predicted energy consumption** for different models.")

# Get the project root directory (parent of dashboard)
project_root = Path(__file__).parent.parent
results_dir = project_root / "results"
figures_dir = results_dir / "figures"

# Load metrics from CSV files
@st.cache_data
def load_metrics():
    """Load metrics from all model CSV files."""
    metrics = {}
    metrics_files = {
        "GRU": results_dir / "gru_metrics.csv",
        "LSTM": results_dir / "lstm_metrics.csv",
        "CNN-GRU-Attn": results_dir / "cnn_gru_attn_metrics.csv",
        "Hybrid": results_dir / "hybrid_metrics.csv",
    }

    for model_name, file_path in metrics_files.items():
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                # Get the latest row (most recent training)
                if len(df) > 0:
                    latest = df.iloc[-1]
                    metrics[model_name] = {
                        "MAE": latest.get("MAE", 0),
                        "RMSE": latest.get("RMSE", 0),
                        "MAPE": latest.get("MAPE", 0),
                        "SMAPE": latest.get("SMAPE", 0),
                    }
            except Exception as e:
                st.warning(f"Could not load metrics for {model_name}: {e}")

    return metrics

metrics = load_metrics()

# Display metrics in columns
if metrics:
    st.subheader("📊 Model Performance Metrics")
    cols = st.columns(len(metrics))

    for idx, (model_name, model_metrics) in enumerate(metrics.items()):
        with cols[idx]:
            st.metric(
                label=model_name,
                value=f"{model_metrics['RMSE']:.4f}",
                delta=f"MAE: {model_metrics['MAE']:.4f}",
                help=f"MAPE: {model_metrics['MAPE']:.2f}% | SMAPE: {model_metrics['SMAPE']:.2f}%"
            )

    # Create comparison chart
    if len(metrics) > 1:
        st.subheader("📈 Model Comparison")
        comparison_df = pd.DataFrame(metrics).T
        fig = px.bar(
            comparison_df,
            y=['MAE', 'RMSE'],
            barmode='group',
            title="Model Performance Comparison (MAE & RMSE)",
            labels={'value': 'Error', 'index': 'Model'}
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ No metrics files found. Please run training first using: `python -m experiments.run_training_workflow --model <model_name>`")

# Display available visualizations
st.subheader("📊 Training Visualizations")
available_models = []
if figures_dir.exists():
    for model_dir in figures_dir.iterdir():
        if model_dir.is_dir():
            available_models.append(model_dir.name)

if available_models:
    selected_model = st.selectbox("Select a model to view visualizations:", available_models)
    model_figures_dir = figures_dir / selected_model

    if model_figures_dir.exists():
        # Define all possible figure types
        all_figure_types = {
            "Training Loss": f"loss_{selected_model}.png",
            "Forecast vs Actual": f"forecast_{selected_model}.png",
            "Residuals": f"residuals_{selected_model}.png",
            "Error Histogram": f"error_hist_{selected_model}.png",
            "Predicted vs Actual": f"pred_vs_actual_{selected_model}.png",
            "QQ Plot": f"qq_{selected_model}.png",
            "Error Over Time": f"error_over_time_{selected_model}.png",
            "Attention Weights": f"attention_{selected_model}.png",
            "Multi-Step Forecast": f"forecast_multi_{selected_model}.png",
        }

        # Only show figures that actually exist
        available_figures = {}
        for fig_name, fig_file in all_figure_types.items():
            fig_path = model_figures_dir / fig_file
            if fig_path.exists():
                available_figures[fig_name] = fig_file

        if available_figures:
            selected_figure = st.selectbox("Select visualization:", list(available_figures.keys()))
            figure_path = model_figures_dir / available_figures[selected_figure]
            st.image(str(figure_path), use_container_width=True)
        else:
            st.warning(f"No visualizations found for {selected_model} model.")
    else:
        st.info(f"No figures directory found for {selected_model}")
else:
    st.info("No model visualizations found. Run training to generate figures.")

st.markdown("---")
st.caption("Data: UCI Energy Consumption Dataset | Built by Monikaa Gaddipati")
