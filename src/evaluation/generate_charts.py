import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(filepath: str = "data/evaluation/experiment_results.json"):
    if not Path(filepath).exists():
        print(f"Error: Results file not found at {filepath}")
        return None
    with open(filepath, "r") as f:
        return json.load(f)

def plot_retrieval_performance(results):
    retrieval_data = next((r for r in results if r["experiment_name"] == "Retrieval_Benchmarking"), None)
    if not retrieval_data: return
    
    metrics = retrieval_data["metrics"]
    labels = ["Semantic Only", "Hybrid (BM25 + Dense)"]
    values = [metrics.get("Semantic_Hit@5", 0), metrics.get("Hybrid_Hit@5", 0)]
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=labels, y=values, palette="viridis")
    plt.title("Retrieval Performance (Hit@5)", fontsize=14, pad=15)
    plt.ylabel("Hit Rate", fontsize=12)
    plt.ylim(0, 1.1)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig("data/evaluation/retrieval_performance.png", dpi=300)
    print("Saved retrieval_performance.png")

def plot_misuse_detection(results):
    misuse_data = next((r for r in results if r["experiment_name"] == "Misuse_Detection"), None)
    if not misuse_data: return
    
    metrics = misuse_data["metrics"]
    labels = ["Precision", "Recall", "F1-Score", "Accuracy"]
    values = [
        metrics.get("Misuse_Precision", 0),
        metrics.get("Misuse_Recall", 0),
        metrics.get("Misuse_F1", 0),
        metrics.get("Accuracy", 0)
    ]
    
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=labels, y=values, palette="magma")
    plt.title("Misuse Detection Performance", fontsize=14, pad=15)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.1)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig("data/evaluation/misuse_detection.png", dpi=300)
    print("Saved misuse_detection.png")

def plot_latency_analysis(results):
    latency_data = next((r for r in results if r["experiment_name"] == "Latency_Analysis"), None)
    if not latency_data: return
    
    metrics = latency_data["metrics"]
    labels = ["Retrieval", "IPC-CAM Scoring", "Total Pipeline"]
    values = [
        metrics.get("Latency_Retrieval_Sec", 0),
        metrics.get("Latency_CAM_Sec", 0),
        metrics.get("Latency_Total_Backend_Sec", 0)
    ]
    
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=labels, y=values, palette="crest")
    plt.title("Component Latency Analysis", fontsize=14, pad=15)
    plt.ylabel("Time (Seconds)", fontsize=12)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}s", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig("data/evaluation/latency_analysis.png", dpi=300)
    print("Saved latency_analysis.png")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') # For headless environments
    sns.set_theme(style="whitegrid")
    
    print("Generating evaluation charts for the research paper...")
    results = load_results()
    if results:
        Path("data/evaluation").mkdir(parents=True, exist_ok=True)
        plot_retrieval_performance(results)
        plot_misuse_detection(results)
        plot_latency_analysis(results)
        print("Done! Charts are ready for inclusion in your paper.")
