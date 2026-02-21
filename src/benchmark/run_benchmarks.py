import os
import time
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def run_script_and_parse_output(script_path: str, framework_name: str) -> dict:
  """
  Runs a training script as a subprocess and parses its stdout for metrics.
  """
  print(f"\n--- Running {framework_name} Benchmark ---")
  
  env = os.environ.copy()
    
  start_time = time.time()
  
  result = subprocess.run(
    ["uv", "run", "python", script_path],
    capture_output=True,
    text=True,
    env=env
  )
  
  total_time = time.time() - start_time
  
  if result.returncode != 0:
    print(f"Error running {framework_name}:")
    print(result.stderr)
    return None
    
  output = result.stdout
  
  metrics = {
    "Framework": framework_name,
    "Total Script Time (s)": round(total_time, 2),
    "Training Time (s)": None,
    "Inference Time (s)": None,
    "Accuracy (%)": None
  }
  
  for line in output.split('\n'):
    if "Training completed in" in line:
      metrics["Training Time (s)"] = float(line.split()[-2])
    elif "Inference completed in" in line:
      metrics["Inference Time (s)"] = float(line.split()[-2])
    elif "Accuracy:" in line:
      metrics["Accuracy (%)"] = float(line.split()[-1].replace('%', ''))
      
  return metrics

def plot_benchmarks(df: pd.DataFrame):
  """
  Creates bar charts comparing the frameworks across different metrics.
  """
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))
  
  df.plot(x="Framework", y="Accuracy (%)", kind="bar", ax=axes[0], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
  axes[0].set_title("Test Accuracy")
  axes[0].set_ylabel("Accuracy (%)")
  axes[0].set_ylim(0, 100)
  axes[0].tick_params(axis='x', rotation=0)
  
  df.plot(x="Framework", y="Training Time (s)", kind="bar", ax=axes[1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
  axes[1].set_title("Training Time")
  axes[1].set_ylabel("Seconds")
  axes[1].tick_params(axis='x', rotation=0)
  
  df.plot(x="Framework", y="Inference Time (s)", kind="bar", ax=axes[2], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
  axes[2].set_title("Inference Time (Test Set)")
  axes[2].set_ylabel("Seconds")
  axes[2].tick_params(axis='x', rotation=0)
  
  plt.tight_layout()
  plt.savefig(PROJECT_ROOT / "benchmark_results.png")
  print(f"\nBenchmark plots saved to {PROJECT_ROOT / 'benchmark_results.png'}")
  plt.show()

def main():
  """
  Runs all models and compiles the benchmark results.
  """
  scripts = [
    ("src/baseline/baseline_model.py", "Scikit-learn (RF)"),
    ("src/pytorch/train.py", "PyTorch (CNN)"),
    ("src/tensorflow/train.py", "TensorFlow (CNN)")
  ]
  
  results = []
  
  for script_path, name in scripts:
    full_path = str(PROJECT_ROOT / script_path)
    metrics = run_script_and_parse_output(full_path, name)
    if metrics:
      results.append(metrics)
      
  if not results:
    print("No benchmarks completed successfully.")
    return
    
  df = pd.DataFrame(results)
  
  print("\n" + "="*50)
  print("FINAL BENCHMARK RESULTS")
  print("="*50)
  print(df.to_string(index=False))
  
  plot_benchmarks(df)

if __name__ == "__main__":
  main()
