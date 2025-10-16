import time
import psutil
import pandas as pd
from pathlib import Path
from functools import wraps

results = []

def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time execution
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        # Record stats
        results.append({
            'function': func.__name__,
            'args': str(args),
            'execution_time': execution_time,
            'memory_used': mem_used
        })
        
        return result
    return wrapper

def save_benchmark_results():
    df = pd.DataFrame(results)
    Path("benchmarks").mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    df.to_csv(f"benchmarks/benchmark_{timestamp}.csv", index=False)
    return df

# Apply this decorator to functions you want to benchmark
@benchmark
def example_function():
    time.sleep(1)
    return "done"