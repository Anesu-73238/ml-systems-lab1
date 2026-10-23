import sys
import importlib.metadata as md

print("Python", sys.version)
for pkg in ["numpy", "scikit-learn", "joblib", "psutil", "memory_profiler"]:
    print(pkg, md.version(pkg))