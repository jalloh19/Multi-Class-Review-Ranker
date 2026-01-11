import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    from src.preprocessing import load_resources
    print("SUCCESS: load_resources imported successfully.")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"FAILURE: Unexpected error: {e}")
