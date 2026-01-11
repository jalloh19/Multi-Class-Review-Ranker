import sys
import os
import time
import threading

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def import_module():
    try:
        print("Attempting to import src.preprocessing...")
        import src.preprocessing
        print("Import successful!")
    except Exception as e:
        print(f"Import failed: {e}")

# Run import in a thread to check for blocking
t = threading.Thread(target=import_module)
t.start()

# Wait max 5 seconds
t.join(timeout=5)

if t.is_alive():
    print("FAILURE: Import timed out (blocking call detected).")
    os._exit(1)
else:
    print("SUCCESS: Import completed efficiently.")
