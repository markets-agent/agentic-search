import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)


def log_if_debug(message: str):
    if os.getenv("WITH_DEBUG_MESSAGES") == "true":
        print(f"\033[36m[DEBUG] \n{message}\n\033[0m")  # Cyan color for debug messages
