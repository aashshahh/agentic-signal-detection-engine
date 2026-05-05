#!/usr/bin/env python3
"""
Signal Detection Engine v3
Run: python run.py
Then open: http://localhost:8000
"""
import subprocess, sys, os

os.makedirs("data", exist_ok=True)
os.makedirs("frontend/public", exist_ok=True)
print(" Signal Detection Engine v3")
print("   http://localhost:8000")
print("   API docs: http://localhost:8000/docs")
print("   Ctrl+C to stop\n")
subprocess.run([sys.executable,"-m","uvicorn","backend.server:app","--host","0.0.0.0","--port","8000","--reload"])
