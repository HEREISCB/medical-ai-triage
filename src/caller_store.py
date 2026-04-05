"""File-based caller info store, shared between web UI and agent processes."""

import json
import os

STORE_DIR = "/tmp/medical_triage_callers"
os.makedirs(STORE_DIR, exist_ok=True)


def set_caller(room_name: str, name: str, phone: str, email: str):
    path = os.path.join(STORE_DIR, f"{room_name}.json")
    with open(path, "w") as f:
        json.dump({"name": name, "phone": phone, "email": email}, f)


def get_caller(room_name: str) -> dict:
    path = os.path.join(STORE_DIR, f"{room_name}.json")
    try:
        with open(path, "r") as f:
            data = json.load(f)
        os.remove(path)  # Clean up
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"name": "there", "phone": "", "email": ""}
