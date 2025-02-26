#!/usr/bin/env python3
import json
import sys

def fix_kernel_metadata(notebook_file):
    try:
        with open(notebook_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {notebook_file}: {e}")
        return

    if "metadata" in data and "kernelspec" in data["metadata"]:
        ks = data["metadata"]["kernelspec"]
        old_name = ks.get("name", "<none>")
        if old_name != "python":
            print(f"Updating kernel name in {notebook_file} from '{old_name}' to 'geotgr'")
            ks["name"] = "python3"
            ks["display_name"] = "geotgr"
            try:
                with open(notebook_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print("Update successful!")
            except Exception as e:
                print(f"Error writing {notebook_file}: {e}")
        else:
            print(f"{notebook_file} already has the correct kernel name ('geotgr').")
    else:
        print("No kernelspec metadata found in the notebook.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: fix_kernel.py <notebook.ipynb>")
        sys.exit(1)
    notebook_file = sys.argv[1]
    fix_kernel_metadata(notebook_file)
