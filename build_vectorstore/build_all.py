import subprocess

for i in range(20):
    subprocess.Popen(
        ["python", "build_faiss_single.py", f"./splits/part_{i}.jsonl", f"./faiss_db_part_{i}"]
    )