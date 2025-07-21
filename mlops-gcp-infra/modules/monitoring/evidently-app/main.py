from fastapi import FastAPI, Request
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
import uuid
import subprocess

app = FastAPI()

@app.post("/generate_report")
async def generate_report(request: Request):
    data = await request.json()
    ref = pd.DataFrame(data["reference"])
    cur = pd.DataFrame(data["current"])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    report_id = str(uuid.uuid4())
    out_path = f"/tmp/{report_id}.html"
    report.save_html(out_path)

    if "bucket_path" in data:
        subprocess.run(["gsutil", "cp", out_path, data["bucket_path"]], check=True)

    with open(out_path, "r") as f:
        return {"report": f.read()}