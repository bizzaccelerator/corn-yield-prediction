from fastapi import FastAPI, Request, HTTPException, Form, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import uuid
import os
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Evidently Monitoring Service")

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("=== EVIDENTLY SERVICE STARTING ===")
    logger.info(f"PORT environment variable: {os.getenv('PORT', 'not set')}")
    
    # Create uploads directory
    os.makedirs("/tmp/uploads", exist_ok=True)
    os.makedirs("/tmp/reports", exist_ok=True)
    
    logger.info("=== STARTUP COMPLETE ===")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evidently Monitoring</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            textarea { width: 100%; height: 150px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Evidently Data Drift Monitoring</h1>
            <p>Upload your reference and current datasets to generate a drift report.</p>
            
            <form id="reportForm">
                <div class="form-group">
                    <label for="reference">Reference Data (JSON format):</label>
                    <textarea id="reference" placeholder='[{"feature1": 1, "feature2": 2}, {"feature1": 3, "feature2": 4}]'></textarea>
                </div>
                
                <div class="form-group">
                    <label for="current">Current Data (JSON format):</label>
                    <textarea id="current" placeholder='[{"feature1": 1.1, "feature2": 2.1}, {"feature1": 3.1, "feature2": 4.1}]'></textarea>
                </div>
                
                <button type="submit">Generate Drift Report</button>
            </form>
            
            <div id="result" class="result" style="display: none;"></div>
        </div>

        <script>
            document.getElementById('reportForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>⏳ Generating report...</p>';
                resultDiv.style.display = 'block';
                resultDiv.className = 'result';
                
                try {
                    const reference = JSON.parse(document.getElementById('reference').value);
                    const current = JSON.parse(document.getElementById('current').value);
                    
                    const response = await fetch('/generate_report', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            reference: reference,
                            current: current
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <h3>✅ Report Generated Successfully!</h3>
                            <p><strong>Report ID:</strong> ${result.report_id}</p>
                            <p><a href="/report/${result.report_id}" target="_blank">📊 View Report</a></p>
                        `;
                    } else {
                        resultDiv.innerHTML = `<p>❌ Error: ${result.detail}</p>`;
                        resultDiv.className = 'result error';
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p>❌ Error: ${error.message}</p>`;
                    resultDiv.className = 'result error';
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "evidently-monitoring"}

@app.post("/generate_report")
async def generate_report(request: Request):
    """Generate Evidently drift report"""
    try:
        # Import evidently here to catch any import errors
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        
        data = await request.json()
        
        # Validate input data
        if "reference" not in data or "current" not in data:
            raise HTTPException(status_code=400, detail="Missing 'reference' or 'current' data")
        
        ref = pd.DataFrame(data["reference"])
        cur = pd.DataFrame(data["current"])
        
        logger.info(f"Processing data - Reference: {ref.shape}, Current: {cur.shape}")

        # Generate Evidently report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)

        # Save report locally
        report_id = str(uuid.uuid4())
        out_path = f"/tmp/reports/{report_id}.html"
        report.save_html(out_path)

        logger.info(f"Report generated successfully with ID: {report_id}")
        
        return {
            "report_id": report_id,
            "status": "success",
            "message": "Report generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/report/{report_id}")
async def get_report(report_id: str):
    """Serve the generated HTML report"""
    report_path = f"/tmp/reports/{report_id}.html"
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(report_path, media_type="text/html")

@app.get("/reports")
async def list_reports():
    """List all available reports"""
    reports_dir = Path("/tmp/reports")
    if not reports_dir.exists():
        return {"reports": []}
    
    reports = []
    for report_file in reports_dir.glob("*.html"):
        reports.append({
            "id": report_file.stem,
            "url": f"/report/{report_file.stem}",
            "created": report_file.stat().st_mtime
        })
    
    return {"reports": reports}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info(f"=== STARTING EVIDENTLY SERVER ON PORT {port} ===")
    uvicorn.run(app, host="0.0.0.0", port=port)