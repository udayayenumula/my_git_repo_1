# main.py - FIXED VERSION
# Added better error handling and validation

import os
import json
import uuid
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from loguru import logger
from pydantic import BaseModel, Field
import uvicorn

from document_processor import DocumentProcessor

# Configuration
KGPT_URL = os.getenv("KGPT_URL", "https://llm-proxy.kpit.com")
API_TOKEN = os.getenv("API_TOKEN", "sk-xZ5joPpGCQImFUm0ioi3uA")
KGPT_MODEL = os.getenv("KGPT_MODEL", "kgpt-multimodal-gpt")
WORKSPACE_ROOT = os.getenv("WORKSPACE_ROOT", str(Path("workspace").resolve()))
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "6000"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

os.makedirs(WORKSPACE_ROOT, exist_ok=True)

# Configure logger with more detail
logger.remove()
logger.add(
    "logs/document_summarizer_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="DEBUG",  # Changed to DEBUG
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
)
logger.add(
    lambda msg: print(msg, end=''),
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)

# FastAPI app
app = FastAPI(
    title="AI Document Summarizer Pro",
    version="3.0.1-FIXED",
    description="Enterprise-grade AI document analysis with robust error handling",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class SummaryJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    filename: Optional[str] = None
    estimated_time: Optional[str] = None
    endpoints: Optional[Dict[str, str]] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    progress: Optional[float] = Field(None, ge=0, le=100)
    current_step: Optional[str] = None
    filename: Optional[str] = None
    title: Optional[str] = None
    pages_processed: Optional[int] = None
    total_pages: Optional[int] = None
    images_analyzed: Optional[int] = None
    total_images: Optional[int] = None
    chunks_created: Optional[int] = None
    chunks_summarized: Optional[int] = None
    api_calls_total: Optional[int] = None
    api_calls_failed: Optional[int] = None
    report_file: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_updated: Optional[str] = None

class SummaryResultResponse(BaseModel):
    job_id: str
    status: str
    title: str
    markdown: str
    report_file: str
    statistics: Dict[str, Any]
    processing_time: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    kgpt_connection: str
    workspace_status: str
    version: str
    uptime: str

# Global state
active_jobs: Dict[str, DocumentProcessor] = {}
start_time = datetime.now()

# Helper Functions
def estimate_processing_time(file_size_mb: float) -> str:
    """Estimate processing time based on file size."""
    estimated_seconds = file_size_mb * 10
    if estimated_seconds < 60:
        return f"{int(estimated_seconds)} seconds"
    elif estimated_seconds < 3600:
        return f"{int(estimated_seconds / 60)} minutes"
    else:
        hours = int(estimated_seconds / 3600)
        minutes = int((estimated_seconds % 3600) / 60)
        return f"{hours}h {minutes}m"

async def cleanup_old_jobs(max_age_hours: int = 24):
    """Clean up jobs older than max_age_hours."""
    try:
        current_time = datetime.now()
        cleaned = 0
        for job_dir in os.listdir(WORKSPACE_ROOT):
            job_path = os.path.join(WORKSPACE_ROOT, job_dir)
            if not os.path.isdir(job_path):
                continue
                
            status_file = os.path.join(job_path, "status.json")
            
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                    
                    completed_at = status.get('completed_at')
                    if completed_at:
                        completed_time = datetime.fromisoformat(completed_at)
                        age_hours = (current_time - completed_time).total_seconds() / 3600
                        
                        if age_hours > max_age_hours:
                            shutil.rmtree(job_path, ignore_errors=True)
                            cleaned += 1
                            logger.info(f"🗑️ Cleaned up old job: {job_dir}")
                except Exception as e:
                    logger.warning(f"Error processing job {job_dir}: {e}")
        
        if cleaned > 0:
            logger.info(f"✅ Cleaned up {cleaned} old jobs")
    except Exception as e:
        logger.error(f"Error cleaning up old jobs: {e}")

# Background Task with Enhanced Error Handling
async def process_document_background(
    job_id: str,
    file_path: str,
    title: str,
    max_tokens: int
):
    """Background task for document processing with comprehensive error handling."""
    logger.info("=" * 80)
    logger.info(f"🚀 Starting document processing for job {job_id}")
    logger.info(f"📄 File: {file_path}")
    logger.info(f"📝 Title: {title}")
    logger.info("=" * 80)
    
    processor = None
    
    try:
        # Initialize processor
        processor = DocumentProcessor(
            job_id=job_id,
            workspace_root=WORKSPACE_ROOT,
            kgpt_url=KGPT_URL,
            api_token=API_TOKEN,
            model=KGPT_MODEL,
            max_tokens_per_chunk=max_tokens
        )
        
        # Store in active jobs
        active_jobs[job_id] = processor
        
        # Validate input file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Process the document
        await processor.process_document(
            file_path=file_path,
            title=title
        )
        
        logger.info("=" * 80)
        logger.info(f"✅ Job {job_id} completed successfully")
        logger.info(f"📊 Final Stats:")
        logger.info(f"   - Pages: {processor.stats['pages_processed']}/{processor.stats['total_pages']}")
        logger.info(f"   - Images: {processor.stats['images_analyzed']}/{processor.stats['total_images']}")
        logger.info(f"   - Chunks: {processor.stats['chunks_summarized']}/{processor.stats['chunks_created']}")
        logger.info(f"   - API Success Rate: {((processor.stats['api_calls_total'] - processor.stats['api_calls_failed']) / max(processor.stats['api_calls_total'], 1) * 100):.1f}%")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ Job {job_id} FAILED with error:")
        logger.error(f"   {type(e).__name__}: {str(e)}")
        logger.error("=" * 80)
        logger.exception("Full traceback:")
        
        # Update status to failed
        try:
            if processor:
                processor.update_status(
                    "failed",
                    f"Processing failed: {str(e)}",
                    error=str(e),
                    error_type=type(e).__name__
                )
        except:
            # If even status update fails, write minimal status file
            try:
                status_file = os.path.join(WORKSPACE_ROOT, job_id, "status.json")
                with open(status_file, "w") as f:
                    json.dump({
                        "job_id": job_id,
                        "status": "failed",
                        "message": f"Critical failure: {str(e)}",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "last_updated": datetime.now().isoformat()
                    }, f, indent=2)
            except Exception as critical_error:
                logger.critical(f"Cannot even write status file: {critical_error}")
    
    finally:
        # Cleanup
        if job_id in active_jobs:
            del active_jobs[job_id]
        
        # Remove uploaded file
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temp file: {e}")

# API Endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with comprehensive API information."""
    return {
        "service": "AI Document Summarizer Pro",
        "version": "3.0.1-FIXED",
        "status": "operational",
        "fixes_applied": [
            "✅ Guaranteed summary file creation with multi-level fallbacks",
            "✅ API retry mechanism with exponential backoff",
            "✅ Comprehensive error handling at each step",
            "✅ File validation after each processing stage",
            "✅ Emergency fallback summaries",
            "✅ Enhanced logging and debugging"
        ],
        "capabilities": {
            "max_pages": "1000+",
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "supported_formats": ["PDF", "DOCX"],
            "features": [
                "✅ Vision AI for image analysis",
                "✅ Intelligent text chunking",
                "✅ Multi-threaded processing",
                "✅ Real-time progress tracking",
                "✅ Comprehensive summaries",
                "✅ Automatic retry on failures",
                "✅ Multiple fallback mechanisms"
            ]
        },
        "endpoints": {
            "upload": "POST /summarize - Upload document",
            "status": "GET /status/{job_id} - Check progress",
            "result": "GET /result/{job_id} - Get summary",
            "download": "GET /download/{job_id} - Download report",
            "stream": "GET /stream/{job_id} - Stream progress (SSE)",
            "jobs": "GET /jobs - List all jobs",
            "delete": "DELETE /job/{job_id} - Delete job",
            "cleanup": "POST /cleanup - Clean old jobs",
            "health": "GET /health - Health check"
        },
        "configuration": { "model": KGPT_MODEL,
            "max_tokens_per_chunk": MAX_TOKENS_PER_CHUNK,
            "workspace": WORKSPACE_ROOT,
            "streaming_enabled": ENABLE_STREAMING,
            "api_timeout": "180s",
            "max_retries": 3
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with API validation."""
    try:
        import requests
        
        # Test KGPT connection
        response = requests.post(
            f"{KGPT_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": KGPT_MODEL,
                "messages": [{"role": "user", "content": "health check"}],
                "max_tokens": 5
            },
            timeout=10
        )
        
        kgpt_status = "✅ Connected" if response.status_code == 200 else f"❌ Failed ({response.status_code})"
        
        # Check workspace
        workspace_writable = os.access(WORKSPACE_ROOT, os.W_OK)
        workspace_readable = os.access(WORKSPACE_ROOT, os.R_OK)
        workspace_status = "✅ Ready" if (workspace_writable and workspace_readable) else "❌ Access issues"
        
        # Calculate uptime
        uptime_seconds = (datetime.now() - start_time).total_seconds()
        uptime_hours = int(uptime_seconds / 3600)
        uptime_minutes = int((uptime_seconds % 3600) / 60)
        uptime = f"{uptime_hours}h {uptime_minutes}m"
        
        overall_status = "healthy" if response.status_code == 200 and workspace_writable else "degraded"
        
        return {
            "status": overall_status,
            "kgpt_connection": kgpt_status,
            "workspace_status": workspace_status,
            "version": "3.0.1-FIXED",
            "uptime": uptime
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "kgpt_connection": f"❌ Error: {str(e)}",
            "workspace_status": "Unknown",
            "version": "3.0.1-FIXED",
            "uptime": "0h 0m"
        }

@app.post("/summarize", response_model=SummaryJobResponse)
async def summarize_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF or DOCX file (supports 1000+ pages)"),
    title: Optional[str] = Form(None, description="Custom title for the summary")
):
    """
    Upload and process a large document with robust error handling.
    
    Returns a job_id for tracking. The document is processed asynchronously.
    """
    logger.info(f"📨 Received upload: {file.filename} ({file.content_type})")
    
    # Validate file type
    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Only PDF or DOCX files are supported."
        )
    
    # Read file
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read uploaded file: {str(e)}"
        )
    
    file_size_mb = len(content) / (1024 * 1024)
    
    # Validate file size
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.2f}MB). Maximum size: {MAX_FILE_SIZE_MB}MB"
        )
    
    if file_size_mb < 0.001:  # Less than 1KB
        raise HTTPException(
            status_code=400,
            detail="File appears to be empty or corrupted"
        )
    
    logger.info(f"📊 File size: {file_size_mb:.2f} MB")
    
    # Create job
    job_id = str(uuid.uuid4())
    job_path = os.path.join(WORKSPACE_ROOT, job_id)
    
    try:
        os.makedirs(job_path, exist_ok=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create job directory: {str(e)}"
        )
    
    # Save uploaded file
    file_extension = Path(file.filename).suffix.lower()
    if not file_extension:
        file_extension = ".pdf" if "pdf" in file.content_type else ".docx"
    
    temp_file_path = os.path.join(job_path, f"input{file_extension}")
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(content)
        logger.info(f"💾 Saved to: {temp_file_path}")
    except Exception as e:
        shutil.rmtree(job_path, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Initialize status
    status_file = os.path.join(job_path, "status.json")
    try:
        with open(status_file, "w") as f:
            json.dump({
                "job_id": job_id,
                "status": "queued",
                "message": "Document queued for processing",
                "progress": 0,
                "filename": file.filename,
                "title": title or file.filename,
                "file_size_mb": round(file_size_mb, 2),
                "started_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
    except Exception as e:
        shutil.rmtree(job_path, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize job: {str(e)}"
        )
    
    # Estimate processing time
    estimated_time = estimate_processing_time(file_size_mb)
    
    # Start background processing
    background_tasks.add_task(
        process_document_background,
        job_id=job_id,
        file_path=temp_file_path,
        title=title or file.filename,
        max_tokens=MAX_TOKENS_PER_CHUNK
    )
    
    logger.info(f"✅ Job {job_id} queued (estimated time: {estimated_time})")
    
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "message": f"Document '{file.filename}' queued for processing",
            "filename": file.filename,
            "estimated_time": estimated_time,
            "endpoints": {
                "status": f"/status/{job_id}",
                "result": f"/result/{job_id}",
                "download": f"/download/{job_id}",
                "stream": f"/stream/{job_id}"
            }
        }
    )

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get detailed status of a processing job."""
    job_path = os.path.join(WORKSPACE_ROOT, job_id)
    status_file = os.path.join(job_path, "status.json")
    
    if not os.path.exists(status_file):
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    try:
        with open(status_file, "r") as f:
            status_data = json.load(f)
        return JSONResponse(content=status_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read job status: {str(e)}"
        )

@app.get("/result/{job_id}", response_model=SummaryResultResponse)
async def get_summary_result(job_id: str):
    """Retrieve the completed summary with validation."""
    job_path = os.path.join(WORKSPACE_ROOT, job_id)
    status_file = os.path.join(job_path, "status.json")
    
    if not os.path.exists(status_file):
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    # Check status
    try:
        with open(status_file, "r") as f:
            status_data = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read status: {str(e)}"
        )
    
    if status_data["status"] == "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Job failed: {status_data.get('error', 'Unknown error')}"
        )
    
    if status_data["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed yet. Current status: {status_data['status']} ({status_data.get('progress', 0):.1f}%)"
        )
    
    # Read report
    report_file = status_data.get("report_file")
    if not report_file:
        # Try default location
        report_file = os.path.join(job_path, "05_complete_summary.md")
    
    if not os.path.isfile(report_file):
        raise HTTPException(
            status_code=500,
            detail="Report file not found. Processing may have been incomplete."
        )
    
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            markdown = f.read()
        
        # Validate markdown has content
        if len(markdown) < 50:
            raise HTTPException(
                status_code=500,
                detail="Report file appears to be empty or corrupted"
            )
        
        # Calculate processing time
        started_at = status_data.get("started_at")
        completed_at = status_data.get("completed_at")
        processing_time = "Unknown"
        
        if started_at and completed_at:
            try:
                start = datetime.fromisoformat(started_at)
                end = datetime.fromisoformat(completed_at)
                duration = (end - start).total_seconds()
                if duration < 60:
                    processing_time = f"{int(duration)} seconds"
                else:
                    minutes = int(duration / 60)
                    seconds = int(duration % 60)
                    processing_time = f"{minutes}m {seconds}s"
            except:
                pass
        
        return JSONResponse(content={
            "job_id": job_id,
            "status": "completed",
            "title": status_data.get("title", "Document Summary"),
            "markdown": markdown,
            "report_file": report_file,
            "statistics": {
                "pages_processed": status_data.get("pages_processed", 0),
                "total_pages": status_data.get("total_pages", 0),
                "images_analyzed": status_data.get("images_analyzed", 0),
                "total_images": status_data.get("total_images", 0),
                "chunks_created": status_data.get("chunks_created", 0),
                "chunks_summarized": status_data.get("chunks_summarized", 0),
                "api_calls_total": status_data.get("api_calls_total", 0),
                "api_calls_failed": status_data.get("api_calls_failed", 0),
                "file_size_mb": status_data.get("file_size_mb", 0)
            },
            "processing_time": processing_time
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read report: {str(e)}"
        )

@app.get("/download/{job_id}")
async def download_report(job_id: str):
    """Download the summary report as a Markdown file."""
    job_path = os.path.join(WORKSPACE_ROOT, job_id)
    status_file = os.path.join(job_path, "status.json")
    
    if not os.path.exists(status_file):
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    try:
        with open(status_file, "r") as f:
            status_data = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read status: {str(e)}"
        )
    
    if status_data["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed yet. Current status: {status_data['status']}"
        )
    
    report_file = status_data.get("report_file")
    if not report_file:
        report_file = os.path.join(job_path, "05_complete_summary.md")
    
    if not os.path.isfile(report_file):
        raise HTTPException(
            status_code=500,
            detail="Report file not found"
        )
    
    filename = f"summary_{job_id[:8]}_{datetime.now().strftime('%Y%m%d')}.md"
    return FileResponse(
        report_file,
        media_type="text/markdown",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

@app.get("/stream/{job_id}")
async def stream_progress(job_id: str):
    """Stream real-time progress updates using Server-Sent Events (SSE)."""
    job_path = os.path.join(WORKSPACE_ROOT, job_id)
    status_file = os.path.join(job_path, "status.json")
    
    if not os.path.exists(status_file):
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    async def event_generator():
        last_status = None
        consecutive_errors = 0
        max_errors = 5
        
        while consecutive_errors < max_errors:
            try:
                if os.path.exists(status_file):
                    with open(status_file, "r") as f:
                        status_data = json.load(f)
                    
                    # Only send if status changed
                    current_status = json.dumps(status_data, sort_keys=True)
                    if current_status != last_status:
                        yield f"data: {json.dumps(status_data)}\n\n"
                        last_status = current_status
                        consecutive_errors = 0  # Reset error counter
                    
                    # Stop streaming if job completed or failed
                    if status_data.get("status") in ["completed", "failed"]:
                        logger.info(f"Streaming ended for job {job_id}: {status_data.get('status')}")
                        break
                else:
                    consecutive_errors += 1
                    logger.warning(f"Status file not found for job {job_id}, attempt {consecutive_errors}/{max_errors}")
                
                await asyncio.sleep(1)
            
            except json.JSONDecodeError as e:
                consecutive_errors += 1
                logger.error(f"JSON decode error in stream for job {job_id}: {e}")
                await asyncio.sleep(1)
            
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error streaming status for job {job_id}: {e}")
                await asyncio.sleep(1)
        
        # Send final error message if max errors reached
        if consecutive_errors >= max_errors:
            error_msg = {
                "job_id": job_id,
                "status": "error",
                "message": "Stream terminated due to errors",
                "error": "Too many consecutive errors reading status"
            }
            yield f"data: {json.dumps(error_msg)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/jobs")
async def list_jobs(
    status_filter: Optional[str] = None,
    limit: int = 50
):
    """
    List all jobs in the workspace.
    
    Args:
        status_filter: Filter by status (queued, processing, completed, failed)
        limit: Maximum number of jobs to return
    """
    jobs = []
    
    if not os.path.exists(WORKSPACE_ROOT):
        return JSONResponse(content={
            "total_jobs": 0,
            "filtered_jobs": 0,
            "jobs": []
        })
    
    try:
        for job_dir in os.listdir(WORKSPACE_ROOT):
            job_path = os.path.join(WORKSPACE_ROOT, job_dir)
            if not os.path.isdir(job_path):
                continue
                
            status_file = os.path.join(job_path, "status.json")
            
            if os.path.isfile(status_file):
                try:
                    with open(status_file, "r") as f:
                        status_data = json.load(f)
                    
                    # Apply status filter if specified
                    if status_filter and status_data.get("status") != status_filter:
                        continue
                    
                    jobs.append(status_data)
                except Exception as e:
                    logger.warning(f"Failed to load status for job {job_dir}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )
    
    # Sort by most recent first
    jobs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    
    # Apply limit
    total_jobs = len(jobs)
    jobs = jobs[:limit]
    
    return JSONResponse(content={
        "total_jobs": total_jobs,
        "filtered_jobs": len(jobs),
        "active_jobs": len(active_jobs),
        "status_filter": status_filter,
        "limit": limit,
        "jobs": jobs
    })

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and all its artifacts."""
    job_path = os.path.join(WORKSPACE_ROOT, job_id)
    
    if not os.path.exists(job_path):
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    try:
        # Check if job is currently processing
        if job_id in active_jobs:
            logger.warning(f"Attempting to delete active job {job_id}")
            del active_jobs[job_id]
        
        # Remove job directory
        shutil.rmtree(job_path)
        logger.info(f"🗑️ Deleted job {job_id}")
        
        return JSONResponse(content={
            "job_id": job_id,
            "status": "deleted",
            "message": "Job and all artifacts deleted successfully"
        })
    
    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete job: {str(e)}"
        )

@app.post("/cleanup")
async def cleanup_jobs(
    max_age_hours: int = 24,
    status_filter: Optional[str] = "completed"
):
    """
    Clean up jobs older than specified hours.
    
    Args:
        max_age_hours: Age threshold in hours (default: 24)
        status_filter: Only clean jobs with this status (default: completed)
    """
    try:
        cleaned_count = 0
        errors = []
        
        current_time = datetime.now()
        
        for job_dir in os.listdir(WORKSPACE_ROOT):
            job_path = os.path.join(WORKSPACE_ROOT, job_dir)
            if not os.path.isdir(job_path):
                continue
                
            status_file = os.path.join(job_path, "status.json")
            
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                    
                    # Check status filter
                    if status_filter and status.get("status") != status_filter:
                        continue
                    
                    # Check age
                    completed_at = status.get('completed_at') or status.get('last_updated')
                    if completed_at:
                        try:
                            completed_time = datetime.fromisoformat(completed_at)
                            age_hours = (current_time - completed_time).total_seconds() / 3600
                            
                            if age_hours > max_age_hours:
                                shutil.rmtree(job_path, ignore_errors=True)
                                cleaned_count += 1
                                logger.info(f"🗑️ Cleaned up old job: {job_dir} (age: {age_hours:.1f}h)")
                        except ValueError as e:
                            logger.warning(f"Invalid date format for job {job_dir}: {e}")
                
                except Exception as e:
                    errors.append({"job_id": job_dir, "error": str(e)})
                    logger.warning(f"Error processing job {job_dir} for cleanup: {e}")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Cleaned up {cleaned_count} jobs older than {max_age_hours} hours",
            "cleaned_count": cleaned_count,
            "errors": errors,
            "criteria": {
                "max_age_hours": max_age_hours,
                "status_filter": status_filter
            }
        })
    
    except Exception as e:
        logger.error(f"Cleanup operation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )

@app.get("/debug/{job_id}")
async def debug_job(job_id: str):
    """
    Get detailed debug information for a job.
    Useful for troubleshooting issues.
    """
    job_path = os.path.join(WORKSPACE_ROOT, job_id)
    
    if not os.path.exists(job_path):
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    debug_info = {
        "job_id": job_id,
        "job_path": job_path,
        "files": {},
        "directories": {},
        "errors": []
    }
    
    try:
        # List all files in job directory
        for root, dirs, files in os.walk(job_path):
            rel_root = os.path.relpath(root, job_path)
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, job_path)
                
                try:
                    file_size = os.path.getsize(file_path)
                    file_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    
                    debug_info["files"][rel_path] = {
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 3),
                        "modified": file_modified,
                        "exists": True
                    }
                except Exception as e:
                    debug_info["errors"].append({
                        "file": rel_path,
                        "error": str(e)
                    })
            
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                rel_path = os.path.relpath(dir_path, job_path)
                
                try:
                    file_count = len(os.listdir(dir_path))
                    debug_info["directories"][rel_path] = {
                        "file_count": file_count,
                        "exists": True
                    }
                except Exception as e:
                    debug_info["errors"].append({
                        "directory": rel_path,
                        "error": str(e)
                    })
        
        # Check for expected files
        expected_files = [
            "status.json",
            "01_extracted_pages.json",
            "02_image_metadata.json",
            "03_text_chunks.json",
            "04_final_text_summary.md",
            "05_complete_summary.md"
        ]
        
        debug_info["expected_files_status"] = {}
        for expected_file in expected_files:
            file_path = os.path.join(job_path, expected_file)
            debug_info["expected_files_status"][expected_file] = {
                "exists": os.path.exists(file_path),
                "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
        
        # Load status if available
        status_file = os.path.join(job_path, "status.json")
        if os.path.exists(status_file):
            try:
                with open(status_file, "r") as f:
                    debug_info["status_data"] = json.load(f)
            except Exception as e:
                debug_info["errors"].append({
                    "file": "status.json",
                    "error": f"Failed to parse JSON: {str(e)}"
                })
        
        # Load processing log if available
        log_file = os.path.join(job_path, "processing.log")
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    log_lines = f.readlines()
                    debug_info["log_preview"] = {
                        "total_lines": len(log_lines),
                        "first_10_lines": log_lines[:10],
                        "last_10_lines": log_lines[-10:] if len(log_lines) > 10 else log_lines
                    }
            except Exception as e:
                debug_info["errors"].append({
                    "file": "processing.log",
                    "error": str(e)
                })
        
        return JSONResponse(content=debug_info)
    
    except Exception as e:
        logger.error(f"Debug operation failed for job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Debug operation failed: {str(e)}"
        )

# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("=" * 80)
    logger.info("🚀 AI Document Summarizer Pro v3.0.1-FIXED - Starting Up")
    logger.info("=" * 80)
    logger.info("🔧 Configuration:")
    logger.info(f"   📍 KGPT URL: {KGPT_URL}")
    logger.info(f"   🤖 Model: {KGPT_MODEL}")
    logger.info(f"   📁 Workspace: {WORKSPACE_ROOT}")
    logger.info(f"   🔢 Max tokens per chunk: {MAX_TOKENS_PER_CHUNK}")
    logger.info(f"   📊 Max file size: {MAX_FILE_SIZE_MB} MB")
    logger.info(f"   🔄 Max retries: 3")
    logger.info(f"   ⏱️  API timeout: 180s")
    logger.info("=" * 80)
    logger.info("🔧 Fixes Applied:")
    logger.info("   ✅ Multi-level fallback summary creation")
    logger.info("   ✅ API retry mechanism with backoff")
    logger.info("   ✅ File validation at each step")
    logger.info("   ✅ Emergency text-only summaries")
    logger.info("   ✅ Enhanced error logging")
    logger.info("=" * 80)
    logger.info(f"🌐 Server: http://0.0.0.0:8000")
    logger.info(f"📖 API Docs: http://localhost:8000/docs")
    logger.info(f"📖 ReDoc: http://localhost:8000/redoc")
    logger.info("=" * 80)
    
    # Test API connection on startup
    try:
        import requests
        response = requests.post(
            f"{KGPT_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": KGPT_MODEL,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            },
            timeout=10
        )
        if response.status_code == 200:
            logger.info("✅ KGPT API connection successful")
        else:
            logger.warning(f"⚠️ KGPT API returned status {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to KGPT API: {e}")
        logger.warning("⚠️ Server will start but API calls may fail")
    
    logger.info("=" * 80)
    logger.info("✅ Server ready to accept requests")
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("=" * 80)
    logger.info("🛑 Shutting down AI Document Summarizer Pro")
    logger.info("=" * 80)
    
    # Log active jobs
    if active_jobs:
        logger.warning(f"⚠️ {len(active_jobs)} jobs still active:")
        for job_id in active_jobs.keys():
            logger.warning(f"   - {job_id}")
    
    # Clean up active jobs
    for job_id in list(active_jobs.keys()):
        del active_jobs[job_id]
    
    logger.info("✅ Cleanup completed")
    logger.info("=" * 80)

# Main
if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs(WORKSPACE_ROOT, exist_ok=True)
    
    # Validate environment variables
    if not API_TOKEN:
        logger.error("❌ API_TOKEN environment variable not set!")
        logger.error("   Please set API_TOKEN before starting the server")
        exit(1)
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        timeout_keep_alive=300  # 5 minutes for long-running requests
    )