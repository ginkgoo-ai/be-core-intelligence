import asyncio
import json
import threading
import uuid
from typing import Optional, List
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src import graph

# Load environment variables
load_dotenv()

app = FastAPI(
    title="签证自动填表工作流系统",
    description="基于AI的英国签证申请自动化系统",
    version="1.0.0"
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    print("🚀 启动签证自动填表工作流系统...")
    
    # Initialize database if using PostgreSQL
    try:
        from src.database.database_config import init_database
        init_database()
        print("✅ 数据库初始化成功")
    except Exception as e:
        print(f"⚠️  数据库初始化失败: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "6011")),
        reload=os.getenv("APP_RELOAD", "true").lower() == "true"
    )

class MessageRequest(BaseModel):
    message: str
    trace_id: Optional[str] = None
    fill_data: Optional[dict] = None


class FileRequest(BaseModel):
    doc_urls: List[str]
    format_json: Optional[str] = None

thread_local = threading.local()


@app.post("/assistant")
async def run_workflow(message: MessageRequest):
    """Run AI workflow for form processing"""
    # 直接调用langgraph应用
    if message.trace_id is None:
        message.trace_id = str(uuid.uuid4())

    graph.global_static_map.update({message.trace_id: message.fill_data})

    msg = graph.system_message.format_messages(messages=message.message)
    result = graph.app.invoke(
        {"messages": msg},
        config={"configurable": {"thread_id": message.trace_id}}
    )
    last_message = result["messages"][-1]
    content = last_message.content
    if not content:
        return {"error": "Content is empty"}

    try:
        json_data = json.loads(content) if isinstance(content, str) else content

    except json.JSONDecodeError:
        return {"error": "Invalid JSON in content"}

    graph.global_static_map.pop(message.trace_id)
    return {
        "result": json_data
    }

@app.post("/files/classify")
async def analyze_image_classify_endpoint(
        file_request: FileRequest,
):
    """Classify document types"""
    try:
        if not file_request.doc_urls:
             return JSONResponse(status_code=400, content={"message": "No document URLs provided"})
        doc_url_to_classify = file_request.doc_urls[0]

        # Consider making API key an environment variable
        classifier = await DocumentClassifier(doc_url_to_classify).classify_document()

        return classifier

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": f"Error performing classification : {str(e)}"}
        )


@app.post("/files/structure")
async def files_structure_endpoint(
    file_request: FileRequest,
):
    """Extract structured data from documents"""
    tasks = []
    # Create a task for each URL
    for doc_url in file_request.doc_urls:
        tasks.append(DocumentResolver(doc_url).resolver())

    try:
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined_results = {}
        # Combine results from all successful tasks, filling fields from the first non-null value
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Log or handle error for specific URL
                print(f"Error processing URL {file_request.doc_urls[i]}: {result}")
            elif isinstance(result, dict):
                for field, value in result.items():
                    if field not in combined_results and value is not None:
                        combined_results[field] = value

        if not combined_results:
             return JSONResponse(status_code=404, content={"message": "No non-null fields could be extracted or all tasks failed."})

        return combined_results

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": f"Error processing documents: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "签证自动填表工作流系统",
        "version": "1.0.0",
        "database": "PostgreSQL" if "postgresql" in os.getenv("DATABASE_URL", "") else "SQLite"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "欢迎使用签证自动填表工作流系统",
        "description": "基于AI的英国签证申请自动化系统",
        "features": [
            "文档分类和结构化提取",
            "AI表单处理工作流",
            "PostgreSQL数据库支持"
        ],
        "docs": "/docs",
        "health": "/health"
    }
