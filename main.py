from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fastapi_cdn_host
from contextlib import asynccontextmanager
import uvicorn
import os

from src.database.database_config import init_database
from src.api.workflow_router import workflow_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 启动签证自动填表工作流系统...")
    print(f"📊 数据库类型: {'PostgreSQL' if 'postgresql' in os.getenv('DATABASE_URL', '') else 'SQLite'}")
    
    # Initialize database
    try:
        init_database()
    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")
        raise
    
    yield
    
    # Shutdown
    print("🛑 关闭签证自动填表工作流系统...")

# Create FastAPI application
app = FastAPI(
    title="签证自动填表工作流系统",
    description="""
    ## 英国签证申请自动填表系统
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)
fastapi_cdn_host.patch_docs(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(workflow_router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "签证自动填表工作流系统",
        "version": "1.0.0",
        "database": "PostgreSQL" if "postgresql" in os.getenv("DATABASE_URL", "") else "SQLite"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "欢迎使用签证自动填表工作流系统",
        "description": "基于AI的12步签证申请自动化系统",
        "features": [
            "12步工作流管理",
            "LangGraph AI表单分析",
            "文档自动解析", 
            "智能数据映射",
            "实时状态跟踪",
            "PostgreSQL数据库支持"
        ],
        "docs": "/docs",
        "health": "/health",
        "database": "PostgreSQL" if "postgresql" in os.getenv("DATABASE_URL", "") else "SQLite"
    }

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "6011")),
        reload=os.getenv("APP_RELOAD", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    ) 