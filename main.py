from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fastapi_cdn_host
from contextlib import asynccontextmanager
import uvicorn
import os
import traceback
from datetime import datetime

from src.database.database_config import init_database
from src.api.workflow_router import workflow_router
from src.utils.logger_config import LoggerConfig, get_logger

# Setup logging
logger = LoggerConfig.setup(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir="logs"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 启动签证自动填表工作流系统...")
    logger.info(f"📊 数据库类型: {'PostgreSQL' if 'postgresql' in os.getenv('DATABASE_URL', '') else 'SQLite'}")
    
    # Initialize database
    try:
        init_database()
        logger.info("✅ 数据库初始化成功")
    except Exception as e:
        LoggerConfig.log_exception(logger, "数据库初始化失败", e)
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 关闭签证自动填表工作流系统...")

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
    """Global exception handler with comprehensive logging"""
    # Log detailed error information
    logger.error("=" * 50)
    logger.error(f"🚨 UNHANDLED EXCEPTION OCCURRED")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {str(exc)}")
    logger.error(f"Full traceback:")
    logger.error(traceback.format_exc())
    
    # Log request headers (excluding sensitive information)
    headers_to_log = {}
    for key, value in request.headers.items():
        if key.lower() not in ['authorization', 'cookie', 'x-api-key']:
            headers_to_log[key] = value
        else:
            headers_to_log[key] = "[REDACTED]"
    logger.error(f"Request headers: {headers_to_log}")
    
    # Try to log request body if available and not too large
    try:
        if hasattr(request, '_body') and request._body:
            body_str = request._body.decode('utf-8')
            if len(body_str) < 1000:  # Only log small bodies
                logger.error(f"Request body: {body_str}")
            else:
                logger.error(f"Request body: [TOO LARGE - {len(body_str)} bytes]")
    except Exception as body_error:
        logger.error(f"Could not log request body: {body_error}")
    
    logger.error("=" * 50)
    
    # Return JSON response to client
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__,
            "timestamp": datetime.utcnow().isoformat()
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
    
    # Configure uvicorn logging
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "watchfiles": {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "watchfiles.main": {"handlers": ["default"], "level": "ERROR", "propagate": False},
        },
        "root": {
            "level": "INFO",
            "handlers": ["default"],
        },
    }
    
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8080")),
        reload=os.getenv("APP_RELOAD", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        log_config=log_config
    ) 