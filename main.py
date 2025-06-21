import os
import traceback
from contextlib import asynccontextmanager
from datetime import datetime

import fastapi_cdn_host
import hypercorn.asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from hypercorn.config import Config

from src.api.workflow_router import workflow_router
from src.database.database_config import init_database
from src.utils.logger_config import LoggerConfig

"""
签证自动填表工作流系统

"""

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
    import asyncio

    load_dotenv()

    # IPv4/IPv6 双栈配置
    # Railway 使用 PORT 环境变量，本地开发使用 APP_PORT
    port = int(os.getenv("PORT", os.getenv("APP_PORT", "8080")))

    # Start the server
    print(f"🚀 启动签证自动填表工作流系统...")
    print(f"📊 服务地址: http://localhost:{port}")
    print(f"📚 API文档: http://localhost:{port}/docs")
    print(f"🔍 健康检查: http://localhost:{port}/health")
    print("按 Ctrl+C 停止服务")

    # Configure and run with hypercorn
    config = Config()

    is_railway = os.getenv("IS_RAILWAY", "true").lower() == "true"

    if is_railway:
        config.bind = [
            f"[::]:{port}"  # IPv6
        ]
    else:
        config.bind = [
            f"0.0.0.0:{port}",  # IPv4
            f"[::]:{port}"  # IPv6
        ]

    config.application_path = "main:app"

    config.reload = os.getenv("APP_RELOAD", "true").lower() == "true"
    config.log_level = os.getenv("LOG_LEVEL", "trace").lower()
    config.workers = 1

    config.worker_class = "asyncio"
    config.access_logfile = "-"
    config.error_logfile = "-"

    # Run with hypercorn
    asyncio.run(hypercorn.asyncio.serve(app, config))
