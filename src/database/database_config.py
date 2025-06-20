from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import os
from contextlib import contextmanager
from typing import Generator

from src.utils.logger_config import get_logger

logger = get_logger(__name__)

class DatabaseConfig:
    """Database configuration class"""
    
    def __init__(self):
        logger.info("Initializing database configuration")
        
        # PostgreSQL configuration from environment variables
        postgres_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
        postgres_port = os.getenv("POSTGRES_PORT", "5432")
        postgres_db = os.getenv("POSTGRES_DB", "postgres")
        postgres_user = os.getenv("POSTGRES_USER", "postgres")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        
        # Construct PostgreSQL URL with workflow schema
        postgres_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}?options=-csearch_path%3Dworkflow"
        
        # Use PostgreSQL by default, fallback to SQLite for development
        self.database_url = os.getenv("DATABASE_URL", postgres_url)
        logger.info(f"Database URL configured: {self.database_url.split('@')[1] if '@' in self.database_url else self.database_url}")
        
        # Engine configuration
        self.engine = create_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
        )
        
        # Session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info("Database configuration initialized successfully")
    
    def create_schema(self):
        """Create workflow schema if it doesn't exist"""
        try:
            logger.info("Creating workflow schema if it doesn't exist")
            with self.engine.connect() as connection:
                # Create workflow schema
                connection.execute(text("CREATE SCHEMA IF NOT EXISTS workflow"))
                connection.commit()
                logger.info("✅ Workflow schema created/verified")
        except Exception as e:
            logger.error(f"❌ Schema creation failed: {e}")
            raise
    
    def create_tables(self):
        """Create all database tables in workflow schema"""
        try:
            logger.info("Creating database tables")
            from src.model.workflow_entities import Base
            
            # First create the schema
            self.create_schema()
            
            # Then create tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database tables created successfully")
        except Exception as e:
            logger.error(f"❌ Table creation failed: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables"""
        try:
            logger.warning("Dropping all database tables")
            from src.model.workflow_entities import Base
            Base.metadata.drop_all(bind=self.engine)
            logger.info("✅ Database tables dropped successfully")
        except Exception as e:
            logger.error(f"❌ Table dropping failed: {e}")
            raise
    
    def test_connection(self):
        """Test database connection"""
        try:
            logger.info("Testing database connection")
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                logger.info("✅ Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            logger.debug("Database session created")
            yield session
            session.commit()
            logger.debug("Database session committed successfully")
        except Exception as e:
            logger.error(f"Database session error, rolling back: {e}")
            session.rollback()
            raise
        finally:
            session.close()
            logger.debug("Database session closed")

# Global database instance
db_config = DatabaseConfig()

def get_db_session() -> Generator[Session, None, None]:
    """Dependency for getting database session"""
    with db_config.get_session() as session:
        yield session

def init_database():
    """Initialize database tables"""
    logger.info("Initializing database...")
    try:
        if db_config.test_connection():
            logger.info("✅ Database connection successful")
            db_config.create_tables()
            logger.info("✅ Database tables created successfully")
        else:
            error_msg = "❌ Database connection failed"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise 