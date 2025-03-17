from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
class DatabaseConfig:
    """Database configuration handler with environment-specific settings."""
    
    def __init__(self):
        self.env = os.getenv("ENVIRONMENT", "development")
        self.db_params = self._get_db_params()
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.Base = declarative_base()
    
    def _get_db_params(self) -> dict:
        """Get database parameters based on environment."""
        params = {
            "development": {
                "host": os.getenv("DEV_DB_HOST", "localhost"),
                "port": os.getenv("DEV_DB_PORT", "5432"),
                "database": os.getenv("DEV_DB_NAME", "hypersyncdt_dev"),
                "user": os.getenv("DEV_DB_USER", "postgres"),
                "password": os.getenv("DEV_DB_PASSWORD", "postgres")
            },
            "testing": {
                "host": os.getenv("TEST_DB_HOST", "localhost"),
                "port": os.getenv("TEST_DB_PORT", "5432"),
                "database": os.getenv("TEST_DB_NAME", "hypersyncdt_test"),
                "user": os.getenv("TEST_DB_USER", "postgres"),
                "password": os.getenv("TEST_DB_PASSWORD", "postgres")
            },
            "production": {
                "host": os.getenv("PROD_DB_HOST"),
                "port": os.getenv("PROD_DB_PORT"),
                "database": os.getenv("PROD_DB_NAME"),
                "user": os.getenv("PROD_DB_USER"),
                "password": os.getenv("PROD_DB_PASSWORD")
            }
        }
        
        if self.env not in params:
            raise ValueError(f"Invalid environment: {self.env}")
        
        return params[self.env]
    
    def _create_engine(self):
        """Create SQLAlchemy engine with appropriate configuration."""
        db_url = (
            f"postgresql://{self.db_params['user']}:{self.db_params['password']}"
            f"@{self.db_params['host']}:{self.db_params['port']}"
            f"/{self.db_params['database']}"
        )
        
        return create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=self.env == "development"
        )
    
    def get_db(self):
        """Database session dependency."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def verify_connection(self) -> bool:
        """Verify database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info(f"Successfully connected to {self.env} database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.env} database: {str(e)}")
            return False
    
    def create_tables(self):
        """Create all defined tables."""
        try:
            self.Base.metadata.create_all(bind=self.engine)
            logger.info("Successfully created database tables")
        except Exception as e:
            logger.error(f"Failed to create database tables: {str(e)}")
            raise

# Create database configuration instance
db_config = DatabaseConfig()

# Example usage
if __name__ == "__main__":
    # Verify database connection
    if db_config.verify_connection():
        print(f"Connected to {db_config.env} database at {db_config.db_params['host']}")
    else:
        print("Failed to connect to database") 