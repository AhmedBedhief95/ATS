import os
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
# Specify the path explicitly to ensure it loads
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print("DB_HOST from env:", os.getenv('DB_HOST'))
    print(f"✅ Loaded environment variables from {env_path}")
else:
    print(f"⚠️ Warning: .env file not found at {env_path}")
    print("   Using system environment variables or defaults")
    load_dotenv()

class Config:
    """Application configuration loaded from environment variables"""
    
    # Database Configuration
    DB_NAME = os.getenv('DB_NAME', 'postgres')
    DB_USER = os.getenv('DB_USER', '')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_HOST = os.getenv('DB_HOST', '')
    DB_PORT = os.getenv('DB_PORT', '5432')
    
    # API Keys
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    
    # Application Settings
    APP_HOST = os.getenv('APP_HOST', '127.0.0.1')
    APP_PORT = int(os.getenv('APP_PORT', '8000'))
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    # Model Settings
    MODEL_DIR = os.getenv('MODEL_DIR', 'models')
    
    # Upload Settings
    UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'uploads')
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '10485760'))
    
    # CPI
    CPI_CLIENT_ID = os.getenv('CPI_CLIENT_ID')
    CPI_CLIENT_SECRET = os.getenv('CPI_CLIENT_SECRET')
    CPI_BASE_URL = os.getenv('CPI_BASE_URL')
    
    @property
    def DB_CONFIG(self):
        """Database connection configuration dictionary"""
        return {
            "dbname": self.DB_NAME,
            "user": self.DB_USER,
            "password": self.DB_PASSWORD,
            "host": self.DB_HOST,
            "port": self.DB_PORT
        }
    
    @classmethod
    def validate(cls):
        """Validate required configuration is present"""
        print("\n" + "="*50)
        print("📋 Validating Configuration")
        print("="*50)
        
        # Check database configuration
        if cls.DB_HOST and cls.DB_HOST != 'your_db_host':
            print(f"✅ Database host: {cls.DB_HOST}")
        else:
            print("❌ ERROR: Invalid DB_HOST. Please check your .env file")
            return False
        
        if cls.DB_USER and cls.DB_USER != 'your_db_user':
            print(f"✅ Database user: {cls.DB_USER}")
        else:
            print("❌ ERROR: Invalid DB_USER. Please check your .env file")
            return False
        
        if cls.DB_PASSWORD and cls.DB_PASSWORD != 'your_db_password':
            print("✅ Database password: [SET]")
        else:
            print("❌ ERROR: Invalid DB_PASSWORD. Please check your .env file")
            return False
        
        # Check API keys (optional but warn if missing)
        if cls.GROQ_API_KEY and cls.GROQ_API_KEY != 'your_groq_api_key_here':
            print("✅ Groq API key: [SET]")
        else:
            print("⚠️  Warning: GROQ_API_KEY not set. Groq model will not work.")
        
        if cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != 'your_gemini_api_key_here':
            print("✅ Gemini API key: [SET]")
        else:
            print("⚠️  Warning: GEMINI_API_KEY not set. Gemini model will not work.")
        
        print("="*50 + "\n")
        return True

# Create a singleton instance
config = Config()

# Auto-validate on import
if __name__ != "__main__":
    config.validate()