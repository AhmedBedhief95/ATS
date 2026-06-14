import os
from dotenv import load_dotenv

# Try to load .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Looking for .env at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")

load_dotenv(env_path)

print("\n--- Loaded Environment Variables ---")
print(f"DB_HOST: '{os.getenv('DB_HOST')}'")
print(f"DB_USER: '{os.getenv('DB_USER')}'")
print(f"DB_NAME: '{os.getenv('DB_NAME')}'")
print(f"DB_PASSWORD: '{'*' * len(os.getenv('DB_PASSWORD', ''))}'")
print(f"GROQ_API_KEY: '{os.getenv('GROQ_API_KEY')[:10]}...'")
print("------------------------------------\n")