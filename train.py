from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

db_password = os.getenv('DB_PASSWORD')
api_key = os.getenv('API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "" # <api-key>
os.environ['OPENAI_API_KEY']    = "" # <api-key>


