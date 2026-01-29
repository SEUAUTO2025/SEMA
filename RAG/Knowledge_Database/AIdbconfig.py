"""
@filename: dbconfig.py
@description: AI database file interface for loading AI database
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Get absolute path of current script file (AIdbconfig.py)
script_path = os.path.abspath(__file__)
# Get directory containing the script
script_dir = os.path.dirname(script_path)
# Get project root directory (parent of script directory)
project_root = os.path.dirname(script_dir)
# Construct absolute path to database file
db_path = os.path.join(project_root, "db_files", "LLM_Knowledge_New_ali_zh.db")

DATABASE_URL = 'sqlite:///' + db_path
engine = create_engine(DATABASE_URL, echo=False)

# Create Session class
session = sessionmaker(bind=engine)()
