"""AICA backend package."""

from dotenv import load_dotenv

# Loaded HERE, not in settings.py, because module-level os.getenv() calls run at
# IMPORT time and several modules that have them (clause_chunker, conversation,
# main) are imported BEFORE settings.py is. Loading .env from the package root
# means every `import backend.*` - run.sh, bare uvicorn, a script, a test - has
# it before any module body executes. Existing process env always wins.
load_dotenv()
