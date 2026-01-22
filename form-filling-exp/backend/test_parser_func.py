import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to sys.path to make imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parser import parse_file

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_parser_func.py <path_to_file> [api_key]")
        return

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    api_key = None
    if len(sys.argv) > 2:
        api_key = sys.argv[2]
    
    # If no API key provided in args, check env var
    if not api_key:
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY")

    print(f"Reading file: {file_path}")
    file_bytes = file_path.read_bytes()
    
    print(f"Parsing file ({len(file_bytes)} bytes)...")
    try:
        result = await parse_file(file_bytes, file_path.name, api_key=api_key)
        print("\n" + "="*40)
        print("PARSING RESULT")
        print("="*40 + "\n")
        print(result)
        print("\n" + "="*40)
    except Exception as e:
        print(f"\nError during parsing: {e}")

if __name__ == "__main__":
    asyncio.run(main())
