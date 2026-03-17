"""Entry point for the trading bot."""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from bot.main import main

if __name__ == "__main__":
    main()
