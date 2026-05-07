# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:10:29 2026

@author: Morgan
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))