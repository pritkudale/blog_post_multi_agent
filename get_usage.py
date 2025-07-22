#!/usr/bin/env python3
"""
Quick model usage checker - run this anytime to see current model usage info
"""

import sys
import os

# Add paths to find the modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from blog.tools.model_tracker import print_usage_summary, track_crew_execution

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "track":
        # Run crew and track usage
        topic = sys.argv[2] if len(sys.argv) > 2 else "AI LLMs"
        track_crew_execution(topic)
    else:
        # Just show current usage info
        print_usage_summary()
