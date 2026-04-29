from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from galaxy_tool_recommendation_ai_agent.build_kb import main


if __name__ == "__main__":
    main()
