import re
import subprocess
from typing import List


def list_tiger_vnc_displays() -> List[str]:
    """
    Lists all available TigerVNC displays using `tigervncserver -list`.
    Returns:
        A list of display numbers (e.g., [:2, :3]).
    """
    try:
        # Run the `tigervncserver -list` command
        result = subprocess.run(
            ["tigervncserver", "-list"], capture_output=True, text=True, check=True
        )
        output = result.stdout

        # Regex to find display numbers (e.g., :2)
        display_regex = re.compile(r"(:\d+)\s+\d+")
        displays = [str(match.group(1)) for match in display_regex.finditer(output)]

        return sorted(displays)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run 'tigervncserver -list': {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while listing displays: {e}")


if __name__ == "__main__":
    print(list_tiger_vnc_displays())
