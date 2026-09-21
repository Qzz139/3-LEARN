"""Load the tested controller from the package's installed legacy directory."""
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory


SHARE = Path(get_package_share_directory('ep_sorting'))
sys.path.insert(0, str(SHARE / 'legacy'))

import ir_pick_place_demo as base  # noqa: E402
import yolo_align_pick_place_demo as align  # noqa: E402
import object_sorting as sorting  # noqa: E402
from sorting_state_machine import SortingStateMachine  # noqa: E402
