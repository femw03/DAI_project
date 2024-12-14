from typing import List, Optional, Tuple

from .wrappers import CarlaWaypoint


def find_next_wp_from(
    waypoints: List[CarlaWaypoint], min_distance=10
) -> Optional[Tuple[CarlaWaypoint, int]]:
    """
    Returns the first waypoint that is at least min_distance away from the first waypoint and it's index
    returns none if such a wp cannot be found
    """
    if len(waypoints) == 0:
        return None
    first_location = waypoints[0].location
    for i, waypoint in enumerate(waypoints):
        if first_location.distance_to(waypoint.location) > min_distance:
            return waypoint, i

    return None
