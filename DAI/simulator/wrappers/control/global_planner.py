# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


"""
This module provides GlobalRoutePlanner implementation.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from loguru import logger
from tqdm import tqdm

from ..carla_core import CarlaMap
from ..carla_utils import (
    CarlaLaneChange,
    CarlaLaneType,
    CarlaLocation,
    CarlaVector3D,
    CarlaWaypoint,
)
from .local_planner import RoadOption


@dataclass
class Edge:
    length: int
    path: List[CarlaWaypoint]
    entry_wp: CarlaWaypoint
    exit_wp: CarlaWaypoint
    entry_vector: CarlaVector3D
    exit_vector: CarlaVector3D
    net_vector: CarlaVector3D
    is_intersection: bool
    type: RoadOption
    change_wp: Optional[CarlaWaypoint] = None


@dataclass
class TopologySegment:
    entry: CarlaWaypoint
    entryxyz: CarlaLocation
    exit: CarlaWaypoint
    exitxyz: CarlaLocation
    path: List[CarlaWaypoint]


class GlobalRoutePlanner(object):
    """
    This class provides a very high level route plan.
    """

    def __init__(self, wmap: CarlaMap, sampling_resolution=2.0):
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap
        self._topology = None
        self._graph = None
        self._id_map: Dict[CarlaLocation, int] = dict()
        self._road_id_to_edge: Dict[int, Dict[int, Tuple[int, int]]] = dict()

        self._intersection_end_node = -1
        self._previous_decision = RoadOption.VOID

        # Build the graph
        self._build_topology()
        logger.info("Finsihed Building topology")

        self._build_graph()
        logger.info("Finsihed Connecting graph")

        self._find_loose_ends()
        logger.info("Finished Fixing the loose ends")

        self._lane_change_link()
        logger.info("Finished Adding lane changes")

    def trace_route(
        self, origin: CarlaLocation, destination: CarlaLocation
    ) -> List[Tuple[CarlaWaypoint, RoadOption]]:
        """
        This method returns list of (carla.Waypoint, RoadOption)
        from origin to destination
        """
        route_trace: List[Tuple[CarlaWaypoint, RoadOption]] = []
        route = self._path_search(origin, destination)
        current_waypoint = self._wmap.get_waypoint(origin)
        destination_waypoint = self._wmap.get_waypoint(destination)

        for i in range(len(route) - 1):
            road_option = self._turn_decision(i, route)
            edge = self._graph.edges[route[i], route[i + 1]]
            edge_data: Edge = edge["data"]
            path: List[CarlaWaypoint] = []

            if (
                edge_data.type != RoadOption.LANEFOLLOW
                and edge_data.type != RoadOption.VOID
            ):
                route_trace.append((current_waypoint, road_option))
                exit_wp = edge_data.exit_wp
                n1, n2 = self._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][
                    exit_wp.lane_id
                ]
                next_edge = self._graph.edges[n1, n2]
                next_edge_data: Edge = next_edge["data"]
                if next_edge_data.path:
                    closest_index = self._find_closest_in_list(
                        current_waypoint, next_edge_data.path
                    )
                    closest_index = min(len(next_edge_data.path) - 1, closest_index + 5)
                    current_waypoint = next_edge_data.path[closest_index]
                else:
                    current_waypoint = next_edge_data.exit_wp
                route_trace.append((current_waypoint, road_option))

            else:
                path = (
                    path + [edge_data.entry_wp] + edge_data.path + [edge_data.exit_wp]
                )
                closest_index = self._find_closest_in_list(current_waypoint, path)
                for waypoint in path[closest_index:]:
                    current_waypoint = waypoint
                    route_trace.append((current_waypoint, road_option))
                    if (
                        len(route) - i <= 2
                        and waypoint.location.distance_to(destination)
                        < 2 * self._sampling_resolution
                    ):
                        break
                    elif (
                        len(route) - i <= 2
                        and current_waypoint.road_id == destination_waypoint.road_id
                        and current_waypoint.section_id
                        == destination_waypoint.section_id
                        and current_waypoint.lane_id == destination_waypoint.lane_id
                    ):
                        destination_index = self._find_closest_in_list(
                            destination_waypoint, path
                        )
                        if closest_index > destination_index:
                            break

        return route_trace

    def _build_topology(self):
        """
        This function retrieves topology from the server as a list of
        road segments as pairs of waypoint objects, and processes the
        topology into a list of dictionary objects with the following attributes

        - entry (carla.Waypoint): waypoint of entry point of road segment
        - entryxyz (tuple): (x,y,z) of entry point of road segment
        - exit (carla.Waypoint): waypoint of exit point of road segment
        - exitxyz (tuple): (x,y,z) of exit point of road segment
        - path (list of carla.Waypoint):  list of waypoints between entry to exit, separated by the resolution
        """
        self._topology: List[TopologySegment] = []
        # Retrieving waypoints to construct a detailed topology
        for segment in tqdm(self._wmap.get_topology(), desc="Building topology"):
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = (
                CarlaLocation.from_native(wp1.transform.location),
                CarlaLocation.from_native(wp2.transform.location),
            )
            # Rounding off to avoid floating point imprecision
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1.native, l2.native
            segment = TopologySegment(
                entry=wp1,
                exit=wp2,
                entryxyz=CarlaLocation(x1, y1, z1),
                exitxyz=CarlaLocation(x2, y2, z2),
                path=[],
            )
            endloc = CarlaLocation.from_native(wp2.transform.location)
            if l1.distance_to(endloc) > self._sampling_resolution:
                w = wp1.get_next(self._sampling_resolution)[0]
                distance = CarlaLocation.from_native(w.transform.location).distance_to(
                    endloc
                )
                while distance > self._sampling_resolution:
                    segment.path.append(w)
                    next_ws = w.get_next(self._sampling_resolution)
                    if len(next_ws) == 0:
                        break
                    w = next_ws[0]
                    distance = CarlaLocation.from_native(
                        w.transform.location
                    ).distance_to(endloc)

            else:
                next_wps = wp1.get_next(self._sampling_resolution)
                if len(next_wps) == 0:
                    continue
                segment.path.append(next_wps[0])
            self._topology.append(segment)

    def _build_graph(self):
        """
        This function builds a networkx graph representation of topology, creating several class attributes:
        - graph (networkx.DiGraph): networkx graph representing the world map, with:
            Node properties:
                vertex: (x,y,z) position in world map
            Edge properties:
                entry_vector: unit vector along tangent at entry point
                exit_vector: unit vector along tangent at exit point
                net_vector: unit vector of the chord from entry to exit
                intersection: boolean indicating if the edge belongs to an  intersection
        - id_map (dictionary): mapping from (x,y,z) to node id
        - road_id_to_edge (dictionary): map from road id to edge in the graph
        """

        self._graph = nx.DiGraph()
        # Map with structure {(x,y,z): id, ... }

        for segment in tqdm(self._topology, desc="Connecting graph"):
            entry_xyz, exit_xyz = segment.entryxyz, segment.exitxyz
            path = segment.path
            entry_wp, exit_wp = segment.entry, segment.exit
            is_intersection = entry_wp.is_junction
            road_id, section_id, lane_id = (
                entry_wp.road_id,
                entry_wp.section_id,
                entry_wp.lane_id,
            )

            for vertex in entry_xyz, exit_xyz:
                # Adding unique nodes and populating id_map
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    self._graph.add_node(new_id, vertex=vertex)
            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_carla_vector = CarlaVector3D(
                entry_wp.transform.rotation.get_forward_vector()
            )
            exit_carla_vector = CarlaVector3D(
                exit_wp.transform.rotation.get_forward_vector()
            )

            # Adding edge with attributes
            self._graph.add_edge(
                n1,
                n2,
                data=Edge(
                    length=len(path) + 1,
                    path=path,
                    entry_wp=entry_wp,
                    exit_wp=exit_wp,
                    entry_vector=CarlaVector3D.fromxyz(
                        entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z
                    ),
                    exit_vector=CarlaVector3D.fromxyz(
                        exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z
                    ),
                    net_vector=CarlaLocation.from_native(
                        entry_wp.transform.location
                    ).vector_to(
                        CarlaLocation.from_native(exit_wp.transform.location),
                    ),
                    is_intersection=is_intersection,
                    type=RoadOption.LANEFOLLOW,
                ),
            )

    def _find_loose_ends(self) -> None:
        """
        This method finds road segments that have an unconnected end, and
        adds them to the internal graph representation
        """
        count_loose_ends = 0
        hop_resolution = self._sampling_resolution
        for segment in tqdm(self._topology, desc="Fixing loose ends"):
            end_wp = segment.exit
            exit_xyz = segment.exitxyz
            road_id, section_id, lane_id = (
                end_wp.road_id,
                end_wp.section_id,
                end_wp.lane_id,
            )
            if (
                road_id in self._road_id_to_edge
                and section_id in self._road_id_to_edge[road_id]
                and lane_id in self._road_id_to_edge[road_id][section_id]
            ):
                pass
            else:
                count_loose_ends += 1
                if road_id not in self._road_id_to_edge:
                    self._road_id_to_edge[road_id] = dict()
                if section_id not in self._road_id_to_edge[road_id]:
                    self._road_id_to_edge[road_id][section_id] = dict()
                n1 = self._id_map[exit_xyz]
                n2 = -1 * count_loose_ends
                self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
                next_wp = end_wp.get_next(hop_resolution)
                path = []
                while (
                    next_wp is not None
                    and next_wp
                    and next_wp[0].road_id == road_id
                    and next_wp[0].section_id == section_id
                    and next_wp[0].lane_id == lane_id
                ):
                    path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)
                if path:
                    n2_xyz = CarlaLocation(
                        path[-1].transform.location.x,
                        path[-1].transform.location.y,
                        path[-1].transform.location.z,
                    )
                    self._graph.add_node(n2, vertex=n2_xyz)
                    self._graph.add_edge(
                        n1,
                        n2,
                        data=Edge(
                            length=len(path) + 1,
                            path=path,
                            entry_wp=end_wp,
                            exit_wp=path[-1],
                            entry_vector=None,
                            exit_vector=None,
                            net_vector=None,
                            is_intersection=end_wp.is_junction,
                            type=RoadOption.LANEFOLLOW,
                        ),
                    )

    def _lane_change_link(self):
        """
        This method places zero cost links in the topology graph
        representing availability of lane changes.
        """

        for segment in tqdm(self._topology, desc="Adding lane changes"):
            left_found, right_found = False, False

            for waypoint in segment.path:
                if not segment.entry.is_junction:
                    next_waypoint, next_road_option, next_segment = None, None, None

                    if (
                        waypoint.right_lane_marking
                        and waypoint.right_lane_marking.lane_change
                        == CarlaLaneChange.RIGHT
                        and not right_found
                    ):
                        next_waypoint = waypoint.right_lane
                        if (
                            next_waypoint is not None
                            and next_waypoint.lane_type == CarlaLaneType.DRIVING
                            and waypoint.road_id == next_waypoint.road_id
                        ):
                            next_road_option = RoadOption.CHANGELANERIGHT
                            next_segment = self._localize(
                                CarlaLocation.from_native(
                                    next_waypoint.transform.location
                                )
                            )
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment["entryxyz"]],
                                    next_segment[0],
                                    data=Edge(
                                        entry_wp=waypoint,
                                        exit_wp=next_waypoint,
                                        is_intersection=False,
                                        exit_vector=None,
                                        entry_vector=None,
                                        net_vector=None,
                                        path=[],
                                        length=0,
                                        type=next_road_option,
                                        change_wp=next_waypoint,
                                    ),
                                )
                                right_found = True
                    if (
                        waypoint.left_lane_marking
                        and waypoint.left_lane_marking.lane_change
                        == CarlaLaneChange.LEFT
                        and not left_found
                    ):
                        next_waypoint = waypoint.left_lane
                        if (
                            next_waypoint is not None
                            and next_waypoint.lane_type == CarlaLaneType.DRIVING
                            and waypoint.road_id == next_waypoint.road_id
                        ):
                            next_road_option = RoadOption.CHANGELANELEFT
                            next_segment = self._localize(
                                CarlaLocation.from_native(
                                    next_waypoint.transform.location
                                )
                            )
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment["entryxyz"]],
                                    next_segment[0],
                                    data=Edge(
                                        entry_wp=waypoint,
                                        exit_wp=next_waypoint,
                                        is_intersection=False,
                                        exit_vector=None,
                                        entry_vector=None,
                                        net_vector=None,
                                        path=[],
                                        length=0,
                                        type=next_road_option,
                                        change_wp=next_waypoint,
                                    ),
                                )
                                left_found = True
                if left_found and right_found:
                    break

    def _localize(self, location: CarlaLocation) -> int:
        """
        This function finds the road segment that a given location
        is part of, returning the edge it belongs to
        """
        waypoint = self._wmap.get_waypoint(location)
        edge = None
        try:
            edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][
                waypoint.lane_id
            ]
        except KeyError:
            pass
        return edge

    def _distance_heuristic(self, n1: int, n2: int) -> float:
        """
        Distance heuristic calculator for path searching
        in self._graph
        """
        l1: CarlaLocation = self._graph.nodes[n1]["vertex"]
        l2: CarlaLocation = self._graph.nodes[n2]["vertex"]
        l1 = l1.array
        l2 = l2.array
        return np.linalg.norm(l1 - l2)

    def _path_search(
        self, origin: CarlaLocation, destination: CarlaLocation
    ) -> List[int]:
        """
        This function finds the shortest path connecting origin and destination
        using A* search with distance heuristic.
        origin      :   carla.Location object of start position
        destination :   carla.Location object of of end position
        return      :   path as list of node ids (as int) of the graph self._graph
        connecting origin and destination
        """
        start, end = self._localize(origin), self._localize(destination)

        route: List[int] = nx.astar_path(
            self._graph,
            source=start[0],
            target=end[0],
            heuristic=self._distance_heuristic,
            weight="length",
        )
        route.append(end[1])
        return route

    def _successive_last_intersection_edge(
        self, index: int, route: List[int]
    ) -> Tuple[int, Any]:
        """
        This method returns the last successive intersection edge
        from a starting index on the route.
        This helps moving past tiny intersection edges to calculate
        proper turn decisions.
        returns a tuple with the last node id and the last edge
        """

        last_intersection_edge = None
        last_node = None
        for node1, node2 in [
            (route[i], route[i + 1]) for i in range(index, len(route) - 1)
        ]:
            candidate_edge = self._graph.edges[node1, node2]
            edge_data: Edge = candidate_edge["data"]
            if node1 == route[index]:
                last_intersection_edge = candidate_edge
            if edge_data.type == RoadOption.LANEFOLLOW and edge_data.is_intersection:
                last_intersection_edge = candidate_edge
                last_node = node2
            else:
                break

        return last_node, last_intersection_edge

    def _turn_decision(
        self, index: int, route: List[int], threshold=math.radians(35)
    ) -> RoadOption:
        """
        This method returns the turn decision (RoadOption) for pair of edges
        around current index of route list
        """

        decision = None
        previous_node = route[index - 1]
        current_node = route[index]
        next_node = route[index + 1]
        next_edge = self._graph.edges[current_node, next_node]
        next_edge_data: Edge = next_edge["data"]
        if index > 0:
            if (
                self._previous_decision != RoadOption.VOID
                and self._intersection_end_node > 0
                and self._intersection_end_node != previous_node
                and next_edge_data.type == RoadOption.LANEFOLLOW
                and next_edge_data.is_intersection
            ):
                decision = self._previous_decision
            else:
                self._intersection_end_node = -1
                current_edge = self._graph.edges[previous_node, current_node]
                current_edge_data: Edge = current_edge["data"]
                calculate_turn = (
                    current_edge_data.type == RoadOption.LANEFOLLOW
                    and not current_edge_data.is_intersection
                    and next_edge_data.type == RoadOption.LANEFOLLOW
                    and next_edge_data.is_intersection
                )
                if calculate_turn:
                    last_node, tail_edge = self._successive_last_intersection_edge(
                        index, route
                    )
                    self._intersection_end_node = last_node
                    if tail_edge is not None:
                        next_edge = tail_edge
                        next_edge_data: Edge = next_edge["data"]
                    cv, nv = current_edge_data.exit_vector, next_edge_data.exit_vector
                    if cv is None or nv is None:
                        return next_edge_data.type
                    cross_list = []
                    for neighbor in self._graph.successors(current_node):
                        select_edge = self._graph.edges[current_node, neighbor]
                        select_edge_data: Edge = select_edge["data"]
                        if select_edge_data.type == RoadOption.LANEFOLLOW:
                            if neighbor != route[index + 1]:
                                sv = select_edge_data.net_vector
                                cross_list.append(np.cross(cv.array, sv.array)[2])
                    next_cross = np.cross(cv.array, nv.array)[2]
                    deviation = math.acos(
                        np.clip(
                            np.dot(cv.array, nv.array)
                            / (np.linalg.norm(cv.array) * np.linalg.norm(nv.array)),
                            -1.0,
                            1.0,
                        )
                    )
                    if not cross_list:
                        cross_list.append(0)
                    if deviation < threshold:
                        decision = RoadOption.STRAIGHT
                    elif cross_list and next_cross < min(cross_list):
                        decision = RoadOption.LEFT
                    elif cross_list and next_cross > max(cross_list):
                        decision = RoadOption.RIGHT
                    elif next_cross < 0:
                        decision = RoadOption.LEFT
                    elif next_cross > 0:
                        decision = RoadOption.RIGHT
                else:
                    decision = next_edge_data.type

        else:
            decision = next_edge_data.type

        self._previous_decision = decision
        return decision

    def _find_closest_in_list(
        self, current_waypoint: CarlaWaypoint, waypoint_list: List[CarlaWaypoint]
    ) -> int:
        min_distance = float("inf")
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.location.distance_to(current_waypoint.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index
