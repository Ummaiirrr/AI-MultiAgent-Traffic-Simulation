"""
simulator.py
============
MODULE 3: Multi-Agent Environment & A* Pathfinding
====================================================
AI Multi-Agent Traffic Simulation System — University Lab Final Project

Core AI Concepts Demonstrated:
  - Informed Search: A* (A-Star) Algorithm
      f(n) = g(n) + h(n)
        g(n) — actual cost from start to node n (path length so far)
        h(n) — admissible heuristic estimating cost from n to goal
               (Manhattan distance + dynamic congestion penalty)
  - Agent-Based Modelling: each VehicleAgent perceives its environment,
    plans a path, and acts by following it step-by-step.
  - Environment Representation: 2D grid where cells carry a state
    (ROAD, OBSTACLE, SIGNAL_RED, SIGNAL_GREEN) and a congestion weight.

Grid Convention:
  0  — ROAD        (traversable)
  1  — OBSTACLE    (building / wall — not traversable)
  2  — SIGNAL_RED  (car must wait; cost penalty applied)
  3  — SIGNAL_GREEN (road with green light)

Heuristic h(n):
  Manhattan distance to goal + (CONGESTION_WEIGHT × local_congestion_density)
  The congestion density is derived from the number of cars currently on
  adjacent cells (dynamic, updated each simulation step).
"""

import heapq
import logging
import random
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid Cell States
# ---------------------------------------------------------------------------
ROAD         = 0
OBSTACLE     = 1
SIGNAL_RED   = 2
SIGNAL_GREEN = 3

# Cost constants
BASE_MOVE_COST: float      = 1.0
SIGNAL_RED_COST: float     = 5.0   # added to g(n) when passing through red light
CONGESTION_WEIGHT: float   = 2.0   # heuristic penalty weight per congested neighbour
OBSTACLE_PASSABLE: bool    = False


# ---------------------------------------------------------------------------
# City Grid Builder
# ---------------------------------------------------------------------------
def build_city_grid(rows: int = 20, cols: int = 20, seed: int = 42) -> np.ndarray:
    """
    Construct a 2D grid representing a simplified city.

    Layout Rules:
      - Even-indexed rows and columns are roads (forming a grid-street network).
      - Odd-indexed row/col intersections that are not on a street axis are obstacles
        (city blocks / buildings).
      - ~15 % of road cells are randomly converted to traffic signals.

    Args:
        rows: Grid height.
        cols: Grid width.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray of shape (rows, cols) with cell state values.
    """
    rng = random.Random(seed)
    grid = np.ones((rows, cols), dtype=np.int8)   # initialise all as OBSTACLE

    for r in range(rows):
        for c in range(cols):
            # Main street axes: every even row or even column is a road
            if r % 2 == 0 or c % 2 == 0:
                grid[r, c] = ROAD

    # Place traffic signals on ~15 % of road cells (excluding border)
    road_cells = [
        (r, c) for r in range(1, rows - 1)
                for c in range(1, cols - 1)
                if grid[r, c] == ROAD
    ]
    signal_count = int(len(road_cells) * 0.15)
    signal_cells = rng.sample(road_cells, signal_count)
    for (r, c) in signal_cells:
        grid[r, c] = SIGNAL_RED if rng.random() < 0.5 else SIGNAL_GREEN

    log.info(
        "City grid built (%dx%d). Roads: %d  Obstacles: %d  Signals: %d",
        rows, cols,
        np.sum(np.isin(grid, [ROAD, SIGNAL_RED, SIGNAL_GREEN])),
        np.sum(grid == OBSTACLE),
        len(signal_cells),
    )
    return grid


# ---------------------------------------------------------------------------
# A* Search Algorithm
# ---------------------------------------------------------------------------
@dataclass(order=True)
class _AStarNode:
    """Priority queue entry for A* open list."""
    f_score: float
    g_score: float = field(compare=False)
    position: tuple[int, int] = field(compare=False)
    parent: Optional["_AStarNode"] = field(compare=False, default=None)


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Manhattan distance heuristic — admissible for grid graphs."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic(
    pos: tuple[int, int],
    goal: tuple[int, int],
    grid: np.ndarray,
    congestion_map: np.ndarray,
) -> float:
    """
    Informed heuristic h(n) for A* search.

    h(n) = Manhattan_Distance(n, goal)
           + CONGESTION_WEIGHT × local_congestion_density(n)

    local_congestion_density is the number of vehicles on adjacent cells
    (drawn from the live congestion_map). This makes the heuristic dynamic:
    heavily-trafficked neighbourhoods are penalised, steering agents onto
    less-loaded alternative routes.

    The heuristic remains ADMISSIBLE because the penalty is additive and
    never over-estimates the actual remaining travel cost when congestion
    weights are properly bounded.

    Args:
        pos:            Current cell (row, col).
        goal:           Goal cell (row, col).
        grid:           Static city grid.
        congestion_map: Float array of vehicle density per cell.

    Returns:
        Estimated cost from pos to goal.
    """
    base_h      = manhattan_distance(pos, goal)
    rows, cols  = grid.shape
    r, c        = pos

    # Sum vehicle counts on the 4-connected neighbours
    neighbours  = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    local_cong  = sum(
        congestion_map[nr, nc]
        for (nr, nc) in neighbours
        if 0 <= nr < rows and 0 <= nc < cols
    )
    return base_h + CONGESTION_WEIGHT * local_cong


def astar_search(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    congestion_map: np.ndarray,
) -> list[tuple[int, int]]:
    """
    A* (A-Star) Informed Search — finds least-cost path from start to goal.

    Algorithm Steps:
      1. Initialise open list (min-heap) with start node; f = h(start).
      2. Pop lowest-f node from open list.
      3. If it equals goal, reconstruct and return path.
      4. For each traversable 4-connected neighbour:
           a. Compute tentative g = g(current) + step_cost(neighbour).
           b. If this g is lower than any previously seen g for neighbour,
              update and push to open list.
      5. Repeat until goal found or open list empty (no path).

    Step cost of a neighbour cell:
      BASE_MOVE_COST  +  SIGNAL_RED_COST  (if cell is SIGNAL_RED)

    Args:
        grid:           2D city grid.
        start:          Origin cell (row, col).
        goal:           Destination cell (row, col).
        congestion_map: Live vehicle density per cell.

    Returns:
        Ordered list of (row, col) tuples from start to goal (inclusive),
        or an empty list if no path exists.
    """
    rows, cols = grid.shape

    # Validate start and goal
    for name, pos in [("Start", start), ("Goal", goal)]:
        r, c = pos
        if not (0 <= r < rows and 0 <= c < cols):
            log.warning("A*: %s (%s) is out of grid bounds.", name, pos)
            return []
        if grid[r, c] == OBSTACLE:
            log.warning("A*: %s (%s) is an obstacle.", name, pos)
            return []

    # g-score table: shortest known cost to reach each cell
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    # Open list — min-heap ordered by f = g + h
    h_start  = heuristic(start, goal, grid, congestion_map)
    open_heap: list[_AStarNode] = []
    heapq.heappush(open_heap, _AStarNode(f_score=h_start, g_score=0.0, position=start))

    closed_set: set[tuple[int, int]] = set()
    node_map: dict[tuple[int, int], _AStarNode] = {}

    while open_heap:
        current = heapq.heappop(open_heap)
        pos     = current.position

        if pos in closed_set:
            continue
        closed_set.add(pos)

        # Goal reached — reconstruct path
        if pos == goal:
            path = []
            node = current
            while node is not None:
                path.append(node.position)
                node = node.parent
            return path[::-1]   # reverse to get start → goal order

        r, c = pos
        neighbours = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

        for (nr, nc) in neighbours:
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in closed_set:
                continue
            if grid[nr, nc] == OBSTACLE:
                continue

            # Step cost: base + red-light penalty
            step_cost = BASE_MOVE_COST
            if grid[nr, nc] == SIGNAL_RED:
                step_cost += SIGNAL_RED_COST

            tentative_g = current.g_score + step_cost

            if tentative_g < g_score.get((nr, nc), math.inf):
                g_score[(nr, nc)] = tentative_g
                h_val = heuristic((nr, nc), goal, grid, congestion_map)
                f_val = tentative_g + h_val
                neighbour_node = _AStarNode(
                    f_score=f_val,
                    g_score=tentative_g,
                    position=(nr, nc),
                    parent=current,
                )
                heapq.heappush(open_heap, neighbour_node)

    log.warning("A*: No path found from %s to %s.", start, goal)
    return []


# ---------------------------------------------------------------------------
# VehicleAgent
# ---------------------------------------------------------------------------
class VehicleAgent:
    """
    Autonomous vehicle agent that navigates the city grid using A* search.

    Attributes:
        agent_id (int):             Unique integer identifier.
        current_position (tuple):   Current (row, col) on the grid.
        destination (tuple):        Target (row, col) the agent aims to reach.
        path (list):                Ordered list of (row, col) waypoints computed by A*.
        step_index (int):           Index into path (how far along the agent is).
        reached_goal (bool):        True once the agent reaches its destination.
        color (str):                Display colour string for the dashboard.
        total_steps (int):          Total number of grid moves made so far.
    """

    _id_counter: int = 0

    def __init__(
        self,
        current_position: tuple[int, int],
        destination: tuple[int, int],
        grid: np.ndarray,
        congestion_map: np.ndarray,
    ) -> None:
        VehicleAgent._id_counter += 1
        self.agent_id        = VehicleAgent._id_counter
        self.current_position = current_position
        self.destination      = destination
        self.path             = []
        self.step_index       = 0
        self.reached_goal     = False
        self.total_steps      = 0
        self.color            = random.choice([
            "#FF5733", "#33A8FF", "#33FF57", "#FF33A8", "#FFA833",
            "#A833FF", "#33FFF5", "#FF3333", "#33FF99", "#FFD733",
        ])
        # Plan the initial path
        self.replan(grid, congestion_map)

    def replan(self, grid: np.ndarray, congestion_map: np.ndarray) -> None:
        """
        Re-run A* from current_position to destination and update self.path.
        Called on spawn and optionally each step to adapt to dynamic congestion.

        Args:
            grid:           City grid.
            congestion_map: Current congestion density map.
        """
        new_path = astar_search(grid, self.current_position, self.destination, congestion_map)
        if new_path:
            self.path       = new_path
            self.step_index = 0
        # If no path found, retain previous path (agent waits)

    def step(self, grid: np.ndarray, congestion_map: np.ndarray) -> bool:
        """
        Advance the agent one step along its planned path.

        If the current cell ahead is a RED signal, the agent waits (does not move).
        Returns True if the agent moved, False if it waited or is already at goal.

        Args:
            grid:           City grid (may have changed signals since last step).
            congestion_map: Live congestion map.

        Returns:
            bool: True if a move was made this step.
        """
        if self.reached_goal or not self.path:
            return False

        next_idx = self.step_index + 1
        if next_idx >= len(self.path):
            self.reached_goal = True
            return False

        next_pos = self.path[next_idx]
        nr, nc   = next_pos

        # Red-light behaviour: agent waits at current cell
        if grid[nr, nc] == SIGNAL_RED:
            return False

        self.current_position = next_pos
        self.step_index       = next_idx
        self.total_steps     += 1

        if self.current_position == self.destination:
            self.reached_goal = True

        return True

    def __repr__(self) -> str:
        return (
            f"VehicleAgent(id={self.agent_id}, pos={self.current_position}, "
            f"dest={self.destination}, steps={self.total_steps}, "
            f"done={self.reached_goal})"
        )


# ---------------------------------------------------------------------------
# Simulation Environment
# ---------------------------------------------------------------------------
class TrafficSimulation:
    """
    Multi-agent traffic simulation environment.

    Responsibilities:
      - Maintain the city grid and traffic signal states.
      - Maintain the live congestion map (vehicle count per cell).
      - Spawn VehicleAgents at random road cells with random destinations.
      - Advance the simulation one discrete time-step at a time.
      - Cycle traffic signals every N steps.

    Attributes:
        grid (np.ndarray):          City grid.
        congestion_map (np.ndarray): Float density map (same shape as grid).
        agents (list):              Active VehicleAgent instances.
        step_count (int):           Current simulation tick.
        signal_cycle (int):         Steps between signal state flips.
        rows (int):                 Grid height.
        cols (int):                 Grid width.
    """

    def __init__(
        self,
        rows: int = 20,
        cols: int = 20,
        signal_cycle: int = 8,
        grid_seed: int = 42,
    ) -> None:
        self.rows         = rows
        self.cols         = cols
        self.signal_cycle = signal_cycle
        self.step_count   = 0
        self.grid         = build_city_grid(rows, cols, seed=grid_seed)
        self.congestion_map = np.zeros((rows, cols), dtype=np.float32)
        self.agents: list[VehicleAgent] = []
        VehicleAgent._id_counter = 0   # reset id counter for fresh sims

    def _road_cells(self) -> list[tuple[int, int]]:
        """Return list of all traversable (non-obstacle) cell coordinates."""
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r, c] != OBSTACLE
        ]

    def _update_congestion_map(self) -> None:
        """Recompute congestion_map from current agent positions."""
        self.congestion_map[:] = 0
        for agent in self.agents:
            r, c = agent.current_position
            self.congestion_map[r, c] += 1

    def _cycle_signals(self) -> None:
        """
        Toggle traffic signals between RED and GREEN.
        Called every self.signal_cycle steps to simulate dynamic signal timing.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == SIGNAL_RED:
                    self.grid[r, c] = SIGNAL_GREEN
                elif self.grid[r, c] == SIGNAL_GREEN:
                    self.grid[r, c] = SIGNAL_RED

    def spawn_agents(self, count: int, replan: bool = False) -> None:
        """
        Spawn `count` new VehicleAgents at random distinct road cells.

        Each agent's origin and destination are chosen at random from
        available road cells (origin ≠ destination).

        Args:
            count:  Number of agents to spawn.
            replan: If True, existing agents also replan their paths
                    (useful after a congestion state change).
        """
        road = self._road_cells()
        if len(road) < 2:
            log.warning("Not enough road cells to spawn agents.")
            return

        spawned = 0
        attempts = 0
        max_attempts = count * 10

        while spawned < count and attempts < max_attempts:
            attempts += 1
            origin, dest = random.sample(road, 2)
            agent = VehicleAgent(
                current_position=origin,
                destination=dest,
                grid=self.grid,
                congestion_map=self.congestion_map,
            )
            if agent.path:   # only add if a valid path was found
                self.agents.append(agent)
                spawned += 1

        self._update_congestion_map()
        log.info("Spawned %d agents. Total active: %d", spawned, len(self.agents))

        if replan:
            for a in self.agents:
                a.replan(self.grid, self.congestion_map)

    def tick(self, replan_interval: int = 5) -> dict:
        """
        Advance the simulation by one discrete time step.

        Actions performed:
          1. Cycle traffic signals every self.signal_cycle steps.
          2. Each agent moves one step (or waits at red light).
          3. Agents that reached their destination are removed.
          4. Every replan_interval steps, agents replan to adapt to new congestion.
          5. Congestion map is refreshed.

        Args:
            replan_interval: How often agents recalculate their paths.

        Returns:
            dict with step stats: step, n_agents, n_moved, n_arrived.
        """
        self.step_count += 1

        # Signal cycling
        if self.step_count % self.signal_cycle == 0:
            self._cycle_signals()

        # Periodic replanning for dynamic congestion adaptation
        if self.step_count % replan_interval == 0:
            for agent in self.agents:
                if not agent.reached_goal:
                    agent.replan(self.grid, self.congestion_map)

        # Agent movement
        n_moved   = 0
        n_arrived = 0
        for agent in self.agents:
            moved = agent.step(self.grid, self.congestion_map)
            if moved:
                n_moved += 1
            if agent.reached_goal:
                n_arrived += 1

        # Remove arrived agents
        self.agents = [a for a in self.agents if not a.reached_goal]

        # Refresh congestion map
        self._update_congestion_map()

        return {
            "step"     : self.step_count,
            "n_agents" : len(self.agents),
            "n_moved"  : n_moved,
            "n_arrived": n_arrived,
        }

    def get_agent_positions(self) -> list[dict]:
        """
        Snapshot of all active agents for rendering.

        Returns:
            List of dicts with keys: agent_id, row, col, color,
                                      dest_row, dest_col, total_steps.
        """
        return [
            {
                "agent_id"  : a.agent_id,
                "row"       : a.current_position[0],
                "col"       : a.current_position[1],
                "color"     : a.color,
                "dest_row"  : a.destination[0],
                "dest_col"  : a.destination[1],
                "total_steps": a.total_steps,
            }
            for a in self.agents
        ]

    def get_grid_state(self) -> dict:
        """
        Return full grid + congestion snapshot for the dashboard heatmap.

        Returns:
            dict with keys: grid (list[list[int]]), congestion (list[list[float]]).
        """
        return {
            "grid"      : self.grid.tolist(),
            "congestion": self.congestion_map.tolist(),
        }


# ---------------------------------------------------------------------------
# Entrypoint — quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("MODULE 3 — Multi-Agent Traffic Simulation Smoke Test")
    log.info("=" * 60)

    sim = TrafficSimulation(rows=20, cols=20, signal_cycle=8)
    sim.spawn_agents(count=10)

    for tick_num in range(20):
        stats = sim.tick()
        log.info(
            "Tick %2d | Agents: %2d | Moved: %2d | Arrived: %d",
            stats["step"], stats["n_agents"], stats["n_moved"], stats["n_arrived"],
        )
        if stats["n_agents"] == 0:
            sim.spawn_agents(count=5)

    log.info("Smoke test complete. Simulation environment is functional.")
