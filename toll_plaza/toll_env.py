"""Discrete-time toll plaza queueing environment for reinforcement learning.

The agent dispatches each arriving vehicle to one of N lanes. Each lane has a
toll booth; ETC lanes serve vehicles faster. The reward is the negative total
number of vehicles waiting, so the agent learns to balance queue lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

MAX_Q = 7
ETC_SPEEDUP = 0.65


@dataclass(frozen=True)
class VehicleType:
    name: str
    color: str
    fee: float
    min_service: float
    max_service: float


VEHICLE_TYPES = [
    VehicleType("Car", "#3b82f6", 1.0, 5.0, 12.0),
    VehicleType("Motorcycle", "#f97316", 0.5, 3.0, 7.0),
    VehicleType("Truck", "#ef4444", 3.0, 18.0, 32.0),
    VehicleType("Bus", "#a855f7", 2.0, 14.0, 22.0),
]
TYPE_PROBS = np.array([0.55, 0.20, 0.15, 0.10])


@dataclass
class Vehicle:
    vid: int
    vtype: VehicleType
    arrival_tick: int
    lane: int = -1
    service_time: float = 0.0
    service_remaining: float = 0.0
    service_start: float = 0.0

    @property
    def wait_time(self) -> float:
        return self.service_start - self.arrival_tick


@dataclass
class Lane:
    index: int
    is_etc: bool
    service_mult: float
    queue: list[Vehicle] = field(default_factory=list)
    current: Vehicle | None = None


class TollPlazaEnv:
    def __init__(
        self,
        n_lanes: int = 4,
        n_etc: int = 1,
        arrival_rate: float = 0.5,
        episode_len: int = 1500,
        max_q: int = MAX_Q,
        seed: int | None = None,
    ):
        if n_etc > n_lanes:
            raise ValueError("n_etc cannot exceed n_lanes")
        self.n_lanes = n_lanes
        self.n_etc = n_etc
        self.arrival_rate = arrival_rate
        self.episode_len = episode_len
        self.max_q = max_q
        self.rng = np.random.default_rng(seed)
        self.n_states = (max_q + 1) ** n_lanes
        self.n_actions = n_lanes
        self.lanes = [
            Lane(i, is_etc=(i < n_etc), service_mult=ETC_SPEEDUP if i < n_etc else 1.0)
            for i in range(n_lanes)
        ]
        self.tick = 0
        self.vehicles: list[Vehicle] = []
        self.wait_times: list[float] = []
        self.revenue = 0.0
        self.served_count = 0
        self.done = False
        self._next_vid = 0

    def reset(self) -> tuple[tuple[int, ...], dict]:
        self.__init__(
            self.n_lanes,
            self.n_etc,
            self.arrival_rate,
            self.episode_len,
            self.max_q,
            seed=None,
        )
        return self.state_tuple(), {}

    def state_tuple(self) -> tuple[int, ...]:
        return tuple(min(len(l.queue), self.max_q) for l in self.lanes)

    def encode(self, state: tuple[int, ...]) -> int:
        idx = 0
        for i, q in enumerate(state):
            idx += q * (self.max_q + 1) ** i
        return idx

    def step(self, action: int) -> tuple[tuple[int, ...], float, bool, dict]:
        self.tick += 1
        served_now, revenue_now = self._advance_booths()

        arrived = self.rng.random() < self.arrival_rate
        if arrived:
            lane = int(np.clip(action, 0, self.n_lanes - 1))
            self._create_vehicle(lane)

        reward = -float(sum(len(l.queue) for l in self.lanes))
        self.done = self.tick >= self.episode_len
        info = {"arrived": arrived, "served": served_now, "revenue": revenue_now}
        return self.state_tuple(), reward, self.done, info

    def _advance_booths(self) -> tuple[int, float]:
        served_now = 0
        revenue_now = 0.0
        for lane in self.lanes:
            if lane.current is not None:
                lane.current.service_remaining -= 1.0
                if lane.current.service_remaining <= 0.0:
                    v = lane.current
                    self.wait_times.append(v.wait_time)
                    self.revenue += v.vtype.fee
                    self.served_count += 1
                    served_now += 1
                    revenue_now += v.vtype.fee
                    lane.current = None
            if lane.current is None and lane.queue:
                v = lane.queue.pop(0)
                v.service_time = self.rng.uniform(
                    v.vtype.min_service, v.vtype.max_service
                ) * lane.service_mult
                v.service_remaining = v.service_time
                v.service_start = float(self.tick)
                lane.current = v
        return served_now, revenue_now

    def _create_vehicle(self, lane: int) -> None:
        t = int(self.rng.choice(len(VEHICLE_TYPES), p=TYPE_PROBS))
        v = Vehicle(
            vid=self._next_vid,
            vtype=VEHICLE_TYPES[t],
            arrival_tick=self.tick,
            lane=lane,
        )
        self._next_vid += 1
        self.vehicles.append(v)
        self.lanes[lane].queue.append(v)

    @property
    def avg_wait(self) -> float:
        return float(np.mean(self.wait_times)) if self.wait_times else 0.0

    @property
    def total_queue(self) -> int:
        return sum(len(l.queue) for l in self.lanes)