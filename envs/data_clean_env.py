from typing import Tuple, Dict, Any
from pydantic import BaseModel

# Define typed models
class Observation(BaseModel):
    raw_data: str

class Action(BaseModel):
    cleaned_data: str

class Reward(BaseModel):
    value: float

class DataCleanEnv:
    def __init__(self):
        # Example dataset: (raw_data, expected_cleaned)
        self.dataset = [
            ("Name, Age\nAlice, 23\nBob, twenty", "Name,Age\nAlice,23\nBob,20"),
            ("ID; Score\n1; ninety\n2; 85", "ID,Score\n1,90\n2,85"),
        ]
        self.index = 0
        self.done = False

    def reset(self) -> Observation:
        self.index = 0
        self.done = False
        return Observation(raw_data=self.dataset[self.index][0])

    def step(self, action: str) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        correct_cleaned = self.dataset[self.index][1]
        # Reward: partial credit based on overlap
        reward = self._grade(action, correct_cleaned)

        self.index += 1
        if self.index >= len(self.dataset):
            self.done = True
            next_obs = Observation(raw_data="")
        else:
            next_obs = Observation(raw_data=self.dataset[self.index][0])

        return next_obs, reward, self.done, {"expected": correct_cleaned}

    def state(self) -> Dict[str, Any]:
        return {"index": self.index, "done": self.done}

    def _grade(self, action: str, expected: str) -> float:
        # Simple grader: proportion of matching characters
        matches = sum(1 for a, b in zip(action, expected) if a == b)
        return round(matches / max(len(expected), 1), 2)