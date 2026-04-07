from typing import Tuple, Dict, Any
from pydantic import BaseModel

# Define typed models
class Observation(BaseModel):
    email_text: str

class Action(BaseModel):
    label: str  # "spam", "urgent", "normal"

class Reward(BaseModel):
    value: float

class EmailEnv:
    def __init__(self):
        # Example dataset: (email_text, correct_label)
        self.dataset = [
            ("Win a free iPhone now!", "spam"),
            ("Meeting at 10 AM tomorrow", "urgent"),
            ("Weekly newsletter attached", "normal"),
        ]
        self.index = 0
        self.done = False

    def reset(self) -> Observation:
        self.index = 0
        self.done = False
        return Observation(email_text=self.dataset[self.index][0])

    def step(self, action: str) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        correct_label = self.dataset[self.index][1]
        reward = 1.0 if action == correct_label else 0.0

        self.index += 1
        if self.index >= len(self.dataset):
            self.done = True
            next_obs = Observation(email_text="")
        else:
            next_obs = Observation(email_text=self.dataset[self.index][0])

        return next_obs, reward, self.done, {"correct": correct_label}

    def state(self) -> Dict[str, Any]:
        return {"index": self.index, "done": self.done}