from typing import Tuple, Dict, Any
from pydantic import BaseModel

# Define typed models
class Observation(BaseModel):
    code_snippet: str

class Action(BaseModel):
    feedback: str  # Agent suggests bug fix or style improvement

class Reward(BaseModel):
    value: float

class CodeReviewEnv:
    def __init__(self):
        # Example dataset: (code_snippet, expected_issue)
        self.dataset = [
            ("def add(a,b): return a+b", "style"),   # Missing spaces after commas
            ("for i in range(10): print(i)", "style"), # Inline print without indentation
            ("def divide(a,b): return a/b", "bug"),  # No zero-division handling
        ]
        self.index = 0
        self.done = False

    def reset(self) -> Observation:
        self.index = 0
        self.done = False
        return Observation(code_snippet=self.dataset[self.index][0])

    def step(self, action: str) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        expected_issue = self.dataset[self.index][1]
        # Reward: 1.0 if agent mentions the expected issue, else 0.0
        reward = 1.0 if expected_issue in action.lower() else 0.0

        self.index += 1
        if self.index >= len(self.dataset):
            self.done = True
            next_obs = Observation(code_snippet="")
        else:
            next_obs = Observation(code_snippet=self.dataset[self.index][0])

        return next_obs, reward, self.done, {"expected_issue": expected_issue}

    def state(self) -> Dict[str, Any]:
        return {"index": self.index, "done": self.done}