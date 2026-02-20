from ..core.base_player import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    def select_action(self, observation, legal_moves):
        if not legal_moves:
            return None
        return int(np.random.choice(legal_moves))