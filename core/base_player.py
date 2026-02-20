class BaseAgent(ABC):
    def __init__(self, player_idx):
        self.player_idx = player_idx

    @abstractmethod
    def select_action(self, observation, legal_moves):
        pass

