from abc import ABC, abstractmethod
import numpy as np


class BaseCardGame(ABC):
    def __init__(self, num_players=4, deck_size=52):
        self.num_players = num_players
        self.deck_size = deck_size

        self.action_space_size = 66
        self.history = []

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_legal_moves(self, player_index=None):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_observation(self, player_index):
        pass

    @abstractmethod
    def card_id_to_string(self, card_id):
        if card_id is None: return "None"
        if card_id >= 52:
            return f"BID_{card_id - 52}"

        suits = ['C', 'D', 'S', 'H']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

        suit = card_id // 13
        rank = card_id % 13
        return f"{suits[suit]}{ranks[rank]}"