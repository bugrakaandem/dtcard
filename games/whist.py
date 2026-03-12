import numpy as np
from .base_game import BaseCardGame


class WhistGame(BaseCardGame):
    OBS_DIM = 212

    def __init__(self, seed=None):
        super().__init__(num_players=4, deck_size=52)
        self.rng = np.random.default_rng(seed)

        self.hands = [[] for _ in range(4)]
        self.current_trick = []
        self.tricks_played = 0
        self.scores = [0] * 4
        self.current_player_idx = 0
        self.trick_leader_idx = 0
        self.trump_suit = 0

        self.played_cards_in_game = set()

    def reset(self):
        deck = np.arange(52, dtype=np.int32)
        self.rng.shuffle(deck)

        self.hands = [sorted(deck[i * 13:(i + 1) * 13].tolist()) for i in range(4)]

        last_card = int(deck[-1])
        self.trump_suit = last_card // 13

        self.current_trick = []
        self.tricks_played = 0
        self.scores = [0] * 4
        self.played_cards_in_game = set()

        self.current_player_idx = 0
        self.trick_leader_idx = 0

        if hasattr(self, "history"):
            self.history = []

        return self.get_observation(self.current_player_idx)

    def get_legal_moves(self, player_index=None):
        if player_index is None:
            player_index = self.current_player_idx

        hand = self.hands[player_index]
        if not hand:
            return []

        if len(self.current_trick) == 0:
            return hand

        lead_card = self.current_trick[0][1]
        lead_suit = lead_card // 13
        same_suit = [c for c in hand if (c // 13) == lead_suit]
        return same_suit if same_suit else hand

    def step(self, action):
        player = self.current_player_idx
        legal = self.get_legal_moves(player)

        if action not in self.hands[player] or action not in legal:
            if not legal:
                return self.get_observation(self.current_player_idx), [0.0] * 4, True, {"error": "no_legal_moves"}
            action = legal[0]

        self.hands[player].remove(action)
        self.current_trick.append((player, action))
        self.played_cards_in_game.add(action)
        self.history.append((player, action))

        rewards = [0.0] * 4
        done = False
        info = {"player": player, "played_card": action, "trump_suit": self.trump_suit}

        if len(self.current_trick) == 4:
            winner = self._resolve_trick(self.current_trick)
            self.scores[winner] += 1
            rewards[winner] = 1.0

            info["trick_winner"] = winner
            info["scores"] = list(self.scores)

            self.current_trick = []
            self.tricks_played += 1

            self.current_player_idx = winner
            self.trick_leader_idx = winner

            if self.tricks_played >= 13:
                done = True
        else:
            self.current_player_idx = (self.current_player_idx + 1) % 4

        return self.get_observation(self.current_player_idx), rewards, done, info


    def get_observation(self, player_index):
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        for c in self.hands[player_index]:
            obs[c] = 1.0

        for _, c in self.current_trick:
            obs[52 + c] = 1.0

        for c in self.played_cards_in_game:
            obs[104 + c] = 1.0

        best = self._current_best_card(self.current_trick)
        if best is not None:
            obs[156 + best] = 1.0

        if 0 <= self.trump_suit <= 3:
            obs[208 + self.trump_suit] = 1.0

        return obs

    @staticmethod
    def _rank(card_id: int) -> int:
        return int(card_id) % 13

    @staticmethod
    def _suit(card_id: int) -> int:
        return int(card_id) // 13

    def _beats(self, a: int, b: int, lead_suit: int) -> bool:
        a_s, b_s = self._suit(a), self._suit(b)
        a_tr, b_tr = (a_s == self.trump_suit), (b_s == self.trump_suit)

        if a_tr and not b_tr:
            return True
        if b_tr and not a_tr:
            return False

        if a_tr and b_tr:
            return self._rank(a) > self._rank(b)

        if a_s == lead_suit and b_s != lead_suit:
            return True
        if b_s == lead_suit and a_s != lead_suit:
            return False
        if a_s == lead_suit and b_s == lead_suit:
            return self._rank(a) > self._rank(b)

        return False

    def _resolve_trick(self, trick):
        lead_card = trick[0][1]
        lead_suit = self._suit(lead_card)

        best_player, best_card = trick[0]
        for p, c in trick[1:]:
            if self._beats(c, best_card, lead_suit):
                best_card = c
                best_player = p
        return best_player

    def _current_best_card(self, trick):
        if not trick:
            return None
        lead_card = trick[0][1]
        lead_suit = self._suit(lead_card)

        best = lead_card
        for _, c in trick[1:]:
            if self._beats(c, best, lead_suit):
                best = c
        return best
