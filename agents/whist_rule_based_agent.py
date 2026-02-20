import random
from collections import Counter
from typing import List, Optional

import numpy as np


class WhistRuleBasedAgent:
    def __init__(self, player_index: int, seed: Optional[int] = None):
        self.player_index = player_index
        self.rng = random.Random(seed)

    def select_action(self, obs: np.ndarray, legal_moves: List[int]) -> int:
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        v = np.asarray(obs, dtype=np.float32).reshape(-1)

        hand = list(np.where(v[0:52] > 0.5)[0])
        table_cards = list(np.where(v[52:104] > 0.5)[0])
        best_oh = v[104:156]
        best_card = int(np.argmax(best_oh)) if float(best_oh.sum()) > 0.5 else None
        trump_oh = v[156:160]
        trump = int(np.argmax(trump_oh)) if float(trump_oh.sum()) > 0.5 else -1

        is_leading = (len(table_cards) == 0)

        if is_leading:
            return self._lead_move(hand, legal_moves, trump)

        lead_suit = self._infer_lead_suit_from_legal(legal_moves, trump)

        current_winner = best_card if best_card is not None else (table_cards[0] if table_cards else None)

        if lead_suit is not None:
            same_suit_legal = [c for c in legal_moves if self._suit(c) == lead_suit]
        else:
            same_suit_legal = []

        if same_suit_legal:
            winning = [c for c in same_suit_legal if self._beats(c, current_winner, lead_suit, trump)]
            if winning:
                return min(winning, key=self._rank_key)
            return min(same_suit_legal, key=self._rank_key)

        trumps_in_legal = [c for c in legal_moves if self._suit(c) == trump]
        if trumps_in_legal:
            winning_trumps = [c for c in trumps_in_legal if self._beats(c, current_winner, lead_suit, trump)]
            if winning_trumps:
                return min(winning_trumps, key=self._rank_key)
            return self._discard_move(hand, legal_moves, trump)

        return self._discard_move(hand, legal_moves, trump)

    def _lead_move(self, hand: List[int], legal_moves: List[int], trump: int) -> int:
        if not hand:
            return self.rng.choice(legal_moves)

        suit_counts = Counter(self._suit(c) for c in hand)
        trump_count = suit_counts.get(trump, 0)

        non_trump_suits = [s for s in suit_counts.keys() if s != trump]
        if non_trump_suits:
            best_suit = max(non_trump_suits, key=lambda s: (suit_counts[s], self._max_rank_in_suit(hand, s)))
            candidates = [c for c in legal_moves if self._suit(c) == best_suit]
            if candidates:
                sorted_cards = sorted(candidates, key=self._rank_key)
                return sorted_cards[-2] if len(sorted_cards) >= 2 else sorted_cards[-1]

        if trump_count >= 5:
            trumps = [c for c in legal_moves if self._suit(c) == trump]
            if trumps:
                return min(trumps, key=self._rank_key)

        return min(legal_moves, key=self._rank_key)

    def _discard_move(self, hand: List[int], legal_moves: List[int], trump: int) -> int:
        suit_counts = Counter(self._suit(c) for c in hand) if hand else Counter()
        non_trumps = [c for c in legal_moves if self._suit(c) != trump]
        if non_trumps:
            return min(non_trumps, key=lambda c: (suit_counts.get(self._suit(c), 99), self._rank_key(c)))
        return min(legal_moves, key=self._rank_key)

    def _beats(self, a: int, b: int, lead_suit: Optional[int], trump: int) -> bool:
        if b is None:
            return True

        a_s, b_s = self._suit(a), self._suit(b)
        a_tr, b_tr = (a_s == trump), (b_s == trump)

        if a_tr and not b_tr:
            return True
        if b_tr and not a_tr:
            return False
        if a_tr and b_tr:
            return self._rank_key(a) > self._rank_key(b)

        if lead_suit is not None:
            if a_s == lead_suit and b_s != lead_suit:
                return True
            if b_s == lead_suit and a_s != lead_suit:
                return False
            if a_s == lead_suit and b_s == lead_suit:
                return self._rank_key(a) > self._rank_key(b)

        return self._rank_key(a) > self._rank_key(b)

    @staticmethod
    def _suit(card_id: int) -> int:
        return int(card_id) // 13

    @staticmethod
    def _rank_key(card_id: int) -> int:
        return int(card_id) % 13

    def _max_rank_in_suit(self, hand: List[int], suit: int) -> int:
        cards = [c for c in hand if self._suit(c) == suit]
        return max((self._rank_key(c) for c in cards), default=-1)

    def _infer_lead_suit_from_legal(self, legal_moves: List[int], trump: int) -> Optional[int]:
        suits = {self._suit(c) for c in legal_moves}
        if len(suits) == 1:
            return next(iter(suits))
        return None
