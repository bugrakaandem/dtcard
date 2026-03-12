import numpy as np
from .base_game import BaseCardGame


class HeartsGame(BaseCardGame):
    def __init__(self):
        super().__init__(num_players=4, deck_size=52)

        self.hands = [[] for _ in range(4)]
        self.current_trick = []
        self.tricks_history = []
        self.scores_round = [0] * 4

        self.hearts_broken = False
        self.current_player_idx = 0
        self.trick_leader_idx = 0

    def reset(self):
        deck = np.random.permutation(52)
        self.hands = [sorted(deck[i * 13:(i + 1) * 13].tolist()) for i in range(4)]

        self.current_trick = []
        self.tricks_history = []
        self.scores_round = [0] * 4
        self.hearts_broken = False

        self.current_player_idx = 0
        found_start = False
        for pid, hand in enumerate(self.hands):
            if 0 in hand:
                self.current_player_idx = pid
                self.trick_leader_idx = pid
                found_start = True
                break

        if not found_start:
            self.current_player_idx = np.random.randint(0, 4)
            self.trick_leader_idx = self.current_player_idx

        return self.get_observation(self.current_player_idx)

    def get_legal_moves(self, player_index=None):
        if player_index is None:
            player_index = self.current_player_idx

        hand = self.hands[player_index]
        if not hand:
            return []

        if len(self.tricks_history) == 0 and len(self.current_trick) == 0:
            if 0 in hand:
                return [0]

        if len(self.current_trick) == 0:
            if self.hearts_broken:
                return hand
            else:
                non_hearts = [c for c in hand if not (39 <= c <= 51)]
                if non_hearts:
                    return non_hearts
                else:
                    return hand

        leader_card = self.current_trick[0][1]
        leader_suit = leader_card // 13

        same_suit_cards = [c for c in hand if (c // 13) == leader_suit]

        if same_suit_cards:
            return same_suit_cards
        else:
            return hand

    def step(self, action):
        player_idx = self.current_player_idx

        if action not in self.hands[player_idx]:
            legal_moves = self.get_legal_moves(player_idx)
            action = legal_moves[0]

        self.hands[player_idx].remove(action)
        self.current_trick.append((player_idx, action))

        if 39 <= action <= 51:
            self.hearts_broken = True

        rewards = [0.0] * 4
        game_over = False
        step_info = {"played_card": action, "player": player_idx}

        if len(self.current_trick) == 4:
            winner_idx, points = self._resolve_trick()

            self.scores_round[winner_idx] += points

            if points > 0:
                rewards[winner_idx] = -points / 26.0

            self.tricks_history.append(list(self.current_trick))
            self.current_trick = []

            self.current_player_idx = winner_idx
            self.trick_leader_idx = winner_idx

            step_info["trick_winner"] = winner_idx
            step_info["trick_points"] = points

            if len(self.tricks_history) == 13:
                game_over = True
                final_rewards = self._calculate_final_rewards()
                rewards = [r + fr for r, fr in zip(rewards, final_rewards)]
        else:
            self.current_player_idx = (self.current_player_idx + 1) % 4

        next_obs = self.get_observation(self.current_player_idx)
        return next_obs, rewards, game_over, step_info

    def _resolve_trick(self):
        leader_card = self.current_trick[0][1]
        leader_suit = leader_card // 13

        highest_rank = -1
        winner_idx = -1
        points = 0

        for p_idx, card in self.current_trick:
            suit = card // 13
            rank = card % 13

            if suit == 3:
                points += 1
            if card == 36:
                points += 13

            if suit == leader_suit:
                if rank > highest_rank:
                    highest_rank = rank
                    winner_idx = p_idx

        return winner_idx, points

    def _calculate_final_rewards(self):
        rewards = [0.0] * 4

        shooter_idx = -1
        for i in range(4):
            if self.scores_round[i] == 26:
                shooter_idx = i
                break

        if shooter_idx != -1:
            for i in range(4):
                if i == shooter_idx:
                    rewards[i] = 2.0
                else:
                    rewards[i] = -1.0
        else:
            min_score = min(self.scores_round)
            for i in range(4):
                if self.scores_round[i] == min_score:
                    rewards[i] = 1.0
                else:
                    rewards[i] = - (self.scores_round[i] / 26.0)

        return rewards

    def get_observation(self, player_index):
        obs = np.zeros(212, dtype=np.float32)

        for c in self.hands[player_index]:
            obs[c] = 1.0

        for _, c in self.current_trick:
            obs[52 + c] = 1.0

        for trick in self.tricks_history:
            for _, c in trick:
                obs[104 + c] = 1.0

        best = self._current_best_card(self.current_trick)
        if best is not None:
            obs[156 + best] = 1.0

        return obs

    def _current_best_card(self, trick):
        if not trick:
            return None
        leader_card = trick[0][1]
        leader_suit = leader_card // 13

        highest_rank = -1
        best_card = leader_card

        for _, card in trick:
            suit = card // 13
            rank = card % 13
            if suit == leader_suit and rank > highest_rank:
                highest_rank = rank
                best_card = card

        return best_card
