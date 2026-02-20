import random


class RuleBasedAgent:
    def __init__(self, player_index, seed=None):
        self.player_index = player_index
        self.random = random.Random(seed) if seed else random.Random()

    def select_action(self, obs, legal_moves):
        if len(legal_moves) == 1:
            return legal_moves[0]

        playable_cards = [self._decode_card(m) for m in legal_moves]

        playable_cards.sort(key=lambda x: x['value'])

        unique_suits = set(c['suit'] for c in playable_cards)
        is_sloughing_opportunity = len(unique_suits) > 1

        if is_sloughing_opportunity:
            best_discard = max(playable_cards, key=lambda x: self._calculate_danger(x))
            return best_discard['id']

        chosen_card = playable_cards[0]
        return chosen_card['id']

    def _calculate_danger(self, card):
        if card['suit_name'] == 'spades' and card['value'] == 12:
            return 1000
        if card['suit_name'] == 'spades' and card['value'] > 12:
            return 15
        if card['suit_name'] == 'hearts':
            return 20 + card['value']

        return card['value']

    def _decode_card(self, card_id):
        suit_id = card_id // 13
        value_idx = card_id % 13

        suits = ['clubs', 'diamonds', 'spades', 'hearts']
        real_value = value_idx + 2

        return {
            'id': card_id,
            'suit': suit_id,
            'suit_name': suits[suit_id],
            'value': real_value
        }