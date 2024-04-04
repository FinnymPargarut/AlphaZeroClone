import torch
import numpy as np
import math


class Node:
    def __init__(self, game, args, state, move_count=0, is_draw=False, parent=None, action_taken=None, player=1, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.player = player
        self.prior = prior

        self.move_count_for_draw = move_count
        self.is_draw = is_draw

        self.children = []

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['exploration_constant'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        states_counter = {str(self.state): 1}
        parent_pointer = self.parent
        while parent_pointer is not None:
            key = str(parent_pointer.state)
            states_counter[key] = states_counter.get(key, 0) + 1
            parent_pointer = parent_pointer.parent

        policy_indices = [(idx, prob) for idx, prob in np.ndenumerate(policy)]
        sorted_policy = sorted(policy_indices, key=lambda x: x[1], reverse=True)
        top_moves = sorted_policy[:5]
        for (start_idx, end_idx), prob in top_moves:
            if prob > 0:
                start_coord = [start_idx // 8, start_idx % 8]
                end_coord = [end_idx // 8, end_idx % 8]
                action = [start_coord, end_coord]

                child_state = self.state.copy()

                # get next state with check draw
                piece = child_state[start_coord[0], start_coord[1]]
                piece_after = child_state[end_coord[0], end_coord[1]] if end_coord[0] <= 7 else 100
                child_state = self.game.get_next_state(child_state, action, self.player, is_training=True)
                self.is_draw = self.game.check_state_for_draw(child_state, states_counter, self.is_draw)
                child_move_count, self.is_draw = self.game.check_moves_for_draw(piece, piece_after,
                                                                                self.move_count_for_draw, self.is_draw)

                child_state = self.game.change_perspective(child_state, player=-1)
                child_player = self.game.get_opponent(self.player)

                child = Node(self.game, self.args, child_state, child_move_count, self.is_draw, self, action,
                             child_player, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def get_policy_matrix(self, start_policy, end_policy, state):
        start_policy = torch.softmax(start_policy, dim=1).squeeze(0).cpu().numpy()
        end_policy = torch.softmax(end_policy, dim=1).squeeze(0).cpu().numpy()

        valid_moves = self.game.get_valid_moves(state)

        policy = np.zeros((len(start_policy), len(end_policy)))
        for start_idx, start_prob in enumerate(start_policy):
            for end_idx, end_prob in enumerate(end_policy):
                action = np.array([[start_idx // 8, start_idx % 8], [end_idx // 8, end_idx % 8]], dtype=np.uint8)
                if any(np.array_equal(action, move) for move in valid_moves):
                    policy[start_idx, end_idx] = start_prob * end_prob

        return policy

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken, node.is_draw)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                start_policy, end_policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = self.get_policy_matrix(start_policy, end_policy, node.state)

                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros((self.game.action_size, self.game.action_size + self.game.promotion_size))
        for child in root.children:
            start_coord, end_coord = child.action_taken[0], child.action_taken[1]
            start_index = start_coord[0] * 8 + start_coord[1]
            end_index = end_coord[0] * 8 + end_coord[1]
            action_probs[start_index, end_index] = child.visit_count
        # normalization in alphazero
        return action_probs
