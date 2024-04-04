import torch
import numpy as np

from AlphaMCTS import Node


class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def get_policy_matrix(self, start_policy, end_policy, state):
        valid_moves = self.game.get_valid_moves(state)

        policy = np.zeros((len(start_policy), len(end_policy)))
        for start_idx, start_prob in enumerate(start_policy):
            for end_idx, end_prob in enumerate(end_policy):
                action = np.array([[start_idx // 8, start_idx % 8], [end_idx // 8, end_idx % 8]], dtype=np.uint8)
                if any(np.array_equal(action, move) for move in valid_moves):
                    policy[start_idx, end_idx] = start_prob * end_prob

        return policy

    @torch.no_grad()
    def search(self, states, spGames):
        for i, spg in enumerate(spGames):
            spg.root = Node(self.game, self.args, states[i])

        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)

                else:
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if
                                  spGames[mappingIdx].node is not None]

            start_policy, end_policy, value = None, None, None  # reference
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])

                encoded_states = np.array([self.game.get_encoded_state(state) for state in states])
                start_policy, end_policy, value = self.model(
                    torch.tensor(encoded_states, device=self.model.device)
                )
                start_policy = torch.softmax(start_policy, dim=1).cpu().numpy()
                end_policy = torch.softmax(end_policy, dim=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_start_policy, spg_end_policy, spg_value = start_policy[i], end_policy[i], value[i]

                spg_policy = self.get_policy_matrix(spg_start_policy, spg_end_policy, node.state)

                node.expand(spg_policy)
                node.backpropagate(spg_value)