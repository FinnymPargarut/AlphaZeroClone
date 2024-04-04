import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import trange

from AlphaMCTSParallel import MCTSParallel


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        states_counter = [{len(self.game.get_initial_state()): 1} for spg in range(self.args['num_parallel_games'])]
        move_count_for_draw = [0 for spg in range(self.args['num_parallel_games'])]
        is_draw = [False for spg in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])

            neutral_states = []
            for i in range(len(states)):
                neutral_states.append(self.game.change_perspective(states[i], player))
            neutral_states = np.array(neutral_states)

            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:
                print("start")
                spg = spGames[i]

                action_probs = np.zeros((self.game.action_size, self.game.action_size + self.game.promotion_size))
                for child in spg.root.children:
                    start_coord, end_coord = child.action_taken[0], child.action_taken[1]
                    start_index = start_coord[0] * 8 + start_coord[1]
                    end_index = end_coord[0] * 8 + end_coord[1]
                    action_probs[start_index, end_index] = child.visit_count
                flat_action_probs = action_probs.flatten()
                flat_action_probs /= np.sum(flat_action_probs)

                spg.memory.append((spg.root.state, flat_action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                # temperature_action_probs /= np.sum(temperature_action_probs)
                start_end = np.random.choice(len(temperature_action_probs), p=temperature_action_probs)
                start, end = np.unravel_index(start_end, action_probs.shape)
                action = np.array([[start // 8, start % 8], [end // 8, end % 8]], dtype=np.uint8)

                # Get next state with check draw
                piece = spg.state[start // 8, start % 8]
                piece_after = spg.state[end // 8, end % 8] if end // 8 <= 7 else 100
                spg.state = self.game.get_next_state(spg.state, action, player, is_training=True)
                is_draw[i] = self.game.check_state_for_draw(spg.state, states_counter[i], is_draw[i])
                move_count_for_draw[i], is_draw[i] = self.game.check_moves_for_draw(piece, piece_after,
                                                                                    move_count_for_draw[i], is_draw[i])

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]

            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_start_policy, out_end_policy, out_value = self.model(state)
            out_policy = torch.empty_like(policy_targets)
            for batch in range(out_start_policy.shape[0]):
                for i in range(out_start_policy.shape[1]):
                    for j in range(out_end_policy.shape[1]):
                        out_policy[batch, i * out_end_policy.shape[1] + j] = out_start_policy[batch, i] * \
                                                                             out_end_policy[batch, j]

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")


class SPG:  # SelfPlayGames
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None