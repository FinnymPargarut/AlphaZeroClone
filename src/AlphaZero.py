import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import trange

from AlphaMCTS import MCTS


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            flat_action_probs = action_probs.flatten()
            flat_action_probs /= np.sum(flat_action_probs)

            memory.append((neutral_state, flat_action_probs, player))

            temperature_action_probs = flat_action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)

            start_end = np.random.choice(len(temperature_action_probs), p=temperature_action_probs)
            start, end = np.unravel_index(start_end, action_probs.shape)
            action = np.array([[start // 8, start % 8], [end // 8, end % 8]], dtype=np.uint8)

            state = self.game.get_next_state(neutral_state, action, player)
            if player == -1:
                state = self.game.change_perspective(state, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)

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
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"models/model_chess_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"models/optimizer_chess_{iteration}.pt")