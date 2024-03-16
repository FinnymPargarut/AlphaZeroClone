import yaml
import torch
import numpy as np

from AlphaMCTS import MCTS
from TicTacToe import TicTacToe
from Resnet import ResNet
from AlphaZero import AlphaZero


def alphaZero_learn_tictactoe():
    with open("config.yml", 'r') as options:
        args = yaml.safe_load(options)["args"]

    tictactoe = TicTacToe()
    model = ResNet(tictactoe, 4, 64)
    # model.load_state_dict(torch.load("models/model_2.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer.load_state_dict(torch.load("models/optimizer_2.pt"))

    alphaZero = AlphaZero(model, optimizer, tictactoe, args)
    alphaZero.learn()


def play_tictactoe():
    game = TicTacToe()
    player = 1

    args = {
        'exploration_constant': 2,
        'num_searches': 600,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
    }

    model = ResNet(game, 4, 64)
    model.load_state_dict(torch.load("models/model_2.pt"))
    model.eval()

    mcts = MCTS(game, args, model)

    state = game.get_initial_state()

    while True:
        print(state)

        if player == 1:
            valid_moves = game.get_valid_moves(state)
            print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("action not valid")
                continue

        else:
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        state = game.get_next_state(state, action, player)

        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break

        player = game.get_opponent(player)


class PseudoAgent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        self.mcts = MCTS(self.game, self.args, self.model)

    def get_action(self, state, player):
        state = self.game.change_perspective(state, player)

        policy = self.mcts.search(state)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)

        if self.args['temperature'] == 0:
            action = int(np.argmax(policy))
        elif self.args['temperature'] == float('inf'):
            action = np.random.choice([r for r in range(self.game.action_size) if policy[r] > 0])
        else:
            policy = policy ** (1 / self.args['temperature'])
            policy /= np.sum(policy)
            action = np.random.choice(self.game.action_size, p=policy)

        return action


def simulate_games(game, model1, model2, args, num_games=50):
    player1 = PseudoAgent(model1, game, args)
    player2 = PseudoAgent(model2, game, args)

    wins = [0, 0]
    draws = 0

    for i in range(num_games):
        state = game.get_initial_state()
        player = 1

        while True:
            if player == 1:
                action = player1.get_action(state, player)
            elif player == -1:
                action = player2.get_action(state, player)
            else:
                action = 0
                print("There is a problem with the player!")

            state = game.get_next_state(state, action, player)

            value, is_terminal = game.get_value_and_terminated(state, action)

            if is_terminal:
                if value == 1:
                    winner = 0 if player == 1 else 1
                    wins[winner] += 1
                else:
                    draws += 1
                break

            player = game.get_opponent(player)

    return wins, draws


def models_play():
    game = TicTacToe()
    args = {
        'exploration_constant': 2,
        'num_searches': 100,
        'dirichlet_epsilon': 0.1,
        'dirichlet_alpha': 0.3,
        'temperature': 0,
    }

    model1 = ResNet(game, 4, 64)
    model1.load_state_dict(torch.load("models/model_2.pt"))
    model1.eval()

    model2 = ResNet(game, 4, 64)
    model2.load_state_dict(torch.load("models/model_3.pt"))
    model2.eval()

    wins1, draws1 = simulate_games(game, model1, model2, args)
    wins2, draws2 = simulate_games(game, model2, model1, args)

    total_wins = [wins1[i] + wins2[i] for i in range(2)]
    total_draws = draws1 + draws2

    print(total_wins, total_draws)


if __name__ == '__main__':
    models_play()
