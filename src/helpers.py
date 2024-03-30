import yaml
import torch
import numpy as np


from AlphaMCTS import MCTS
from Chess import Chess
from Resnet import ResNet
from AlphaZero import AlphaZero


def alphaZero_learn_chess():
    with open("config.yml", 'r') as options:
        args = yaml.safe_load(options)["args_for_training"]

    game = Chess()
    device = torch.device("cpu")
    model = ResNet(game, 8, 128, device)
    # model.load_state_dict(torch.load("models/model_2.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer.load_state_dict(torch.load("models/optimizer_2.pt"))

    alphaZero = AlphaZero(model, optimizer, game, args)
    alphaZero.learn()


def play(game, mcts):
    state = game.get_initial_state()
    player = 1
    while True:
        print(state)

        if player == 1:
            valid_moves = game.get_valid_moves(state)

            for move in valid_moves:
                print(' '.join(str(m) for m in move))
            row, column, row_after, column_after = map(int, input(f"{player}:").split())
            action = np.array([[row, column], [row_after, column_after]], dtype=np.int8)

            if not any(np.array_equal(action, move) for move in valid_moves):
                print("action not valid")
                continue

            state = game.get_next_state(state, action, player)

        else:
            neutral_state = game.change_perspective(state, player)

            print("before probs")
            mcts_probs = mcts.search(neutral_state)
            print("after_probs")
            max_indices = np.argwhere(mcts_probs == np.max(mcts_probs))
            start, end = max_indices[0]
            action = np.array([[start // 8, start % 8], [end // 8, end % 8]], dtype=np.uint8)

            state = game.get_next_state(neutral_state, action, player)
            state = game.change_perspective(state, player)

        print(state, game.get_value_and_terminated(state, action))
        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break

        player = game.get_opponent(player)


def play_chess_with_model():
    game = Chess()

    with open("config.yml", 'r') as options:
        args = yaml.safe_load(options)["args_for_play_with_model"]

    device = torch.device("cpu")
    model = ResNet(game, 8, 128, device)
    model.load_state_dict(torch.load("models/test_model_chess.pt"))
    model.eval()

    mcts = MCTS(game, args, model)

    play(game, mcts)


def play_chess_multiplayer():
    game = Chess()
    state = game.get_initial_state()
    player = 1
    while True:
        print(state)

        neutral_state = game.change_perspective(state, player)

        valid_moves = game.get_valid_moves(neutral_state)

        for move in valid_moves:
            print(' '.join(str(m) for m in move))
        row, column, row_after, column_after = map(int, input(f"{player}:").split())
        action = np.array([[row, column], [row_after, column_after]], dtype=np.int8)

        if not any(np.array_equal(action, move) for move in valid_moves):
            print("action not valid")
            continue

        state = game.get_next_state(neutral_state, action, player)
        if player == -1:
            state = game.change_perspective(state, player)

        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
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
        policy = self.mcts.search(state)

        if self.args['temperature'] == 0:
            max_indices = np.argwhere(policy == np.max(policy))
            start, end = max_indices[0]
            action = np.array([[start // 8, start % 8], [end // 8, end % 8]], dtype=np.uint8)
        elif self.args['temperature'] == float('inf'):
            flat_policy = policy.flatten()
            action = np.random.choice([r for r in range(len(flat_policy)) if flat_policy[r] > 0])
        else:
            flat_policy = policy.flatten()
            policy = flat_policy ** (1 / self.args['temperature'])
            policy /= np.sum(policy)
            action = np.random.choice(self.game.action_size, p=policy)

        return action


def simulate_games(game, model1, model2, args, num_games):
    player1 = PseudoAgent(model1, game, args)
    player2 = PseudoAgent(model2, game, args)

    wins = [0, 0]
    draws = 0

    for i in range(num_games):
        game = Chess()
        state = game.get_initial_state()
        player = 1

        while True:
            print(state)
            state = game.change_perspective(state, player)
            if player == 1:
                action = player1.get_action(state, player)
            else:
                action = player2.get_action(state, player)
            state = game.get_next_state(state, action, player)
            state = game.change_perspective(state, player)

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


def models_play_chess(path_model_1, path_model_2, num_games=100):
    game = Chess()

    with open("config.yml", 'r') as options:
        args = yaml.safe_load(options)["args_for_models_play"]

    device = torch.device("cpu")
    model1 = ResNet(game, 8, 128, device)
    model1.load_state_dict(torch.load(path_model_1))
    model1.eval()

    model2 = ResNet(game, 8, 128, device)
    model2.load_state_dict(torch.load(path_model_2))
    model2.eval()

    wins1, draws1 = simulate_games(game, model1, model2, args, num_games=num_games//2)
    wins2, draws2 = simulate_games(game, model2, model1, args, num_games=num_games//2)

    total_wins = [wins1[i] + wins2[i] for i in range(2)]
    total_draws = draws1 + draws2

    print(total_wins, total_draws)
