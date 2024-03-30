from helpers import play_tictactoe, models_play_tictactoe

from Chess import Chess
from Resnet import ResNet
from AlphaMCTS import MCTS
from AlphaZero import AlphaZero
import numpy as np
import torch
import yaml


def alphaZero_learn_chess():
    with open("config.yml", 'r') as options:
        args = yaml.safe_load(options)["args"]

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
            print(max_indices)
            start, end = max_indices[0]
            print(max_indices, start, end)
            # 49 41
            # 6 1 5 1
            action = np.array([[start // 8, start % 8], [end // 8, end % 8]], dtype=np.uint8)
            print(action)

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

    args = {
        'exploration_constant': 2,
        'num_searches': 400,
    }

    model = ResNet(game, 8, 128)
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


if __name__ == '__main__':
    # models_play_tictactoe("models/model_2.pt", "models/model_3.pt")
    # play_chess_multiplayer()
    # play_chess_with_model()
    alphaZero_learn_chess()
