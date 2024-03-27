from helpers import play_tictactoe, models_play_tictactoe

from Chess import Chess
from Resnet import ResNet
from AlphaZero import AlphaZero
import numpy as np
import torch
import yaml


def alphaZero_learn_chess():
    with open("config.yml", 'r') as options:
        args = yaml.safe_load(options)["args"]

    game = Chess()
    model = ResNet(game, 8, 128)
    # model.load_state_dict(torch.load("models/model_2.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer.load_state_dict(torch.load("models/optimizer_2.pt"))

    alphaZero = AlphaZero(model, optimizer, game, args)
    alphaZero.learn()


def play_chess():
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
    # play_chess()
    alphaZero_learn_chess()
