import logging

import coloredlogs

from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.mini_with_recursive_mcts.ARGS import args
from ws.RLAgents.self_play.alpha_zero.train.Coach import Coach
from ws.RLEnvironments.self_play_games.othello.OthelloGame import OthelloGame as Game
from ws.RLAgents.self_play.alpha_zero._game.othello._ml_lib.pytorch.NNet import NeuralNetWrapper
# from ws.RLAgents.self_play.alpha_zero.misc.utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(5)

    log.info('Loading %s...', NeuralNetWrapper.__name__)
    nnet = NeuralNetWrapper(args, g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
