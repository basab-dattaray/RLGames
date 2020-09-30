import copy
import inspect
import logging
import os
import sys
from collections import deque, namedtuple
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np


from ws.RLAgents.self_play.alpha_zero.play.Arena import Arena
from ws.RLAgents.self_play.alpha_zero.search.MctsSelector import MctsSelector
# from ws.RLAgents.self_play.alpha_zero.search.recursive.MCTS import MCTS
from ws.RLUtils.monitoring.tracing.tracer import tracer

log = logging.getLogger(__name__)


def coach(game, nnet, args):
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in demo_train.py.
    """

    DEBUG_FLAG = False

    # pnet = nnet.__class__(args, game)  # the competitor network
    pnet = copy.deepcopy(nnet)

    mcts = MctsSelector(game, nnet, args)
    trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
    skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode():
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board_this = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = game.getCanonicalForm(board_this, curPlayer)
            temp = int(episodeStep < args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            if pi is None:
                return None

            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board_next, player_next = game.getNextState(board_this, curPlayer, action)

            if DEBUG_FLAG:
                print()
                print('player:{}'.format(curPlayer))
                print()
                print(board_next)

            curPlayer = player_next
            board_this = board_next

            r = game.getGameEnded(board_this, curPlayer)

            if r != 0 or player_next is None:
                # return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
                return fn_form_sample_data(curPlayer, r, trainExamples)

    def fn_form_sample_data(current_player, run_result, training_samples, early_termination=False):
        sample_data = []
        for x in training_samples:
            state, player, action_prob = x[0], x[1], x[2]
            result = run_result * (1 if (player == current_player) else -1)
            sample_data.append([state, action_prob, result])
        return sample_data

    @tracer(args)
    def fn_learn():
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for iteration in range(1, args.numIters + 1):
            fn_run_iteration(iteration)


    @tracer(args)
    def fn_run_iteration(iteration):

        args.recorder.fn_record_message(f'-- Iter {iteration} of {args.numIters}',
                                             indent=0)
        # bookkeeping
        trainExamples = fn_generate_samples(iteration)
        # training new network, keeping a copy of the old one
        nnet.save_checkpoint(rel_folder=args.checkpoint, filename='temp.pth.tar')
        pnet.load_checkpoint(rel_folder=args.checkpoint, filename='temp.pth.tar')
        pmcts = MctsSelector(game, pnet, args)

        nnet.fn_adjust_model_from_examples(trainExamples)

        nmcts = MctsSelector(game, nnet, args)

        args.recorder.fn_record_message()
        args.recorder.fn_record_message(f'* Comptete with Previous Version', indent=0)

        arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                      lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), game)
        pwins, nwins, draws = arena.playGames(args.arenaCompare)
        # args.recorder.fn_record_message(f'pwins:{pwins} nwins:{nwins} draws:{draws}          Update Threshold: {args.updateThreshold}')
        update_threshold = 'update threshold: {}'.format(args.updateThreshold)
        args.recorder.fn_record_message(update_threshold)

        score = f'nwins:{nwins} pwins:{pwins} draws:{draws}'
        args.recorder.fn_record_message(score)

        if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < args.updateThreshold:
            # args.fn_record('REJECTING NEW MODEL')
            args.recorder.fn_record_message('REJECTED New Model')
            nnet.load_checkpoint(rel_folder=args.checkpoint, filename='temp.pth.tar')
        else:
            # args.fn_record('ACCEPTING NEW MODEL')
            args.recorder.fn_record_message('ACCEPTED New Model')
            nnet.save_checkpoint(rel_folder=args.checkpoint, filename=getCheckpointFile(iteration))
            nnet.save_checkpoint(rel_folder=args.checkpoint, filename='best.pth.tar')

    @tracer(args)
    def fn_generate_samples(iteration):
        # examples of the iteration
        if not skipFirstSelfPlay or iteration > 1:
            iterationTrainExamples = deque([], maxlen=args.maxlenOfQueue)

            for episode_num in range(1, args.numEps + 1):
                args.recorder.fn_record_message(f'Episode {episode_num} of {args.numEps}')

                mcts = MctsSelector(game, nnet, args)  # reset search tree
                episode_result = executeEpisode()
                if episode_result is not None:
                    iterationTrainExamples += episode_result

            # save the iteration examples to the history
            trainExamplesHistory.append(iterationTrainExamples)
        if len(trainExamplesHistory) > args.numItersForTrainExamplesHistory:
            log.warning(
                f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(trainExamplesHistory)}")
            trainExamplesHistory.pop(0)
        # backup history to a file
        # NB! the examples were collected using the model from the previous iteration, so (iteration-1)
        saveTrainExamples(iteration - 1)
        # shuffle examples before training
        trainExamples = []
        for e in trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)

        return trainExamples

    def getCheckpointFile(iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(iteration):
        folder = args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(trainExamplesHistory)
        # f.closed

    def loadTrainExamples():
        modelFile = os.path.join(args.load_folder_file[0], args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            args.fn_record("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                trainExamplesHistory = Unpickler(f).load()
            args.fn_record('Loading done!')

            # examples based on the model were already collected (loaded)
            skipFirstSelfPlay = True

    ret_refs = namedtuple('_', ['fn_learn','loadTrainExamples' ,'fn_test_againt_random'])

    ret_refs.fn_learn = fn_learn
    ret_refs.loadTrainExamples = loadTrainExamples

    return ret_refs