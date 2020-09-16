import inspect
import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np


from ws.RLAgents.self_play.alpha_zero.play.Arena import Arena
from ws.RLAgents.self_play.alpha_zero.search.MctsSelector import MctsSelector
# from ws.RLAgents.self_play.alpha_zero.search.recursive.MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in demo_train.py.
    """

    DEBUG_FLAG = False

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(args, self.game)  # the competitor network
        self.args = args
        self.mcts = MctsSelector(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
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
        board_this = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board_this, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            if pi is None:
                return None

            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board_next, player_next = self.game.getNextState(board_this, self.curPlayer, action)

            if Coach.DEBUG_FLAG:
                print()
                print('player:{}'.format(self.curPlayer))
                print()
                print(board_next)

            self.curPlayer = player_next
            board_this = board_next

            r = self.game.getGameEnded(board_this, self.curPlayer)

            if r != 0 or player_next is None:
                # return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
                return self.fn_form_sample_data(self.curPlayer, r, trainExamples)

    def fn_form_sample_data(self, current_player, run_result, training_samples, early_termination=False):
        sample_data = []
        for x in training_samples:
            state, player, action_prob = x[0], x[1], x[2]
            result = run_result * (1 if (player == current_player) else -1)
            sample_data.append([state, action_prob, result])
        return sample_data

    def fn_learn(self):
        self.args.recorder.fn_record_func_title_begin(inspect.stack()[0][3])
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for iteration in range(1, self.args.numIters + 1):
            self.fn_run_iteration(iteration)

        self.args.recorder.fn_record_func_title_end()

    def fn_run_iteration(self, iteration):
        self.args.recorder.fn_record_func_title_begin(inspect.stack()[0][3])

        self.args.recorder.fn_record_message(f'-- Iter {iteration} of {self.args.numIters}',
                                             indent=0)
        # bookkeeping
        trainExamples = self.fn_generate_samples(iteration)
        # training new network, keeping a copy of the old one
        self.nnet.save_checkpoint(rel_folder=self.args.checkpoint, filename='temp.pth.tar')
        self.pnet.load_checkpoint(rel_folder=self.args.checkpoint, filename='temp.pth.tar')
        pmcts = MctsSelector(self.game, self.pnet, self.args)

        self.nnet.fn_adjust_model_from_examples(trainExamples)

        nmcts = MctsSelector(self.game, self.nnet, self.args)

        self.args.recorder.fn_record_message()
        self.args.recorder.fn_record_message(f'* Comptete with Previous Version', indent=0)

        arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                      lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
        pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
        # self.args.recorder.fn_record_message(f'pwins:{pwins} nwins:{nwins} draws:{draws}          Update Threshold: {self.args.updateThreshold}')
        update_threshold = 'update threshold: {}'.format(self.args.updateThreshold)
        self.args.recorder.fn_record_message(update_threshold)

        score = f'nwins:{nwins} pwins:{pwins} draws:{draws}'
        self.args.recorder.fn_record_message(score)

        if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
            # self.args.fn_record('REJECTING NEW MODEL')
            self.args.recorder.fn_record_message('REJECTED New Model')
            self.nnet.load_checkpoint(rel_folder=self.args.checkpoint, filename='temp.pth.tar')
        else:
            # self.args.fn_record('ACCEPTING NEW MODEL')
            self.args.recorder.fn_record_message('ACCEPTED New Model')
            self.nnet.save_checkpoint(rel_folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration))
            self.nnet.save_checkpoint(rel_folder=self.args.checkpoint, filename='best.pth.tar')

        self.args.recorder.fn_record_func_title_end()

    def fn_generate_samples(self, iteration):
        self.args.recorder.fn_record_func_title_begin(inspect.stack()[0][3])

        # examples of the iteration
        if not self.skipFirstSelfPlay or iteration > 1:
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            for episode_num in range(1, self.args.numEps + 1):
                self.args.recorder.fn_record_message(f'Episode {episode_num} of {self.args.numEps}')

                self.mcts = MctsSelector(self.game, self.nnet, self.args)  # reset search tree
                episode_result = self.executeEpisode()
                if episode_result is not None:
                    iterationTrainExamples += episode_result

            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)
        if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            log.warning(
                f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
            self.trainExamplesHistory.pop(0)
        # backup history to a file
        # NB! the examples were collected using the model from the previous iteration, so (iteration-1)
        self.saveTrainExamples(iteration - 1)
        # shuffle examples before training
        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)

        self.args.recorder.fn_record_func_title_end()
        return trainExamples

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        # f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            self.args.fn_record("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            self.args.fn_record('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
