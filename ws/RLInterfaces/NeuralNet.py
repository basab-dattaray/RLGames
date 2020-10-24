class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board_pieces.

    See othello/neural_net_mgt.py for an example implementation.
    """

    def __init__(self, game):
        pass

    def fn_adjust_model_from_examples(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board_pieces, action_probs, v). action_probs is the MCTS informed policy vector for
                      the given board_pieces, and v is its value. The examples has
                      board_pieces in its canonical form.
        """
        pass

    def predict(self, board):
        """
        Input:
            board_pieces: current board_pieces in its canonical form.

        Returns:
            action_probs: a policy vector for the current board_pieces- a numpy array of length
                game.fn_get_action_size
            v: a float in [-1,1] that gives the value of the current board_pieces
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass
