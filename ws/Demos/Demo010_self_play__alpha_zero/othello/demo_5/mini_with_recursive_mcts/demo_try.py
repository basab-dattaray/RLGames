from ws.Demos.Demo010_self_play__alpha_zero.othello.demo_5.mini_with_recursive_mcts.ARGS import args
from ws.RLAgents.CAT4_self_play.alpha_zero.misc.agent_mgt import agent_mgt

if __name__ == "__main__":
    agent_mgt(args, __file__). \
    fn_show_args(). \
    fn_change_args({
#
    }). \
    fn_show_args(). \
    fn_archive_log_file()
