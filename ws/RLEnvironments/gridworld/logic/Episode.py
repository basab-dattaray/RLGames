

class Episode:
    KEY_DESTINATION_REACHED = 1
    KEY_DESTINATION_NOT_REACHED = 0
    KEY_DESTINATION_BLOCKED = -1

    def __init__(self):
        self._episodeStatus = Episode.KEY_DESTINATION_NOT_REACHED

    def fn_should_episode_continue(self):
        return True if self._episodeStatus == Episode.KEY_DESTINATION_NOT_REACHED else False

    def fn_update_episode(self, reward):
        self._episodeStatus = Episode.KEY_DESTINATION_NOT_REACHED if reward == 0 \
            else Episode.KEY_DESTINATION_REACHED if reward > 0 \
            else Episode.KEY_DESTINATION_BLOCKED

    def fn_get_episode_status(self):
        return self._episodeStatus

