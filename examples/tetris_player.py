from pygame.constants import K_LEFT
from pygame_player import PyGamePlayer, function_intercept
import games.tetris


class TetrisPlayer(PyGamePlayer):
    def __init__(self):
        """
        Example class for playing Tetris
        """
        super(TetrisPlayer, self).__init__(desired_fps=5)
        self._toggle_down_key = True
        self._new_reward = 0.0

        def add_removed_lines_to_reward(lines_removed, *args, **kwargs):
            self._new_reward += lines_removed
            return lines_removed

        # to get the reward we will intercept the removeCompleteLines method and store what it returns
        games.tetris.removeCompleteLines = function_intercept(games.tetris.removeCompleteLines,
                                                              add_removed_lines_to_reward)

    def get_keys_pressed(self, screen_array, feedback):
        # TODO: put an actual learning agent here
        # toggle key presses so we get through the start menu

        if self._toggle_down_key:
            self._toggle_down_key = False
            return [K_LEFT]
        else:
            self._toggle_down_key = True
            return []

    def get_feedback(self):
        temp = self._new_reward
        self._new_reward = 0.0
        return temp

if __name__ == '__main__':
    with TetrisPlayer():
        games.tetris.main()