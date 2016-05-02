# PyGamePlayer
Module to help with running learning agents against PyGame games. Hooks into the PyGame screen update and event.get methods so you can run PyGame games with zero touches to the underlying game file. Can even deal with games with no main() method.

Project contains three examples of running this, 2 a minimal examles one with Pong and one with Tetris and a full example of Deep-Q learning against Pong with tensorflow.

More information available here http://www.danielslater.net/2015/12/how-to-run-learning-agents-against.html

Requirements
----------
- python 2 or 3
- pygame
- numpy

Getting started
-----------
PyGame is probably the best supported library for games in Python it can be downloaded and installed from http://www.pygame.org/download.shtml

[Numpy](http://www.scipy.org/scipylib/download.html) is also required

Create a Python 2 or 3 environment with both of these in it.

Import this project and whatever PyGame game you want to train against into your working area. A bunch of PyGame games can be found here http://www.pygame.org/projects/6 or alternatly just use Pong or Tetris that are included with this project.

[exmples/deep_q_pong_player.py](https://github.com/DanielSlater/PyGamePlayer/blob/master/examples/deep_q_pong_player.py) also requires that [tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html) and [matplotlib](http://matplotlib.org/users/installing.html) be installed

Example usage for Pong game
-----------
```
from pygame_player import PyGamePlayer


class PongPlayer(PyGamePlayer):
    def __init__(self):
        super(PongPlayer, self).__init__(force_game_fps=10) 
        # force_game_fps fixes the game clock so that no matter how many real seconds it takes to run a fame 
        # the game behaves as if each frame took the same amount of time
        # use run_real_time so the game will actually play at the force_game_fps frame rate
        
        self.last_bar1_score = 0.0
        self.last_bar2_score = 0.0

    def get_keys_pressed(self, screen_array, feedback):
        # TODO: put an actual learning agent here
        from pygame.constants import K_DOWN
        return [K_DOWN] # just returns the down key

    def get_feedback(self):
        # import must be done here because otherwise importing would cause the game to start playing
        from games.pong import bar1_score, bar2_score

        # get the difference in score between this and the last run
        score_change = (bar1_score - self.last_bar1_score) - (bar2_score - self.last_bar2_score)
        self.last_bar1_score = bar1_score
        self.last_bar2_score = bar2_score

        return score_change


if __name__ == '__main__':
    player = PongPlayer()
    player.start()
```

Games
--------
- [Pong](https://github.com/DanielSlater/PyGamePlayer/blob/master/games/pong.py)
- [Tetris](https://github.com/DanielSlater/PyGamePlayer/blob/master/games/tetris.py)
- [Mini Pong](https://github.com/DanielSlater/PyGamePlayer/blob/master/games/mini_pong.py) - modified version of pong to run in lower resolutions
- [Half Pong](https://github.com/DanielSlater/PyGamePlayer/blob/master/games/half_pong.py) - simplified version of pong with just one bar 