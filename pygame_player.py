import pygame
import numpy  # import is unused but required or we fail later
from pygame.constants import K_DOWN, K_UP, KEYDOWN, KEYUP, QUIT
import pygame.surfarray
import pygame.key


def function_intercept(intercepted_func, intercepting_func):
    """
    Intercepts a method call and calls the supplied intercepting_func with the result of it's call and it's arguments

    Example:
        def get_event(result_of_real_event_get, *args, **kwargs):
            # do work
            return result_of_real_event_get

        pygame.event.get = function_intercept(pygame.event.get, get_event)

    :param intercepted_func: The function we are going to intercept
    :param intercepting_func:   The function that will get called after the intercepted func. It is supplied the return
    value of the intercepted_func as the first argument and it's args and kwargs.
    :return: a function that combines the intercepting and intercepted function, should normally be set to the
             intercepted_functions location
    """

    def wrap(*args, **kwargs):
        real_results = intercepted_func(*args, **kwargs)  # call the function we are intercepting and get it's result
        intercepted_results = intercepting_func(real_results, *args, **kwargs)  # call our own function a
        return intercepted_results

    return wrap


class PyGamePlayer(object):
    def __init__(self, force_game_fps=10, run_real_time=False, pass_quit_event=True):
        """
        Abstract class for learning agents, such as running reinforcement learning neural nets against PyGame games.

        The get_keys_pressed and get_feedback methods must be overriden by a subclass to use

        Call start method to start playing intercepting PyGame and training our machine
        :param force_game_fps: Fixes the pygame timer functions so the ai will get input as if it were running at this
                               fps
        :type force_game_fps: int
        :param run_real_time: If True the game will actually run at the force_game_fps speed
        :type run_real_time: bool
        :param pass_quit_event: If True the ai will be asked for the quit event
        :type pass_quit_event: bool
        """
        self.force_game_fps = force_game_fps
        """Fixes the pygame timer functions so the ai will get input as if it were running at this fps"""
        self.run_real_time = run_real_time
        """If True the game will actually run at the force_game_fps speed"""
        self.pass_quit_event = pass_quit_event
        """Decides whether the quit event should be passed on to the game"""
        self._keys_pressed = []
        self._last_keys_pressed = []
        self._playing = False
        self._default_flip = pygame.display.flip
        self._default_update = pygame.display.update
        self._default_event_get = pygame.event.get
        self._default_time_clock = pygame.time.Clock
        self._default_get_ticks = pygame.time.get_ticks
        self._game_time = 0.0

    def get_keys_pressed(self, screen_array, feedback, terminal):
        """
        Called whenever the screen buffer is refreshed. returns the keys we want pressed in the next until the next
        screen refresh

        :param screen_array: 3d numpy.array of float. screen_width * screen_height * rgb
        :param feedback: result of call to get_feedback
        :param terminal: boolean, True if we have reached a terminal state, meaning the next frame will be a restart
        :return: a list of the integer values of the keys we want pressed. See pygame.constants for values
        """
        raise NotImplementedError("Please override this method")

    def get_feedback(self):
        """
        Overriden method should hook into game events to give feeback to the learning agent

        :return: First = value we want to give as reward/punishment to our learning agent
                 Second = Boolean true if we have reached a terminal state
        :rtype: tuple (float, boolean)
        """
        raise NotImplementedError("Please override this method")

    def start(self):
        """
        Start playing the game. We will now start listening for screen updates calling our play and reward functions
        and returning our intercepted key presses
        """
        if self._playing:
            raise Exception("Already playing")

        pygame.display.flip = function_intercept(pygame.display.flip, self._on_screen_update)
        pygame.display.update = function_intercept(pygame.display.update, self._on_screen_update)
        pygame.event.get = function_intercept(pygame.event.get, self._on_event_get)
        pygame.time.Clock = function_intercept(pygame.time.Clock, self._on_time_clock)
        pygame.time.get_ticks = function_intercept(pygame.time.get_ticks, self.get_game_time_ms)
        # TODO: handle pygame.time.set_timer...

        self._playing = True

    def stop(self):
        """
        Stop playing the game. Will try and return PyGame to the state it was in before we started
        """
        if not self._playing:
            raise Exception("Already stopped")

        pygame.display.flip = self._default_flip
        pygame.display.update = self._default_update
        pygame.event.get = self._default_event_get
        pygame.time.Clock = self._default_time_clock
        pygame.time.get_ticks = self._default_get_ticks

        self._playing = False

    @property
    def playing(self):
        """
        Returns if we are in a state where we are playing/intercepting PyGame calls
        :return: boolean
        """
        return self._playing

    @playing.setter
    def playing(self, value):
        if self._playing == value:
            return
        if self._playing:
            self.stop()
        else:
            self.start()

    def get_ms_per_frame(self):
        return 1000.0 / self.force_game_fps

    def get_game_time_ms(self):
        return self._game_time

    def _on_time_clock(self, real_clock, *args, **kwargs):
        return self._FixedFPSClock(self, real_clock)

    def _on_screen_update(self, _, *args, **kwargs):
        surface_array = pygame.surfarray.array3d(pygame.display.get_surface())
        reward, terminal = self.get_feedback()
        keys = self.get_keys_pressed(surface_array, reward, terminal)
        self._last_keys_pressed = self._keys_pressed
        self._keys_pressed = keys

        # now we have processed a frame increment the game timer
        self._game_time += self.get_ms_per_frame()

    def _on_event_get(self, _, *args, **kwargs):
        key_up_events = []
        if len(self._last_keys_pressed) > 0:
            diff_list = list(set( self._last_keys_pressed) - set(self._keys_pressed))
            key_up_events = [pygame.event.Event(KEYUP, {"key": x}) for x in diff_list] 

        key_down_events = [pygame.event.Event(KEYDOWN, {"key": x}) for x in self._keys_pressed]

        result = []

        # have to deal with arg type filters
        if args:
            if hasattr(args[0], "__iter__"):
                args = args[0]

            for type_filter in args:
                if type_filter == QUIT:
                    if type_filter == QUIT:
                        if self.pass_quit_event:
                            for e in _:
                                if e.type == QUIT:
                                    result.append(e)
                    else:
                        pass  # never quit
                elif type_filter == KEYUP:
                    result = result + key_up_events
                elif type_filter == KEYDOWN:
                    result = result + key_down_events
        else:
            result = key_down_events + key_up_events
            if self.pass_quit_event:
                for e in _:
                    if e.type == QUIT:
                        result.append(e)

        return result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    class _FixedFPSClock(object):
        def __init__(self, pygame_player, real_clock):
            self._pygame_player = pygame_player
            self._real_clock = real_clock

        def tick(self, _=None):
            if self._pygame_player.run_real_time:
                return self._real_clock.tick(self._pygame_player.force_game_fps)
            else:
                return self._pygame_player.get_ms_per_frame()

        def tick_busy_loop(self, _=None):
            if self._pygame_player.run_real_time:
                return self._real_clock.tick_busy_loop(self._pygame_player.force_game_fps)
            else:
                return self._pygame_player.get_ms_per_frame()

        def get_time(self):
            return self._pygame_player.get_game_time_ms()

        def get_raw_time(self):
            return self._pygame_player.get_game_time_ms()

        def get_fps(self):
            return int(1.0 / self._pygame_player.get_ms_per_frame())
