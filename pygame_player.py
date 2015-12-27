import pygame
import numpy  # imports unused but required or we fail later
from pygame.constants import K_DOWN, KEYDOWN, KEYUP, QUIT
import pygame.surfarray
import pygame.key


def function_intercept(intercepting_func, intercepted_func):
    """
    Intercepts a method call and calls the supplied intercepting_func with the result of it's call and it's arguments

    Example:
        def get_event(result_of_real_event_get, *args, **kwargs):
            # do work
            return result_of_real_event_get

        pygame.event.get = function_intercept(get_event, pygame.event.get)

    :param intercepting_func:   The function that will get called, is supplied the return value of the intercepted_func
                                as the first argument and
    :param intercepted_func: The function we are going to intercept
    :return: a function that combines the intercepting and intercepted function, should normally be set to the
             intercepted_functions location
    """
    def wrap(*args, **kwargs):
        real_results = intercepted_func(*args, **kwargs)
        intercepted_results = intercepting_func(real_results, *args, **kwargs)
        return intercepted_results

    return wrap


class PyGamePlayer(object):
    def __init__(self):
        """
        Abstract class for learning agents, such as running reinforcement learning neural nets against PyGame games.

        The get_keys_pressed and get_feedback methods must be overriden by a subclass to use

        Call start method to start playing intercepting PyGame and training our machine
        """
        self._keys_pressed = []
        self._last_keys_pressed = []
        self._playing = False
        self._default_flip = pygame.display.flip
        self._default_update = pygame.display.update
        self._default_event_get = pygame.event.get

    def get_keys_pressed(self, screen_array, feedback):
        """
        Called whenever the screen buffer is refreshed. returns the keys we want pressed in the next until the next
        screen refresh

        :param screen_array: 3d numpy.array of float. screen_width * screen_height * rgb
        :param feedback: result of call to get_feedback
        :return: a list of the integer values of the keys we want pressed. See pygame.constants for values
        """
        raise NotImplementedError("Please override this method")

    def get_feedback(self):
        """
        Overriden method should hook into game events to give feeback to the learning agent

        :return: value we want to give feedback, reward/punishment to our learning agent
        """
        raise NotImplementedError("Please override this method")

    def start(self):
        """
        Start playing the game. We will now start listening for screen updates calling our play and reward functions
        and returning our intercepted key presses
        """
        if self._playing:
            raise Exception("Already playing")

        self._default_flip = pygame.display.flip
        self._default_update = pygame.display.update
        self._default_event_get = pygame.event.get

        pygame.display.flip = function_intercept(self._on_screen_update, pygame.display.flip)
        pygame.display.update = function_intercept(self._on_screen_update, pygame.display.update)
        pygame.event.get = function_intercept(self._on_event_get, pygame.event.get)
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

        self._playing = False

    def _on_screen_update(self, real_method_result, *args, **kwargs):
        surface_array = pygame.surfarray.array3d(pygame.display.get_surface())
        reward = self.get_feedback()
        keys = self.get_keys_pressed(surface_array, reward)
        self._last_keys_pressed = self._keys_pressed
        self._keys_pressed = keys
        return None

    def _on_event_get(self, real_method_result, *args, **kwargs):
        key_down_events = [pygame.event.Event(KEYDOWN, {"key": x})
                           for x in self._keys_pressed if x not in self._last_keys_pressed]
        key_up_events = [pygame.event.Event(KEYUP, {"key": x})
                         for x in self._last_keys_pressed if x not in self._keys_pressed]

        result = []

        # have to deal with arg type filters
        if args:
            if hasattr(args[0], "__iter__"):
                args = args[0]

            for type in args:
                if type == QUIT:
                    pass #never quit
                elif type == KEYUP:
                    result = result + key_up_events
                elif type == KEYDOWN:
                    result = result + key_down_events
        else:
            result = key_down_events + key_up_events

        return result


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

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()