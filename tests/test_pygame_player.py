import time
import pygame
from unittest import TestCase
from pygame_player import PyGamePlayer


class DummyPyGamePlayer(PyGamePlayer):
    def __init__(self, force_game_fps=10, run_real_time=False):
        super(DummyPyGamePlayer, self).__init__(force_game_fps=force_game_fps, run_real_time=run_real_time)

    def get_keys_pressed(self, screen_array, feedback, terminal):
        pass

    def get_feedback(self):
        return 0.0, False


class TestPyGamePlayer(TestCase):
    DISPLAY_X = 1
    DISPLAY_Y = 1

    def setUp(self):
        pygame.init()
        pygame.display.set_mode((self.DISPLAY_X, self.DISPLAY_Y), 0, 32)

    def tearDown(self):
        pygame.quit()

    def test_restores_pygame_methods_after_exit(self):
        pygame_flip, pygame_update, pygame_event = pygame.display.flip, pygame.display.update, pygame.event.get

        with PyGamePlayer():
            # methods should be replaced
            self.assertNotEqual(pygame_flip, pygame.display.flip)
            self.assertNotEqual(pygame_update, pygame.display.update)
            self.assertNotEqual(pygame_event, pygame.event.get)

        # original methods should be restored
        self.assertEqual(pygame_flip, pygame.display.flip)
        self.assertEqual(pygame_update, pygame.display.update)
        self.assertEqual(pygame_event, pygame.event.get)

    def test_fixing_frames_per_second(self):
        fix_fps_to = 3
        with DummyPyGamePlayer(force_game_fps=fix_fps_to):
            clock = pygame.time.Clock()
            start_time_ms = clock.get_time()

            for _ in range(fix_fps_to):
                pygame.display.update()

            end_time_ms = clock.get_time()

        self.assertAlmostEqual(end_time_ms - start_time_ms, 1000.0,
                               msg='Expected only 1000 milliseconds to have passed on the clock after screen updates')

    def test_get_keys_pressed_method_sets_event_get(self):
        fixed_key_pressed = 24

        class FixedKeysReturned(DummyPyGamePlayer):
            def get_keys_pressed(self, screen_array, feedback, terminal):
                return [fixed_key_pressed]

        with FixedKeysReturned():
            pygame.display.update()
            key_pressed = pygame.event.get()

        self.assertEqual(key_pressed[0].key, fixed_key_pressed)

    def test_get_screen_buffer(self):
        class TestScreenArray(DummyPyGamePlayer):
            def get_keys_pressed(inner_self, screen_array, feedback, terminal):
                self.assertEqual(screen_array.shape[0], self.DISPLAY_X)
                self.assertEqual(screen_array.shape[1], self.DISPLAY_Y)

        with TestScreenArray():
            pygame.display.update()

    def test_run_real_time(self):
        fix_fps_to = 3

        with PyGamePlayer(force_game_fps=fix_fps_to, run_real_time=True):
            start = time.time()

            clock = pygame.time.Clock()
            for _ in range(fix_fps_to):
                clock.tick(42343)

            end = time.time()

        self.assertAlmostEqual(end-start, 1.0, delta=0.1)
