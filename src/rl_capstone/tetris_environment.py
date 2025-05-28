# base human game working through timer


### Code originally taken from:
# https://levelup.gitconnected.com/writing-tetris-in-python-2a16bddb5318
# highly modified
import pygame
import random
from . import tetris
import gymnasium
from gymnasium import spaces
import numpy as np


Tetris = tetris.Tetris


# Global variables
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
TRANSPARENCY = 50  # out of 255

# colors order is linked to figure order in figure.py, this seems like a potential problem
colors = [
    (255, 255, 255),  # white for empty
    (0, 255, 255),  # I- Cyan
    (0, 0, 255),  # J - Blue
    (255, 127, 0),  # L - Orange
    (255, 255, 0),  # O - Yellow
    (0, 255, 0),  # S - Green
    (128, 0, 128),  # T - Purple
    (255, 0, 0),  # Z - Red
    (255, 215, 0),  # GOLD, not implemented
    (194, 189, 176),  # SILVER, not implemented
]


class Tetris_Env(gymnasium.Env):
    size = (800, 800)
    done = False
    clock = pygame.time.Clock()
    fps = 60
    actions = {"no_op": 0, "left": 1, "right": 2, "down": 3, "cw": 4, "ccw": 5, "swap": 6, "hard": 7}
    # From gym documentation: https://www.gymlibrary.dev/content/environment_creation/
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    human_playing = False  # used to set text in display function

    def __init__(self, render_mode="rgb_array"):
        # KEY ATTRIBUTE
        self.game = Tetris(player=1)  # probably should be passed in
        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"'{render_mode}' is NOT a valid render mode. Use 'human' or 'rgb_array'"
        )
        self.render_mode = render_mode
        self.screen = None

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("Tetris")
            self.game_start_time = pygame.time.get_ticks()  # game start time
            self.cur_time = 0
            self._render_frame()
        ##################################

        # parameters for motion
        self.counter = 0
        # self.pressing_down = False
        # self.pressing_sideways = 0
        self.cur_time = 0

        # avoiding magic numbers
        NUM_TETROMINOS = 7
        NUM_ROTATIONS = 4

        # FROM DOCUMENTATION, adjusted for Tetris
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]), high=np.array([4, 200, 200, 200, 7, 7]), dtype=int
        )
        #         self.observation_space = spaces.Dict(
        #             {
        #                 # just use a 1 if piece is there
        #                 "board": spaces.Box(low=0, high=1,
        #                                     shape=(self.game.height, self.game.width),
        #                                     dtype=int),
        #                 "agent": spaces.Dict({
        #                         "x": spaces.Discrete(self.game.width),
        #                         "y": spaces.Discrete(self.game.height + self.game.buffer),
        #                         "piece": spaces.Discrete(NUM_TETROMINOS),
        #                         "rotation": spaces.Discrete(NUM_ROTATIONS)
        #                     }
        #                 ),
        #                 "queue": spaces.Box(low = 1, high = NUM_TETROMINOS,
        #                                     shape = (self.game.n_queue,),
        #                                     dtype = int),
        #                 "swap": spaces.Discrete(NUM_TETROMINOS+1), # in case empty (0)
        #                 "has_swapped": spaces.Discrete(2)

        #             }
        #         )

        self.action_space = spaces.Box(low=np.array([0, -1, -6]), high=np.array([1, 2, 4]), dtype=int)

    def _get_obs(self):
        if self.game.state == "gameover":
            return
        return self.game.get_next_states()

    # ONLY HERE BECAUSE GYM INDICATED THAT IT WAS NEEDED
    def _get_info(self):
        return {"score": self.game.score, "lines": self.game.lines}

    def reset(self, render_mode="rgb_array", player=1, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.render_mode = render_mode
        # FOR TRAINING, player is 1 and we have no control during rendering!
        if render_mode == "human":
            if not (self.screen):
                self.__init__(render_mode=render_mode)
            self.game.__init__(player=player)
            self._render_frame()
        else:
            self.game.__init__(player=player)

        observation = self._get_obs()
        info = self._get_info()
        self.done = False

        return observation, info

    # called only by agent!
    def step(self, action_list):
        # Move a sequence of actions forward
        # action should be a dict
        # where      (swap, rotations, shifts, hard drop)
        #              0 or 1, -1 to 2, -6 to 3, action 7
        # if action is sequence of moves to land block, will need to go through several frames
        if action_list[0] == 1:
            self.next_frame(action=self.actions["swap"])

        rot = action_list[1]
        if rot == -1:
            self.next_frame(action=self.actions["ccw"])
        elif rot == 1:
            self.next_frame(action=self.actions["cw"])
        elif rot == 2:
            self.next_frame(action=self.actions["cw"])
            if self.game.state != "gameover":
                self.next_frame(action=self.actions["cw"])
        shift = action_list[2]
        if shift < 0:
            for _ in range(abs(shift)):
                if self.game.state != "gameover":
                    self.next_frame(action=self.actions["left"])
        else:
            for _ in range(shift):
                if self.game.state != "gameover":
                    self.next_frame(action=self.actions["right"])

        lines_cleared = self.next_frame(action=self.actions["hard"])

        if self.game.state == "gameover":
            reward = -10
        else:
            reward = 4 + lines_cleared**2
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, self.game.state == "gameover", False, info

    # takes action in form of number to make game do something
    # NOTE that the action should always be passed in with self.actions['cw'] , or whatever the action is
    def do_naive_action(self, action=None):
        # actions = ['no_op', 'left', 'right', 'down', 'cw', 'ccw', 'swap', 'hard']
        assert not (action) or action in range(8), f"action = {action} is invalid, needs to be None or in [0,7]"

        lines = 0
        if not (action):  # if no action passed in, do something random
            action = random.randint(0, 6)  # -2 because want to skip hard, inclusive here
        if action == self.actions["no_op"]:  # no op
            pass
        elif action == self.actions["left"]:  # left
            self.game.go_side(-1)
        elif action == self.actions["right"]:  # go right
            self.game.go_side(1)
        elif action == self.actions["down"]:  # soft down
            lines = self.game.go_down()
        elif action == self.actions["cw"]:  # cw
            self.game.rotate(direction=1)
        elif action == self.actions["ccw"]:  # ccw
            self.game.rotate(direction=-1)
        elif action == self.actions["swap"]:  # swap
            self.game.swap()
        elif action == self.actions["hard"]:  # hard drop
            lines = self.game.go_space()
            self.counter = 0  # shouldn't matter except when it is fast

        return lines

    # PLAY GAME SHOULD ONLY BE CALLED BY HUMAN!!!!!
    def play_game(self):
        # RESET IS CALLED WHENEVER NEW GAME, regardless human or agent training
        self.reset(render_mode="human", player=0)
        self.human_playing = True
        # self.done would exit when you press q
        hit_close = False
        while not hit_close:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # self.done = True
                    hit_close = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.game.__init__(player=self.game.player)
                        self.cur_time = 0
                        self.game_start_time = pygame.time.get_ticks()
                    if event.key == pygame.K_p:
                        # self.game.change_player = True
                        self.game.player = (self.game.player + 1) % 2
                    if self.game.player == 1:
                        break  # should exit FOR loop if computer playing, don't take inputs
                    if event.key == pygame.K_l:
                        self.game.lines += 1
                        if self.game.lines % self.game.lines_per_level == 0:
                            self.game.level += 1
                            self.game.frames_per_drop = self.game.level_frames_per_drop[
                                min(self.game.level, self.game.max_level)
                            ]
                    if event.key == pygame.K_k:
                        self.game.level += 1
                        self.game.frames_per_drop = self.game.level_frames_per_drop[
                            min(self.game.level, self.game.max_level)
                        ]
                    if self.game.state == "gameover":
                        # self.pressing_down = False
                        # self.pressing_sideways = 0
                        break
                    if event.key == pygame.K_RSHIFT:
                        self.game.rotate(direction=1)
                    if event.key == pygame.K_SLASH:
                        self.game.rotate(direction=-1)
                    if event.key == pygame.K_DOWN:
                        self.game.go_down()
                        self.counter = 0
                        # self.pressing_down = True

                    if event.key == pygame.K_LEFT:
                        self.game.go_side(-1)
                        # self.pressing_sideways = -1
                    elif event.key == pygame.K_RIGHT:
                        # self.pressing_sideways = 1
                        self.game.go_side(1)
                    # else:
                    #     self.pressing_sideways = 0
                    if event.key == pygame.K_SPACE:
                        self.game.go_space()
                        self.counter = 0
                    if event.key in [
                        pygame.K_1,
                        pygame.K_2,
                        pygame.K_3,
                        pygame.K_4,
                        pygame.K_5,
                        pygame.K_6,
                        pygame.K_7,
                    ]:
                        self.game.new_figure(
                            mode=event.key - pygame.K_0
                        )  # ASSUME K_0 is 48 and rest of numbers go up by 1
                    if event.key == pygame.K_s:
                        self.game.swap()
            self.next_frame()
        self.close()

    def next_frame(self, action=None):
        #### ACTIONS COME FROM AGENT CHOOSING, NOT THE ENVIRONMENT
        # make computer choose random action- MAY NEED TO FIX FOR AGENT TRAINING
        # player 0 for human play
        # DON'T do anything if it's gameover

        lines = 0

        if self.game.player == 1:
            if self.game.state == "gameover":
                return
            lines = self.do_naive_action(action)

        # Player 0 will be getting inputs in play game function!
        # do_naive_action() would perform a random action if called with None

        # Up the counter, make the clock tick a frame
        # training mode will base moves on counter and not use clock (used in render)
        self.counter += 1
        if self.counter % self.game.frames_per_drop == 0 and self.game.state == "start":
            self.game.go_down()
            self.counter = 0

        if self.render_mode == "human":
            self._render_frame()

        return lines

    # check for user inputs if player == 0
    # update screen with surface
    # update screen with text
    def _render_frame(self):  # should pass in game and get rid of all the self.game nonsense
        self.screen.fill(WHITE)
        self.make_plots()
        self.display_text()
        # update screen
        pygame.display.flip()
        # make clock tick to make speed work
        self.clock.tick(self.fps)

    def make_plots(self):
        # Transparency for block shadow, code taken from: https://stackoverflow.com/questions/6339057/draw-a-transparent-rectangles-and-polygons-in-pygame
        def draw_rect_alpha(surface, color, rect):
            shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
            surface.blit(shape_surf, rect)

        # Drawing screen
        for i in range(self.game.buffer, self.game.full_height):
            for j in range(self.game.width):
                # WHAT IS ZOOM of 20 doing for the rectangle drawing
                pygame.draw.rect(
                    self.screen,
                    GRAY,
                    [
                        self.game.x + self.game.zoom * j,
                        self.game.y + self.game.zoom * i,
                        self.game.zoom,
                        self.game.zoom,
                    ],
                    width=3,
                )
                if self.game.board[i][j] > 0 and i >= self.game.buffer:
                    pygame.draw.rect(
                        self.screen,
                        colors[self.game.board[i][j]],
                        [
                            self.game.x + self.game.zoom * j + 1,
                            self.game.y + self.game.zoom * i + 1,
                            self.game.zoom - 2,
                            self.game.zoom - 1,
                        ],
                    )

        # UPDATE#
        if self.game.figure is not None:
            shadow_y = self.game.shadow_height()
            for ind in self.game.figure.image():
                i = ind // 4
                j = ind % 4
                p = i * 4 + j

                # Plotting of shadow piece
                if shadow_y + i >= self.game.buffer:
                    draw_rect_alpha(
                        self.screen,
                        tuple(list(colors[self.game.figure.piece]) + [TRANSPARENCY]),
                        [
                            self.game.x + self.game.zoom * (j + self.game.figure.x) + 1,
                            self.game.y + self.game.zoom * (i + shadow_y) + 1,
                            self.game.zoom - 2,
                            self.game.zoom - 2,
                        ],
                    )

                # Plotting of actual piece
                if self.game.figure.y + i >= self.game.buffer:
                    pygame.draw.rect(
                        self.screen,
                        colors[self.game.figure.piece],
                        [
                            self.game.x + self.game.zoom * (j + self.game.figure.x) + 1,
                            self.game.y + self.game.zoom * (i + self.game.figure.y) + 1,
                            self.game.zoom - 2,
                            self.game.zoom - 2,
                        ],
                    )

        # Plot SWAP piece if it has been set aside
        if self.game.swap_piece:
            for ind in self.game.swap_piece.image():
                i = ind // 4
                j = ind % 4
                pygame.draw.rect(
                    self.screen,
                    colors[self.game.swap_piece.piece],
                    [
                        self.game.swap_x + self.game.zoom * j,
                        self.game.swap_y + self.game.zoom * i,
                        self.game.zoom - 2,
                        self.game.zoom - 2,
                    ],
                )
        else:
            # draw something indicating what this spot is for
            pass

        fig_i = 0
        for fig in self.game.queue:
            for ind in fig.image():
                i = ind // 4
                j = ind % 4
                pygame.draw.rect(
                    self.screen,
                    colors[fig.piece],
                    [
                        self.game.queue_x + self.game.zoom * j,
                        self.game.queue_y + self.game.zoom * (i + fig_i * 5),  # testing coordinates
                        self.game.zoom - 2,
                        self.game.zoom - 2,
                    ],
                )
            fig_i += 1

    def display_text(self):
        # Displaying screen text
        font = pygame.font.SysFont("Calibri", 25, True, False)
        font1 = pygame.font.SysFont("Calibri", 65, True, False)
        text_score = font.render("Score: " + str(self.game.score), True, BLACK)
        text_lines = font.render("Lines: " + str(self.game.lines), True, BLACK)
        text_level = font.render("Level: " + str(self.game.level), True, BLACK)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press q", True, (255, 215, 0))
        text_swap = font.render("SWAP!", True, BLACK)
        text_queue = font.render("Queue:", True, BLACK)
        # text_reward = font.render(f'Reward: {round(self.game.get_reward(),2)}', True, BLACK)

        if self.game.player == 0:
            p = "Human"
        else:
            p = "Computer"
        text_player = font.render(f"{p}: 'p' to swap", True, (200, 50, 100))

        controlsX = 10
        controlsY = 300
        position = controlsX, controlsY
        font = pygame.font.SysFont("Calibri", 15)
        text_control = [
            "Controls",
            "/: CCW rotation",
            "rShift': CW rotation",
            "up,down,left,right: movement",
            "space: hard drop",
            "0-6: debug blocks",
            "s: swap",
            "q: restart game",
            "l: free line",
        ]
        label = []
        if self.human_playing:
            self.screen.blit(text_player, [400, 50])

            for line in text_control:
                label.append(font.render(line, True, GRAY))
            for line in range(len(label)):
                self.screen.blit(label[line], (position[0], position[1] + (line * 15) + (15 * line)))

        self.screen.blit(text_score, [100, 50])
        self.screen.blit(text_lines, [100, 100])
        self.screen.blit(text_level, [100, 150])

        self.screen.blit(text_swap, [50, 250])
        self.screen.blit(text_queue, [self.game.queue_x, self.game.queue_y - 50])
        # self.screen.blit(text_reward, [0,0])
        if self.game.state == "gameover":
            self.screen.blit(text_game_over, [250, 80])
            self.screen.blit(text_game_over1, [250, 140])
        else:
            # update time if game is still going
            seconds = (pygame.time.get_ticks() - self.game_start_time) / 1000
            text_timer = font.render(f"Time: {round(seconds)} s", True, BLACK)
            self.screen.blit(text_timer, [10, 50])

    def close(self):
        if self.screen is not None:
            print("\t~attempting to close screens~")
            pygame.display.quit()
            pygame.quit()
        # print(f'score for game was = {self.game.get_reward()}')
        print(f"after, screen = {self.screen}.")
