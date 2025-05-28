import cv2
import numpy as np
from PIL import Image


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        """
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        """
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=None, y=None):
        """Move the blob.

        If ``x`` or ``y`` is ``None``, the respective coordinate will move in a
        random direction ``[-1, 0, 1]``. A value of ``0`` means no movement in
        that axis.
        """

        # If no value for x, move randomly
        if x is None:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if y is None:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size - 1:
            self.x = self.size - 1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size - 1:
            self.y = self.size - 1


class BlobEnv:
    SIZE = 10
    # MOVE_PENALTY = -1
    # Provide positive feedback for moving toward the food so PPO
    # has a clearer gradient signal during training.
    MOVE_CLOSER_REWARD = 1
    MOVE_FARTHER_REWARD = -1
    ENEMY_PENALTY = -300
    FOOD_REWARD = 25
    good_move = False

    # OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    EMPTY_N = 0
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    # BLUE GREEN RED FOR OPEN CV SOMEREASON
    d = {
        1: (255, 0, 0),  # player blue
        2: (0, 255, 0),  # food green
        3: (0, 0, 255),
    }  # enemy red
    episode_step = 0
    done = False

    def dist(self, blob_a, blob_b):
        return np.sqrt((blob_a.x - blob_b.x) ** 2 + (blob_a.y - blob_b.y) ** 2)

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)

        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0
        self.done = False

        # observation = [self.player.x, self.player.y,
        #                self.food.x, self.food.y,
        #                self.enemy.x, self.enemy.y]
        observation = [
            self.player.x - self.food.x,
            self.player.y - self.food.y,
            self.player.x - self.enemy.x,
            self.player.y - self.enemy.y,
        ]
        return observation

    def step(self, action):
        self.episode_step += 1
        # clear board of player position
        # self.board[self.player.y][self.player.x] = self.EMPTY_N

        dist_before = self.dist(self.player, self.food)
        self.player.action(action)
        # player may be on food or enemy, board does NOT effect rendering

        dist_after = self.dist(self.player, self.food)
        # IMAGE RENDERING STUFF

        new_observation = [
            self.player.x - self.food.x,
            self.player.y - self.food.y,
            self.player.x - self.enemy.x,
            self.player.y - self.enemy.y,
        ]
        # [self.player.x, self.player.y,
        #                    self.food.x, self.food.y,
        #                    self.enemy.x, self.enemy.y]

        if self.player == self.enemy:
            reward = self.ENEMY_PENALTY
            self.good_move = False
        elif self.player == self.food:
            self.good_move = True

            reward = self.FOOD_REWARD
        else:
            if dist_after < dist_before:
                reward = self.MOVE_CLOSER_REWARD
                self.good_move = True
            else:
                reward = self.MOVE_FARTHER_REWARD
                self.good_move = False

        self.done = False
        if reward == self.FOOD_REWARD or reward == self.ENEMY_PENALTY or self.episode_step >= 200:
            self.done = True

        return new_observation, reward, self.done

    def render(self):
        step = self.episode_step
        board = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        pause = 200  # 0.05 seconds for normal movement,
        long_pause = 1000
        # SCREEN GREEN
        if self.player == self.food:
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    if abs(i - self.player.y) + abs(j - self.player.x) <= 4:
                        board[i, j, :] = self.d[self.FOOD_N]

        # SCREEN RED
        elif self.player == self.enemy:
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    board[i, j, :] = self.d[self.ENEMY_N]
        # Normal moving
        else:
            if self.good_move:
                board[self.player.y, self.player.x, :] = self.d[self.PLAYER_N]
            else:
                board[self.player.y, self.player.x, :] = (150, 150, 150)

            board[self.food.y, self.food.x, :] = self.d[self.FOOD_N]
            board[self.enemy.y, self.enemy.x, :] = self.d[self.ENEMY_N]

        img = Image.fromarray(board, "RGB")

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("unique_title", np.array(img))  # show it!
        cv2.setWindowTitle("unique_title", f"step = {step}")

        if self.done:
            cv2.waitKey(long_pause)
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # needed to actually close image, could just leave image up too?
        else:
            cv2.waitKey(pause)


def main():
    """Run a simple random policy in the Blob environment."""
    env = BlobEnv()
    env.reset()
    env.render()
    done = False
    while not done:
        action = np.random.randint(0, env.ACTION_SPACE_SIZE)
        _, _, done = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
