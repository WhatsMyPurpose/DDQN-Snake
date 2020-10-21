import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Snake:
    def __init__(self, starting_snake_size=3, grid_size_width=10):
        self.grid_size = grid_size_width
        self.snake_size = starting_snake_size
        self.action_space = 4
        self.state_space = (6,)
        self.max_score = self.grid_size**2-self.snake_size

    def reset(self):
        self.reward, self.total_reward, self.fruit_ate = 0, 0, 0
        self.prev_action = 2  # starts pointing up
        self.create_snake()
        self.drop_fruit()
        self.history = []
        self.history.append(self.get_grid())
        return self.get_state()

    def create_snake(self):
        from collections import deque
        headCenter = self.grid_size//2
        self.snake = deque([(headCenter, headCenter+i) for i in range(self.snake_size)])
        self.prev_head = (headCenter, headCenter)

    def move_snake(self, action):
        head = self.snake[0]
        self.prev_head = head
        self.prev_tail = self.snake[-1]
        if action == 0:  # left
            p = (head[0] - 1, head[1])
        elif action == 1:  # right
            p = (head[0] + 1, head[1])
        elif action == 2:  # up
            p = (head[0], head[1] - 1)
        elif action == 3:  # down
            p = (head[0], head[1] + 1)
        self.snake.appendleft(p)
        self.snake.pop()

    def grow(self):
        self.snake.append(self.prev_tail)

    def self_bite(self):
        return len(self.snake) > len(set(self.snake))

    def hit_border(self):
        return -1 in self.snake[0] or self.grid_size in self.snake[0]

    def drop_fruit(self):
        if len(self.snake) >= (self.grid_size - 2) ** 2:
            self.fruit = (-1, -1)
            pass
        while True:
            fruit = np.random.randint(0, self.grid_size, 2)
            fruit = (fruit[0], fruit[1])
            if fruit in self.snake:
                continue
            self.fruit = fruit
            break

    def dist(self, a, b):
        x1, y1 = a
        x2, y2 = b
        return (x1-x2)**2+(y1-y2)**2

    def step(self, action, render=False):
        done = False
        self.move_snake(action)
        if self.fruit == self.snake[0]:
            self.reward = 1.
            self.fruit_ate += 1
            self.grow()
            if self.fruit_ate == self.max_score:
                print('We have a winner!')
                self.reward = 100
                done = True
                return self.get_state(render), self.reward, done, 'info'
            self.drop_fruit()
        elif self.self_bite() or self.hit_border():
            self.reward = -10.
            done = True
        else:
            self.reward = 0.1 if self.dist(self.fruit, self.snake[0]) < self.dist(self.fruit, self.prev_head) else -0.1
        self.total_reward += self.reward
        if done:
            self.history.append(self.get_grid())
            return np.zeros(self.state_space), self.reward, done, 'info'
        return self.get_state(render), self.reward, done, 'info'

    def get_grid(self):
        grid = -np.ones((self.grid_size+2, self.grid_size+2))
        grid[1:-1, 1:-1] = 0
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                grid[y+1, x+1] = 2  # set value of head
            else:
                grid[y+1, x+1] = 1  # set value of body
        x, y = self.fruit
        grid[y+1, x+1] = 4  # set value of fruit
        return grid

    def get_state(self, render=False):
        head_x, head_y = self.snake[0]
        fruit_x, fruit_y = self.fruit
        fruit_abv_snake = (head_y > fruit_y)
        fruit_right_snake = (head_x < fruit_x)

        grid = self.get_grid()
        if render:
            self.history.append(grid)

        neg_pad = 1  # extra padding introduced by negative grid border
        head_x += neg_pad
        head_y += neg_pad
        obst_above = (abs(grid[head_y-1, head_x]) == 1)
        obst_below = (abs(grid[head_y+1, head_x]) == 1)
        obst_right = (abs(grid[head_y, head_x+1]) == 1)
        obst_left = (abs(grid[head_y, head_x-1]) == 1)

        return np.array([fruit_abv_snake, fruit_right_snake, obst_above,
                         obst_below, obst_right, obst_left])

    def render(self, path, game):
        fig = plt.figure()
        ims = []
        for hist in self.history:
            im = plt.imshow(hist.reshape(self.grid_size+2, self.grid_size+2),
                            animated=True)
            plt.axis('off')
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True,
                                        repeat_delay=1)
        ani.save(path+'/snake_' + str(game) + '_' + str(self.fruit_ate) + '.mp4')
        plt.close()
