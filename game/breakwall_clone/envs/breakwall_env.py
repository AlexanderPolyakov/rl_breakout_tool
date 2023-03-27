import gymnasium as gym
from gymnasium import spaces, utils
from gymnasium.utils import seeding

import random
from random import randint
import numpy as np
from numpy import linalg as LA

def check_line_coll(pos, rad, norm, d):
  diff = np.dot(pos, norm) - rad
  if diff <= d:
    return (np.array(norm), diff - d)
  return None

def check_box_coll(pos, rad, lt, rb):
  if pos[0] < lt[0] - rad or pos[0] > rb[0] + rad or pos[1] < lt[1] - rad or pos[1] > rb[1] + rad:
    return None
  cpos = [max(min(pos[0], rb[0]), lt[0]), max(min(pos[1], rb[1]), lt[1])]
  diff = pos - cpos
  dist = LA.norm(pos - cpos)
  if dist <= rad:
    if dist == 0.0:
      diff = pos - (lt + rb) * 0.5
      return (diff * (1.0 / LA.norm(diff)), -rad)
    return (diff * (1.0 / dist), dist - rad)
  return None

def check_ext_coll(pos, rad, size):
  lt = check_line_coll(pos, rad, [1, 0], 0.0)
  if lt is not None:
    return lt
  rt = check_line_coll(pos, rad, [-1, 0], -size[0])
  if rt is not None:
    return rt
  tp = check_line_coll(pos, rad, [0, 1], 0.0)
  if tp is not None:
    return tp
  bt = check_line_coll(pos, rad, [0, -1], -size[1])
  return bt

class Ball():
  def __init__(self, pos, vel_inc, max_spd):
    # Init ball variables
    self.pos = pos
    self.radius = 4
    self.velocity = np.array([random.choice([-0.5, 0.5]), -0.5])
    self.maxSpeed = max_spd
    self.velocityInc = vel_inc

  def checkWalls(self, screen_size):
    coll = check_ext_coll(self.pos, self.radius, screen_size)
    if coll is not None:
      norm, d = coll
      if norm[1] == -1.0:
        return True
      self.velocity = self.velocity - norm * (2.0 * np.dot(self.velocity, norm))
      self.pos = self.pos - norm * d
      self.bumpVel()

  def checkVel(self):
    if self.velocity[1] < 0.1 and self.velocity[1] > -0.1:
      spd = LA.norm(self.velocity)
      self.velocity[1] = 0.1 if self.velocity[1] >= 0.0 else -0.1
      self.velocity = self.velocity * (spd / LA.norm(self.velocity))

  def bumpVel(self):
    self.velocity = self.velocity * self.velocityInc
    curSpd = LA.norm(self.velocity)
    if curSpd > self.maxSpeed:
      self.velocity = self.velocity * (self.maxSpeed / curSpd)

class Brick():
  def __init__(self, posx, posy, width, height, color):
    # Init brick variables
    self.size = np.array([width, height])
    self.pos = np.array([posx, posy])
    self.color = color

class Paddle():
  def __init__(self, screen_width, screen_height, width, height):
    # Init paddle variables
    self.size = np.array([width, height])
    self.screen_width = screen_width
    offs = screen_height / 10
    self.pos = np.array([(screen_width - width) / 2, screen_height - height - offs])

  # Move paddle left
  def moveLeft(self, speed):
    self.pos[0] = max(self.pos[0] - speed, 0)

  # Move paddle right
  def moveRight(self, speed):
    self.pos[0] = min(self.pos[0] + speed, self.screen_width - self.size[0])

# Utility function to initialise the bricks wall
def initBricks(rows=8, columns=14, col_width=15, offset_row = 10, brick_color = [66, 72, 200]):
  # Grid custom layout
  offset_x = col_width
  offset_y = 7

  bricks = []
  width = col_width
  height = offset_y
  # Create grid of bricks
  for row in range(rows):
    for col in range(columns):
      # Create brick and position
      x = offset_x * col
      y = offset_row + (offset_y * row) + height 
      brick = Brick(x, y, width, height, brick_color)
      bricks.append(brick)
  return bricks

def draw_rect(buffer, pos, size, rgb):
  y0 = max(int(pos[1]), 0)
  y1 = min(int(pos[1] + size[1]), buffer.shape[0])
  x0 = max(int(pos[0]), 0)
  x1 = min(int(pos[0] + size[0]), buffer.shape[1])
  buffer[y0:y1, x0:x1] = rgb

def draw_circle(buffer, pos, rad, rgb):
  mdx = -min(int(pos[0] - rad), 0)
  mdy = -min(int(pos[1] - rad), 0)
  mmdx = -max(int(pos[0] + rad + 1 - buffer.shape[1]), 0)
  mmdy = -max(int(pos[1] + rad + 1 - buffer.shape[0]), 0)
  for y in range(-rad + mdy, +rad + 1 + mmdy):
    for x in range(-rad + mdx, +rad + 1 + mmdx):
      dist = x ** 2 + y ** 2
      if dist > rad ** 2:
        continue
      buffer[int(pos[1] + y)][int(pos[0] + x)] = rgb


class BreakWallGame():
  def __init__(self, width, height, vel_inc, max_spd, paddle_spd, lives_num, offset_row, brick_color):
    # Initialise game environment variables
    self.screen_width = width
    self.screen_height = height

    self.buffer = np.zeros((height, width, 3), dtype=np.uint8)
    self.velInc = vel_inc
    self.maxSpd = max_spd
    self.paddleSpd = paddle_spd
    self.livesNum = lives_num

    self.bricksOffsetRow = offset_row
    self.brickColor = brick_color

    self.reset()
    
  def reset(self):
    self.initial_score = 0
    self.game_over = False
    self.lives = self.livesNum
    
    # Initialise paddle and ball
    self.paddle = Paddle(self.screen_width, self.screen_height,
                         self.screen_width / 5, self.screen_height / 10)
    self.ball = Ball(self.paddle.pos + self.paddle.size * [0.5, -1.0], self.velInc, self.maxSpd)

    # Initialise wall of bricks
    # Display them in a grid
    self.bricks = initBricks(offset_row = self.bricksOffsetRow, brick_color = self.brickColor)

  def update(self, dt):
    # update ball
    self.ball.pos = self.ball.pos + self.ball.velocity * dt

    # Check if the ball is bouncing against any of the 3 walls (left, right, top)
    res = self.ball.checkWalls([self.screen_width - 1.0, self.screen_height - 1.0])

    # Check collision with bottom wall (ground)
    if res == True:
      # fall through
      self.lives -= 1
      if self.lives <= 0:
        self.game_over = True

    # Check ball-paddle collision
    coll = check_box_coll(self.ball.pos, self.ball.radius, self.paddle.pos, self.paddle.pos + self.paddle.size)
    if coll is not None:
      norm, d = coll
      # redo the norm as an elleptical pad
      if norm[1] == -1.0:
        hdiff = self.ball.pos[0] - (self.paddle.pos[0] + self.paddle.size[0] * 0.5)
        normMod = hdiff / self.paddle.size[0]
        norm = np.array([normMod, -1.0])
        norm = norm / LA.norm(norm)
      veldot = np.dot(self.ball.velocity, norm)
      if veldot < 0.0:
        self.ball.velocity = self.ball.velocity - norm * (2.0 * veldot)
        self.ball.bumpVel()
      else:
        self.ball.pos -= norm * d

    # Check ball-bricks collision
    while True:
      collision = False
      for brick in self.bricks:
        coll = check_box_coll(self.ball.pos, self.ball.radius, brick.pos, brick.pos + brick.size)
        if coll is None:
          continue
        norm, d = coll
        if np.dot(self.ball.velocity, norm) < 0.0:
          self.ball.velocity = self.ball.velocity - norm * (2.0 * np.dot(self.ball.velocity, norm))
          self.ball.bumpVel()
          self.initial_score += 1
        self.bricks.remove(brick)
        collision = True
        break
      if not collision:
        break

    # check if ball speed is correct after all collisions
    self.ball.checkVel()
 
    # Check win status
    if len(self.bricks) == 0:
      self.game_over = True

  def step(self, action):
    assert action in ["NOOP", "FIRE", "LEFT", "RIGHT"], "Invalid action" + action
    if not self.game_over:
      if action == "LEFT":
        self.paddle.moveLeft(self.paddleSpd)
      if action == "RIGHT":
        self.paddle.moveRight(self.paddleSpd)

      spd = LA.norm(self.ball.velocity)
      if spd >= self.ball.maxSpeed * 0.5:
        for _ in range(2):
          self.update(0.5)
      else:
        self.update(1.0)

    # faster way to make black background
    self.buffer = self.buffer * 0

    # Draw all the sprites in the game
    for brick in self.bricks:
      draw_rect(self.buffer, brick.pos, brick.size, brick.color)

    # Update full display screen
    draw_rect(self.buffer, self.paddle.pos, self.paddle.size, [200, 72, 72])

    draw_circle(self.buffer, self.ball.pos, self.ball.radius, [255, 255, 255])

  def getScreenRGB(self):
    return self.buffer


class BreakWall(gym.Env):
  """
  This is a simple, high level wrapper class for the breakwall game which presents it with an openai gym compatible interface as per:
  https://github.com/openai/gym/blob/master/docs/creating_environments.md
  """
  metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

  def __init__(self, render_mode = None, vel_inc = 1.005, max_spd = 3.0, paddle_spd = 2.5, lives_num = 1, offset_row = 10,
               brick_color = [66, 72, 200]):
    self.render_mode = render_mode
    self.game = BreakWallGame(210, 160, vel_inc, max_spd, paddle_spd, lives_num, offset_row, brick_color)
    self.screen = None
    # these variables allow atari wrapper to work
    self._action_set = [0,1,3,4] # NOOP, FIRE, LEFT, RIGHT
    self.action_space = spaces.Discrete(len(self._action_set))
    self.screen_width = self.game.screen_width
    self.screen_height = self.game.screen_height
    self.observation_space = spaces.Box(low=0, high=255, shape=(self.game.screen_height, self.game.screen_width, 3), dtype=np.uint8)
    self.seed()

    # need this to provide a self.ale object with a lives function
    class ale:
      def __init__(self, game):
        self.game = game 
      def lives(self):
        return self.game.lives
    self.ale = ale(self.game)

    self.ACTION_MEANING = {
      0: "NOOP",
      1: "FIRE",
      2: "UP",
      3: "RIGHT",
      4: "LEFT",
      5: "DOWN",
      6: "UPRIGHT",
      7: "UPLEFT",
      8: "DOWNRIGHT",
      9: "DOWNLEFT",
      10: "UPFIRE",
      11: "RIGHTFIRE",
      12: "LEFTFIRE",
      13: "DOWNFIRE",
      14: "UPRIGHTFIRE",
      15: "UPLEFTFIRE",
      16: "DOWNRIGHTFIRE",
      17: "DOWNLEFTFIRE",
    }
  
  def step(self, action):
    """
    """
    start_score = self.game.initial_score
    action = self._action_set[action]
    self.game.step(self.ACTION_MEANING[action])
    state = self.game.getScreenRGB()
    end_score = self.game.initial_score
    reward = end_score - start_score
    terminal = self.game.game_over
    if self.render_mode == "human":
        self.render(self.render_mode)
    return state, reward, terminal, False, {}


  def seed(self, seed=None):
    self.np_random, seed1 = seeding.np_random(seed)

  def reset(self, *, seed = None, options = None):
    super().reset(seed=seed)
    self.game.reset()
    state = self.game.getScreenRGB()
    return state, {}
    
  def render(self):
    """
    show the current screen
    """
    if self.render_mode == 'rgb_array':
        img = self.game.getScreenRGB()
        return img
    elif self.render_mode == 'human':
        import pygame
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.game.screen_width,self.game.screen_height))
            self.clock = pygame.time.Clock()
            
            pygame.display.set_caption("Breakwall")

        pygame.surfarray.blit_array(self.screen, self.game.getScreenRGB())
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

  def close(self):
    pass
  
  def get_action_meanings(self):
    """
    copied from atari_env.py
    """
    return [self.ACTION_MEANING[i] for i in self._action_set]

