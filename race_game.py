import random
import math
import numpy as np


class RaceGame:
    def __init__(self, track_width=800, track_height=600, num_obstacles=20):
        self.track_width = track_width
        self.track_height = track_height
        self.num_obstacles = num_obstacles
        self.reset()
        self.state_size = 6  # x, y, angle, speed, nearest obstacle distance, fuel
        self.action_size = 4  # steer left, steer right, accelerate, brake

    def reset(self):
        self.car_x = self.track_width // 2
        self.car_y = self.track_height - 50
        self.car_angle = 0
        self.car_speed = 0
        self.car_fuel = 100
        self.obstacles = [(random.randint(0, self.track_width), random.randint(0, self.track_height))
                          for _ in range(self.num_obstacles)]
        return self._get_state()

    def _get_state(self):
        nearest_obstacle_dist = min(math.sqrt((self.car_x - ox) ** 2 + (self.car_y - oy) ** 2)
                                    for ox, oy in self.obstacles)
        return np.array([self.car_x, self.car_y, self.car_angle, self.car_speed,
                         nearest_obstacle_dist, self.car_fuel])

    def step(self, action):
        if action == 0:  # steer left
            self.car_angle = (self.car_angle + 5) % 360
        elif action == 1:  # steer right
            self.car_angle = (self.car_angle - 5) % 360
        elif action == 2 and self.car_fuel > 0:  # accelerate
            self.car_speed = min(10, self.car_speed + 1)
            self.car_fuel -= 1
        elif action == 3:  # brake
            self.car_speed = max(0, self.car_speed - 1)

        self.car_x += self.car_speed * math.sin(math.radians(self.car_angle))
        self.car_y -= self.car_speed * math.cos(math.radians(self.car_angle))

        self.car_x = max(0, min(self.track_width, self.car_x))
        self.car_y = max(0, min(self.track_height, self.car_y))

        collision = any(math.sqrt((self.car_x - ox) ** 2 + (self.car_y - oy) ** 2) < 20
                        for ox, oy in self.obstacles)

        done = self.car_y <= 10 or self.car_fuel <= 0 or collision

        if self.car_y <= 10:  # Reached the top of the track
            reward = 1000 + self.car_fuel
        elif collision:
            reward = -500
        elif self.car_fuel <= 0:
            reward = -100
        else:
            reward = self.car_speed - 5

        return self._get_state(), reward, done