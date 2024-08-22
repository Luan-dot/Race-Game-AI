import tkinter as tk
import numpy as np
import math
import time
from race_game import RaceGame
from race_game_ai import RaceGameAI


class RaceGameVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Race Game AI Visualizer")
        self.canvas = tk.Canvas(master, width=800, height=600)
        self.canvas.pack()

        self.game = RaceGame(track_width=800, track_height=600)
        self.ai = RaceGameAI(self.game.state_size, self.game.action_size)

        self.car = self.canvas.create_polygon(0, 0, 0, 0, 0, 0, fill='red', outline='black')
        self.obstacles = []
        self.fuel_bar = self.canvas.create_rectangle(700, 20, 780, 40, fill='green')
        self.episode_label = self.canvas.create_text(400, 20, text="Episode: 0")
        self.reward_label = self.canvas.create_text(400, 40, text="Total Reward: 0")

        self.total_reward = 0
        self.episode = 0

        self.master.after(0, self.train_episode)

    def update_car_polygon(self):
        car_points = [(-10, -20), (10, -20), (10, 20), (-10, 20)]
        rotated_points = []
        for x, y in car_points:
            rx = x * math.cos(math.radians(self.game.car_angle)) - y * math.sin(math.radians(self.game.car_angle))
            ry = x * math.sin(math.radians(self.game.car_angle)) + y * math.cos(math.radians(self.game.car_angle))
            rotated_points.extend([rx + self.game.car_x, ry + self.game.car_y])
        self.canvas.coords(self.car, *rotated_points)

    def train_episode(self):
        state = self.game.reset()
        self.total_reward = 0
        self.episode += 1

        self.canvas.delete('obstacle')
        self.obstacles = [self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill='gray', tags='obstacle')
                          for x, y in self.game.obstacles]

        done = False
        while not done:
            state = np.reshape(state, [1, self.game.state_size])
            action = self.ai.act(state)
            next_state, reward, done = self.game.step(action)
            next_state = np.reshape(next_state, [1, self.game.state_size])
            self.ai.remember(state, action, reward, next_state, done)
            state = next_state
            self.total_reward += reward

            self.update_car_polygon()
            self.canvas.coords(self.fuel_bar, 700, 20, 700 + self.game.car_fuel * 0.8, 40)
            self.canvas.itemconfig(self.reward_label, text=f"Total Reward: {self.total_reward:.2f}")
            self.canvas.itemconfig(self.episode_label, text=f"Episode: {self.episode}")

            self.master.update()
            time.sleep(0.05)

        if len(self.ai.memory) > 32:
            self.ai.replay(32)

        if self.episode % 10 == 0:
            self.ai.model.save(f'race_game_model_{self.episode}.h5')

        self.master.after(100, self.train_episode)


if __name__ == "__main__":
    root = tk.Tk()
    visualizer = RaceGameVisualizer(root)
    root.mainloop()
