import random
import tkinter
import time

UNITS = 45

from PIL import Image, ImageTk

class Road(): 
    def __init__(self, n_obst=5, use_tkinter=True):
        self.use_tkinter = use_tkinter
        self.road_width = 40
        self.road_height = 2
        self.friction = 0.01

        self.goal_velocity = 5
        self.n_obst = n_obst
        
        self.actions = {
            'no_change': 0,
            'speed_up': 1,
            'speed_up_2x': 3,
            'slow_down': -1,
            'slow_down_2x': -3
        }

        self.state_features = [
            'self_position', 
            'self_velocity',
            
        ]
        for i in range(self.n_obst):
            self.state_features.append(f'obst_position_{i}')

        self.reward = 0

        self.action_reward = {
            'good_velocity': 50,
            'under_speed': -30,
            'step_cost': -15,
            'over_speed': -5,
            'over_speed_near_ped': -50,
            'negetive_speed': -100,
            'action_change': -10,
            'too_late': -100
        }

        self.max_velocity = 9
        self.min_veloctiy = 1
        self.max_ped_velocity = 2
        self.max_time = 50

        self.reset()

        if use_tkinter:
            self.root = tkinter.Tk('Simulation')
            self.root.configure(bg='gray15')
            self.root.title("Simulation")
            
            self.obs_img = ImageTk.PhotoImage(Image.open('obst.png').resize((UNITS, UNITS), Image.BICUBIC))
            self.car_img = ImageTk.PhotoImage(Image.open('car.png').resize((UNITS, UNITS), Image.BICUBIC))

            self.root.geometry(f'{self.road_width * UNITS}x{self.road_height * UNITS}')
            self.canvas = tkinter.Canvas(self.root, bg="gray15", height=self.road_height * UNITS, width=self.road_width * UNITS,  highlightthickness=0)
            self.canvas.pack()

            
            self.canvas.create_line(0, self.road_height * UNITS / 5, self.road_width * UNITS, self.road_height * UNITS / 5, fill='gray90', width=0.1 * UNITS, dash=(100, 5))
            self.canvas.create_line(0, self.road_height * UNITS / 5 * 4, self.road_width * UNITS, self.road_height * UNITS / 5 * 4, fill='gray90', width=0.1 * UNITS, dash=(100, 5))
            #self.self_id = self.canvas.create_rectangle(self.self_pos * UNITS, 0 * UNITS, (self.self_pos + 1) * UNITS, (0 + 1) * UNITS, fill='white', outline='')
            self.self_id = self.canvas.create_image(i * UNITS, 0.5 * UNITS, image=self.car_img)
            self.canvas.create_oval((self.road_width - 7) * UNITS + 0.1 * UNITS, 0 + 0.1 * UNITS, (self.road_width - 7) * UNITS + UNITS - 0.1 * UNITS, UNITS - 0.1 * UNITS, fill='yellow', outline='goldenrod', width=0.05 * UNITS)
            self.canvas.create_line(0, self.road_height * UNITS / 2, self.road_width * UNITS, self.road_height * UNITS / 2, fill='gray90', width=0.3 * UNITS, stipple='gray75')

            self.obst_ids = []
            for i in self.obst_pos:
                #self.obst_ids.append(self.canvas.create_rectangle(i * UNITS, 1 * UNITS, (i + 1) * UNITS, (1 + 1) * UNITS, fill='red', outline='black'))
                self.obst_ids.append(self.canvas.create_image(i * UNITS, 1.5 * UNITS, image=self.obs_img))

    def reset(self):
        random_obst = [random.randint(1, self.road_width) for _ in range(self.n_obst)]

        self.obst_pos = random_obst
        self.goal_pos = self.road_width
        self.self_pos = 0
        self.self_vel = 0
        self.self_previous_pos = 0
        self.self_previous_vel = 0
        self.time = 0

        state_to_return = self.process_state(
            [self.self_pos, self.self_vel, *self.obst_pos]
        )

        return state_to_return

    def transition(self, action):
        if action == 'no_change':
            self.self_vel *= self.friction
        return max(self.self_vel + self.actions[action], 0)

    def step(self, action):
        self.self_previous_pos = self.self_pos
        self.self_previous_vel = self.self_vel
        self.previous_action = action

        self.self_vel = self.transition(action)
        self.self_pos += self.self_vel # Assume t=1, then => d_x = v

        reward, finished = self.reward_fn(action)

        state_to_return = self.process_state(
            [self.self_pos, self.self_vel, *self.obst_pos]
        )

        self.time += 1        
        
        if self.use_tkinter:
            self.canvas.delete(self.self_id)
            #self.self_id = self.canvas.create_rectangle(self.self_pos * UNITS, 0 * UNITS, (self.self_pos + 1) * UNITS, (0 + 1) * UNITS, fill='white', outline='')
            
            self.self_id = self.canvas.create_image(self.self_pos * UNITS, 0.5 * UNITS, image=self.car_img)

            for i in self.obst_ids:
                self.canvas.delete(i)

            self.obst_ids = []
            for i in self.obst_pos:
                #self.obst_ids.append(self.canvas.create_rectangle(i * UNITS, 1 * UNITS, (i + 1) * UNITS, (1 + 1) * UNITS, fill='red', outline='black'))
                self.obst_ids.append(self.canvas.create_image(i * UNITS, 1.5 * UNITS, image=self.obs_img))

            self.root.update()

        if finished:
            self.reset()

        #time.sleep(0.1)

        return state_to_return, reward, finished

    def process_state(self, state):
        return state

    def reward_fn(self, action):
        self.reward = 0

        finished = False

        if self.self_vel != self.self_previous_vel:
            d_v = self.self_vel - self.self_previous_vel
            self.reward += self.action_reward['action_change'] * abs(d_v)

        if self.self_pos > self.goal_pos:
            self.self_pos = self.goal_pos

            if self.self_vel >= self.goal_velocity:
                self.reward += self.action_reward['good_velocity']
            else:
                self.reward += self.action_reward['under_speed']

            finished = True
        else:
            self.reward += self.action_reward['step_cost']
            finished = False

        if self.self_vel > self.max_velocity:
            excess_in_velocity = self.self_vel - self.max_velocity
            self.reward += self.action_reward['over_speed'] * excess_in_velocity

        if self.self_vel < self.min_veloctiy: 
            excess_in_velocity = self.min_veloctiy - self.self_vel
            self.reward += self.action_reward['under_speed'] * excess_in_velocity

            if self.self_vel < 0:
                excess_in_velocity = abs(self.min_veloctiy)
                self.self_vel = 0
                self.reward += self.action_reward['negative_speed'] * excess_in_velocity

        for j in self.obst_pos:
            if self.self_previous_pos <= j <= self.self_pos:
                if self.self_vel > self.max_ped_velocity:
                    excess_in_velocity = self.self_vel - self.max_ped_velocity
                    self.reward += self.action_reward['over_speed_near_ped'] * excess_in_velocity
        
        if self.time > self.max_time:
            finished = True
            self.reward += self.action_reward['too_late']


        return self.reward, finished
