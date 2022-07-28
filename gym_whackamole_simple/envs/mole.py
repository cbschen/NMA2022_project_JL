from gym import spaces
import numpy as np
import pygame

class Mole(spaces.Box):
    def __init__(self, low, high, shape, window_size, params=None):
        super().__init__(low = low, high = high, shape = shape)
        self.window_size = window_size 
        self.set_task_parameters()
        self.am_I_hit = 0
        self._mole_life = 0
        self._mole_location = np.array([0,0])
        self.params = params
        self.reset()
        

    def die(self):
        self.am_I_hit = 0
        self._mole_life = 0
        self._mole_location = np.array([0,0])

    def reset(self):
        self.die()
        self.set_task_parameters()



    def set_task_parameters(self):
        if self.params == None:
            params = dict()
            params['p_popping'] = 0.5
            params['max_life'] = 20
            params['radius'] = 10
            params['reward_hit'] = 100
            params['reward_miss'] = -50
            params['version_resample'] = dict({
                "cond": "uniform",
                "value": None
            })
            params['version_needhit'] = True
    
        self.params = params
       



    def step(self, gaze, action_hit): 
        if self.am_I_hit == 1: # if I was hit in the previous frame
            self.die()
        else:
            if self._mole_life > 0: # if mole is alive
                self._mole_life -= 1
                if self._mole_life == 0:
                    self.die()
            elif np.random.random() < self.params['p_popping']:
                self.pop()

        self.am_I_hit = 0 # reset am_I_hit
        reward = 0
        if self.params['version_needhit']:
            if action_hit == 1:
                if self._mole_life > 0 and self.collide(self.obs(),gaze):
                    self.am_I_hit = 1
                    reward = self.params['reward_hit']
                else: # you actually hit and missed/no mole available
                    self.am_I_hit = -1
                    reward = self.params['reward_miss']
       
        return reward

    def collide(self, mole, gaze):
        xy_mole = mole["xy"]
        r_mole = mole["radius"]
        xy_gaze = gaze["xy"]
        r_gaze = gaze["radius"]
        dis = np.sqrt(np.sum((xy_gaze - xy_mole) ** 2))
        if dis < np.abs(r_gaze + r_mole): # as long as it touches
            return True
        else:
            return False

    def obs(self):
        is_visible = 1 if self._mole_life > 0 else 0
        return {"xy": self._mole_location, "radius": self.params['radius'], "isvisible": is_visible, "ishit": self.am_I_hit}
    
    def sample_pos(self):
        if self.params['version_resample']['cond'] == "uniform":
            t = np.random.random(size = 2) * self.window_size
        elif self.params['version_resample']['cond'] == "fixed":
            tpos = self.params['version_resample']['value']
            t = np.array(tpos) * self.window_size
        return t[0], t[1]

    def pop(self):
        self.am_I_hit = 0
        self._mole_life = self.params['max_life']
        tx, ty = self.sample_pos()
        self.set_pos(tx, ty)

   



    def set_pos(self, x, y):
        self._mole_location = np.array([x,y])

  

    def get_task_parameters(self):
        return self.params

    def _render_frame(self, canvas):
        if self._mole_life > 0:
            if self.am_I_hit == 1:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 255),
                    self._mole_location,
                    self.params["radius"]*2,
                )
            else:
                pygame.draw.circle(
                    canvas,
                    (0, 255, 0),
                    self._mole_location,
                    self.params["radius"],
                )