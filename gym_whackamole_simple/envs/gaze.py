from gym import spaces
import math
import numpy as np
import pygame

class Gaze(spaces.Box):
    def __init__(self, low, high, shape, window_size, params = None):
        super().__init__(low = low, high = high, shape = shape)
        self.window_size = window_size
        self._gaze_location = None
        self.speed = None
        self.radius = None
        self.reset()


    def set_pos(self, x, y):
        x = float(x)
        y = float(y)
        self._gaze_location = np.array([x,y])

    def reset(self):
        tx, ty = [self.window_size/2, self.window_size/2]
        self.set_pos(tx, ty)
        self.phi = 0
        self.speed = 10
        self.radius = 50
         

   

   

    

    def step(self, angle): # action is just the phi angle if within 0-359, it is hit if it is 360
        if angle == 360:

            return
        # move gaze
        self.phi = angle
        self._gaze_location[0] += self.speed * np.cos(angle)  
        self._gaze_location[1] += self.speed * np.sin(angle)  
        if self._gaze_location[0] > self.window_size:
            self._gaze_location[0] = self.window_size
        elif self._gaze_location[0] < 0:
            self._gaze_location[0] = 0

        if self._gaze_location[1] > self.window_size:
            self._gaze_location[1] = self.window_size
        elif self._gaze_location[1] < 0:
            self._gaze_location[1] = 0


    

        

    def is_valid_xy(self, x, y):
        if x > 0 and x < self.window_size[0] and y > 0 and y < self.window_size[1]:
            return True
        else:
            return False

   

    

        
       

    def obs(self):
        return {"xy":self._gaze_location, "phi": self.phi, "radius": self.params['radius'], 
                "v_step": self.speed}

    def get_task_parameters(self):
        return self.params

    def get_xy_front(self):
        x, y = self._gaze_location
        x = np.cos(self.phi) * self.params['radius'] + x
        y = np.sin(self.phi) * self.params['radius'] + y
        return np.append(x, y)

    def _render_frame(self, canvas, ishit = 0, width_line = 1):
        if ishit == -1:
            width_gaze = 0
        elif ishit == 1:
            width_gaze = 5
        else:
            width_gaze = 1
        col_gaze = (255, 0, 0)
        pygame.draw.circle(
                canvas,
                col_gaze,
                self._gaze_location,
                self.raidus,
                width = width_gaze
            )
        pygame.draw.line(
                canvas, 
                col_gaze, 
                self._gaze_location, 
                self.get_xy_front(), 
                width = width_line
            )

        