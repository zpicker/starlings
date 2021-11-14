"""
Starlings.py
"""

import numpy as np
import cv2
from scipy.spatial import KDTree
import time
import pyaudio
import pyautogui
import win32api,win32process,win32con



class Viz:
    def __init__(self):
        #windows shit
        self.ES_CONTINUOUS = 0x80000000
        self.ES_SYSTEM_REQUIRED = 0x00000001
        pyautogui.FAILSAFE = False
        
        #frame init
        self.frame_xres = 600
        self.frame_yres = 300
        self.frame = np.ones((self.frame_yres,self.frame_xres,3))
        self.open = 0
        self.time = time.time()
        self.num_frames = 0
        self.center_xres = 400
        self.center_yres = 200
        
        #starling params      
        self.timestep = 0.1
        self.reaction_time = 0.5
        self.cruise_speed = 30
        self.mass = 80
        self.lift_drag = 3.3
        self.default_lift,self.default_thrust,self.banking_in,self.banking_out = 0.78,0.24,10,1
        self.speed_control = 1
        self.max_sep_rad = 2
        self.grav = 9.8
        self.relax_time = .01
        
        #starling weights
        self.sep_weight = 200
        self.coh_weight = 100
        self.align_weight = 1500
        self.random_weight = 100
        self.roost_h_weight = 10
        self.roost_v_weight = 100
        self.lift_weight = 1.5
        self.drag_weight = 1
        self.thrust_weight = 1
        
        #flock params
        self.num_birds = 750
        self.bird_pos = ((np.random.rand(2,self.num_birds)*0.2)+0.2) * [[self.frame_yres],[self.frame_xres]]
        self.bird_pos_round = np.floor(self.bird_pos).astype(int)
        self.bird_vel = np.ones((2,self.num_birds))*self.cruise_speed + (np.random.rand(2,self.num_birds)-1/2)*self.cruise_speed
        self.num_neighb = 5
        self.sep_sigma = 5
        self.num_check_force = int(.3*self.num_birds)
        self.num_check_neighb = int(.1*self.num_check_force)
        self.which_birds = np.random.choice(self.num_birds,self.num_check_force,replace=False)
        
        #starling object init
        self.F_steer = np.zeros((2,self.num_birds))
        self.F_flight = np.zeros((2,self.num_birds))
        self.vel = (self.bird_vel[0,:]**2 + self.bird_vel[1,:]**2)**(0.5)
        self.bird_dir = self.bird_vel/self.vel
        self.center = np.average(self.bird_pos,axis=1)
        self.center_dist = np.linalg.norm(self.center - np.transpose(self.bird_pos),axis=1)
        self.center_dist = self.center_dist/np.sum(self.center_dist)
        self.roost_loc = np.array([self.frame_yres/2,self.frame_xres/2])
        self.roost_vel = (np.random.rand(2)-.5)*0.5
        
        #start neighborhood
        self.tree = KDTree(np.transpose(self.bird_pos))
        self.neighborhood=np.zeros((self.num_birds,self.num_neighb))
        self.neighborhood_dists=np.zeros((self.num_birds,self.num_neighb))
        for i in range(self.num_birds):
            self.neighborhood[i,:] = self.tree.query(self.bird_pos[:,i],k=self.num_neighb,p=2)[1]
            self.neighborhood_dists[i,:] = self.tree.query(self.bird_pos[:,i],k=self.num_neighb,p=2)[0]
    
        #color objects
        self.order = np.random.choice([0,1,2,3,4],3,replace=False)
        
        #music init
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100 
        self.volume = 1
        p=pyaudio.PyAudio()

        self.stream = p.open(
            format=self.FORMAT, 
            input_device_index=0,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK)
        self.music_norm = 400
        self.get_music_data()
        self.mean_v = 0
        self.data_scale = 10
        self.bass = 0
        self.beat = 0
        self.no_bass = 0
        self.beat_bass_num = 3
        self.drop_bass_num = 10

        self.drop_v = 100
        self.drop = 0
        self.no_bass_timer = time.time()
        
        #timing params
        self.total_time = 4*60 #mins
        self.total_time = self.total_time*60 #s
        self.time_in = 2*60
        self.starttime = time.time()-(self.time_in*60)
        self.color_time = 0.3*self.total_time
        self.sample_time_s = 0.15*self.total_time
        self.sample_time_f = self.total_time
        self.drop_time_s = 0.1*self.total_time
        self.drop_time_f = self.total_time
        self.bar_time_s = 0.5*self.total_time
        self.tracer_time_s = 0.5*self.total_time
        self.shift_time_s = 0.4*self.total_time
        self.kal_time_s = 0.6*self.total_time
        self.bars_on,self.bar_start,self.bar_length = 0,0,0
        self.tracers_on,self.tracers_length,self.tracers_start=0,0,0
        self.shift_on,self.shift_length,self.shift_start = 0,0,0
        self.sample_on,self.sample_length,self.sample_start=0,0,0
        self.kal_on,self.kal_length,self.kal_start,self.kal_num_flips,self.kal_which_flips=0,0,0,0,1
    
    def frame_adv(self):
        
        self.color_fact = 0.7 * np.min([(time.time()-self.starttime)/self.color_time,1])
        self.sample_fact = 0.3* (self.mean_v/self.data_scale)* np.max([np.min([(time.time()-self.starttime-self.sample_time_s)/(self.sample_time_f-self.sample_time_s),1]),0])
        self.drop_fact = np.max([np.min([(time.time()-self.starttime-self.drop_time_s)/(self.drop_time_f-self.drop_time_s),1]),0])
        self.bar_fact = 0.1 *(self.mean_v/self.data_scale)* np.max([np.min([(time.time()-self.starttime-self.bar_time_s)/(self.total_time-self.bar_time_s),1]),0])
        self.tracer_fact = 0.1 *(self.mean_v/self.data_scale)* np.max([np.min([(time.time()-self.starttime-self.tracer_time_s)/(self.total_time-self.tracer_time_s),1]),0])
        self.shift_fact = 0.1 *(self.mean_v/self.data_scale)* np.max([np.min([(time.time()-self.starttime-self.shift_time_s)/(self.total_time-self.shift_time_s),1]),0])
        self.kal_fact = 0.1 *(self.mean_v/self.data_scale)* np.max([np.min([(time.time()-self.starttime-self.kal_time_s)/(self.total_time-self.kal_time_s),1]),0])

        try:
            self.get_music_data()
            self.bird_music_beat()
            self.bird_music_drop()
            self.bird_music_sample()
        except Exception as e:
            print('music broke:\n'+str(e))
                
        try:
            self.draw_birds_simple()
        except Exception as e:
            print('draw_birds_simple broke:\n'+str(e))
            
        try:
            self.frame_shift()
        except Exception as e:
            print('shift broke:\n'+str(e))
            
        try:
            self.frame_bars()
        except Exception as e:
            print('bars broke:\n'+str(e))
    
        try:
            self.frame_kal()
        except Exception as e:
            print('kal broke:\n'+str(e))         
            
        return self.frame
                   
    def draw_viz(self):
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)                             
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)  

        while True:
            try:    
                self.num_frames += 1
                self.time = time.time()
                cv2.imshow('window',self.frame_adv())      
                keypress = cv2.waitKey(1) & 0xFF
                if keypress == 27: # esc: quit visualizer
                    cv2.destroyAllWindows()
                    self.open = 1
                    break
                elif keypress == 91: # [: volume down
                    self.volume = self.volume*0.9
                elif keypress == 93: # ]: volume up
                    self.volume = self.volume*1.1
                elif keypress == 32:
                    self.roost_loc = np.random.randint(self.frame_yres/3,self.frame_yres*2/3,2)
                if np.random.rand()<0.01:
                    pyautogui.moveTo(0,1,duration=0,_pause=False)
                if np.random.rand()>0.99:
                    pyautogui.moveTo(1,0,duration=0,_pause=False)
                    pyautogui.click(_pause=False)
            except Exception as e:
                print('draw_viz broke:\n'+str(e))
                cv2.destroyAllWindows()
                self.open = 1
                break
        return
    
    def draw_birds_simple(self):

        self.old_bird_pos = self.bird_pos
        self.update_birds()

        self.bird_draw_y = self.bird_pos_round[0,:][(self.bird_pos_round[0,:]>=0)&(self.bird_pos_round[0,:]<self.frame_yres)&(self.bird_pos_round[1,:]>=0)&(self.bird_pos_round[1,:]<self.frame_xres)]
        self.bird_draw_x = self.bird_pos_round[1,:][(self.bird_pos_round[0,:]>=0)&(self.bird_pos_round[0,:]<self.frame_yres)&(self.bird_pos_round[1,:]>=0)&(self.bird_pos_round[1,:]<self.frame_xres)]
        self.bird_draw_dir_y = self.bird_dir[0,:][(self.bird_pos_round[0,:]>=0)&(self.bird_pos_round[0,:]<self.frame_yres)&(self.bird_pos_round[1,:]>=0)&(self.bird_pos_round[1,:]<self.frame_xres)]
        self.bird_draw_dir_x = self.bird_dir[1,:][(self.bird_pos_round[0,:]>=0)&(self.bird_pos_round[0,:]<self.frame_yres)&(self.bird_pos_round[1,:]>=0)&(self.bird_pos_round[1,:]<self.frame_xres)]       
        
        
        self.frame_birds[self.bird_draw_y,self.bird_draw_x,:] = [0,0,0]
#        self.frame_birds[int(self.roost_loc[0]),int(self.roost_loc[1]),:]=[1,0,0]
        
        self.bird_color_simple()
        
        self.frame = 1- self.frame_birds
        return
    
    def bird_color_simple(self):
        
        r1 = np.abs(self.bird_draw_dir_y)*self.color_fact
        r2 = np.abs(self.bird_draw_dir_x)*self.color_fact
        r3 = (np.sin(time.time()-self.starttime)+1)/2 * np.ones(len(r1)) *self.color_fact
        r4 = (1-np.abs(self.bird_draw_dir_y))*self.color_fact
        r5 = (1-np.abs(self.bird_draw_dir_x))*self.color_fact
        r6 = (np.cos(time.time()-self.starttime)+1)/2 * np.ones(len(r1)) *self.color_fact

        self.rlist = [r1,r2,r3,r4,r5,r6]
        if np.random.rand()<0.002:
            self.order = np.random.choice([0,1,2,3,4,5],3,replace=True)
#          
        self.frame_birds[self.bird_draw_y,self.bird_draw_x,:] = np.transpose(np.vstack([np.vstack([self.rlist[int(self.order[0])],self.rlist[int(self.order[1])]]),self.rlist[int(self.order[2])]]))

#        try:
#            self.frame_tracers()
#        except Exception as e:
#            print('tracers broke:\n'+str(e))
            
        return
    
    def draw_birds_center(self):

        frame_small = self.frame_birds[np.max([int(self.center[0]-(self.center_yres/2)),0]):np.min([int(self.center[0]+(self.center_yres/2)),self.frame_yres]),
                                       np.max([int(self.center[1]-(self.center_xres/2)),0]):np.min([int(self.center[1]+(self.center_xres/2)),self.frame_xres]),:]
 
        self.frame = 1- frame_small
        
        
    def update_birds(self):
        self.gen_vel_objects()
        
        self.update_F_steer()
        self.update_F_flight()
        self.update_neighb_inside()


        self.bird_vel = np.clip(self.bird_vel + (1/self.mass)*self.timestep*(self.F_steer+self.F_flight),-3*self.cruise_speed,3*self.cruise_speed)
        self.bird_pos = self.bird_pos + self.timestep*self.bird_vel
        
        self.roost_vel = self.roost_vel + np.clip((np.random.rand(2)-.5)*.075,-self.cruise_speed*.8,self.cruise_speed*.8)
        self.roost_loc = np.mod(self.roost_loc + self.roost_vel,np.array([self.frame_yres,self.frame_xres]))
        if self.roost_loc[0]<self.frame_yres/3 or self.roost_loc[0]>self.frame_yres*2/3:
            self.roost_vel[0] = -self.roost_vel[0] *.98
            self.roost_vel[1] = (np.random.rand()-.5) * 0.5
        if self.roost_loc[1]<self.frame_xres/3 or self.roost_loc[1]>self.frame_xres*2/3:
            self.roost_vel[1] = -self.roost_vel[1] * .98
            self.roost_vel[0] = (np.random.rand()-.5) * 0.5
            
        self.bird_pos_round = np.floor(self.bird_pos).astype(int)
        
        self.frame_birds = np.ones((self.frame_yres,self.frame_xres,3))
    
        return
        
    def update_F_steer(self):

        f_sep = np.zeros((2,self.num_birds))
        f_coh = np.zeros((2,self.num_birds))
        f_al = np.zeros((2,self.num_birds))
        f_rst = np.zeros((2,self.num_birds))

        self.which_birds = np.random.choice(self.num_birds,self.num_check_force,replace=False,p=self.center_dist)
        for i in self.which_birds:
            #separation
            seps = np.exp(-((self.neighborhood_dists[i,:]-self.max_sep_rad)**2)/self.sep_sigma) \
            * (self.bird_pos[:,self.neighborhood[i,:].astype(int)]-np.array([[self.bird_pos[0,i]],[self.bird_pos[1,i]]]))
            f_sep[:,i] = -(self.sep_weight/self.num_neighb) * np.sum(seps,axis=1)
            #cohesion
            cfact = np.linalg.norm((self.center-np.array([[self.bird_pos[0,i]],[self.bird_pos[1,i]]])))/(self.frame_yres/4)
            cohs = (self.bird_pos[:,self.neighborhood[i,:].astype(int)]-np.array([[self.bird_pos[0,i]],[self.bird_pos[1,i]]]))
            f_coh[:,i] = (cfact * self.coh_weight/self.num_neighb) * np.sum(cohs,axis=1)
            #alignment
            als = (self.bird_dir[:,self.neighborhood[i,:].astype(int)]-np.array([[self.bird_dir[0,i]],[self.bird_pos[1,i]]]))
            f_al[:,i] = (self.sep_weight)*np.sum(als,axis=1)/np.linalg.norm(np.sum(als,axis=1)) #not working atm
            #roost
            rst = (self.roost_loc-self.bird_pos[:,i])
            f_rst[:,i] = rst*np.array([self.roost_v_weight,self.roost_h_weight])
                        
        self.F_steer = f_sep + f_coh + f_al + f_rst
        return
    
    def update_F_flight(self):
        #speed control
        f_tau = (self.mass/self.reaction_time)*(self.cruise_speed-self.vel)*self.bird_dir
        #lift force
        lift = ((np.linalg.norm(self.bird_vel,axis=0)**2)/(self.cruise_speed**2)) * self.mass*self.grav
        f_lift = self.lift_weight * lift * np.array([-self.bird_dir[1,:]*np.sign(self.bird_dir[1,:]),self.bird_dir[0,:]*np.sign(self.bird_dir[1,:])])
        #drag force
        drag = -self.lift_drag * lift
        f_drag = self.drag_weight * drag * self.bird_dir
        #thrust force
        f_thrust  = self.thrust_weight * self.default_thrust * self.bird_dir
        #gravity
        f_grav = - np.array([[-self.mass*self.grav],[0]])*np.ones((2,self.num_birds))
        #random
        f_rand = self.random_weight*(np.random.rand(2,self.num_birds)-1/2)
        
        self.F_flight = f_tau + f_lift + f_drag + f_thrust + f_grav + f_rand
        return
    
    def gen_vel_objects(self):
        self.vel = (self.bird_vel[0,:]**2 + self.bird_vel[1,:]**2)**(0.5)
        self.bird_dir = self.bird_vel/self.vel
        self.center = np.average(self.bird_pos,axis=1)
        self.center_dist = np.linalg.norm(self.center - np.transpose(self.bird_pos),axis=1)
        self.center_dist = self.center_dist/np.sum(self.center_dist)
        return
    
    def update_neighb(self):
        while self.open==0:
            self.tree = KDTree(np.transpose(self.bird_pos))
#            for i in range(self.num_birds):
            whichbird = np.random.randint(self.num_birds)
            self.neighborhood[whichbird,:] = self.tree.query(self.bird_pos[:,whichbird],k=self.num_neighb,p=2)[1]
            self.neighborhood_dists[whichbird,:] = self.tree.query(self.bird_pos[:,whichbird],k=self.num_neighb,p=2)[0]
        return
    
    def update_neighb_inside(self):
        self.tree = KDTree(np.transpose(self.bird_pos))
        for i in range(self.num_check_neighb):
            whichbird = np.random.choice(self.which_birds)
            self.neighborhood[whichbird,:] = self.tree.query(self.bird_pos[:,whichbird],k=self.num_neighb,p=2)[1]
            self.neighborhood_dists[whichbird,:] = self.tree.query(self.bird_pos[:,whichbird],k=self.num_neighb,p=2)[0]
    
    def inhibit(self):
        import ctypes
#        print("Preventing Windows from going to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(
            self.ES_CONTINUOUS | \
            self.ES_SYSTEM_REQUIRED)
        return

    def uninhibit(self):
        import ctypes
#        print("Allowing Windows to go to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(
            self.ES_CONTINUOUS)
        return
    
    def get_music_data(self):        
        self.data = np.frombuffer(self.stream.read(self.CHUNK),dtype=np.int16)
        T = 1.0 / (12*self.CHUNK)
        self.yf = np.abs(np.fft.fft(self.data))*self.volume
        self.yf = 2.0/self.CHUNK *np.abs(self.yf[:self.CHUNK//2])
        self.xf = np.linspace(0.0, 1.0/(2.0*T), int(self.CHUNK/2))
#        self.data = self.music_norm * self.data/np.max([np.max(self.data),1])
        self.bass_v = np.average(self.yf[0:5])
        self.mean_v = np.average(self.yf)
        return
    
    def bird_music_sample(self):
        if np.random.rand()<self.sample_fact and self.sample_on == 0 and self.beat == 1:
            self.sample_on = 1
            self.sample_start=time.time()
            self.sample_length = np.random.randint(3,10)
        if time.time()>self.sample_start+ 2*self.sample_length and self.beat == 1:
            self.sample_on=0
        if self.sample_on == 1 and time.time()<self.sample_start+self.sample_length:
            idx = np.round(np.linspace(0, len(self.data) - 1, self.num_birds)).astype(int)
            bird_idx = np.argsort(self.bird_pos[1,:]).astype(int)
            self.bird_pos[0,bird_idx] += self.data[idx]/200
        return
    
    def bird_music_beat(self):

        if self.bass_v > self.drop_v:
            self.bass += 1
        else:
            self.bass = 0
                          
        if self.bass == self.beat_bass_num:
            self.beat = 1
            self.bass = 0
        else:
            self.beat = 0
            
        return
    
    def bird_music_drop(self):
        if self.bass_v < self.drop_v:
            self.no_bass += 1
        else:
            self.no_bass -= 1
            self.no_bass = np.max([0,self.no_bass])
    
        if self.no_bass > self.drop_bass_num:
            self.drop = 1
        if self.beat == 1 and self.drop == 1:
            f_drop = np.transpose(self.bird_pos)-np.transpose(self.center)
            self.bird_vel += np.transpose(f_drop)*4*self.drop_fact
            self.drop = 0
            self.no_bass=0
            
    def frame_bars(self):
        if np.random.rand()<self.bar_fact and self.bars_on == 0 and self.beat == 1:
            self.bars_on = 1
            self.bar_start=time.time()
            self.bar_length = np.random.randint(20,60)
            self.num_bars = np.random.randint(8,16)
            self.barorients = np.random.randint(0,2,self.num_bars)      
            self.barwidth = np.random.randint(np.min([self.frame_yres,self.frame_xres])/60,np.min([self.frame_yres,self.frame_xres])/6,self.num_bars).astype(int) #widths of bars
            self.barlocmax = np.max(self.barwidth)+1 #farthest a bar can start from the left hand side
            for i in range(self.num_bars): #generate bar locations and the locations of bars they're copying from
                if self.barorients[i]==0:
                    self.barlocs = np.random.randint(1,self.frame_yres-self.barlocmax-10,self.num_bars)
                    self.barlocs2 = np.random.randint(1,self.frame_yres-self.barlocmax-10,self.num_bars)
                elif self.barorients[i]==1:
                    self.barlocs = np.random.randint(1,self.frame_xres-self.barlocmax-10,self.num_bars)
                    self.barlocs2 = np.random.randint(1,self.frame_xres-self.barlocmax-10,self.num_bars)
            self.bar_movement = np.random.choice([-1,0,1],size=self.num_bars)
                    
        if time.time()>self.bar_start+ 2*self.bar_length and self.beat == 1:
            self.bars_on=0
            
        if self.bars_on == 1 and time.time()<self.bar_start+self.bar_length:
                res = np.shape(self.frame)
                self.barlocs += self.bar_movement
#                self.barlocs2 += self.bar_movement
                for i in range(self.num_bars):
                    if self.barorients[i]==0: #swap vertical bars with other vertical bars
                        self.barlocs[i] = np.mod(self.barlocs[i],self.frame_yres)
                        width = np.min([self.barwidth[i],res[0]-self.barlocs[i],res[0]-self.barlocs2[i]])
                        if width > 1:
                            self.frame[self.barlocs[i]:self.barlocs[i]+width, :, :] = self.frame[self.barlocs2[i]:self.barlocs2[i]+width, :, :]
                    elif self.barorients[i]==1: #same but horizontally
                        self.barlocs[i] = np.mod(self.barlocs[i],self.frame_xres)
                        width = np.min([self.barwidth[i],res[1]-self.barlocs[i],res[1]-self.barlocs2[i]])
                        if width > 1:
                            self.frame[:, self.barlocs[i]:self.barlocs[i]+width, :] = self.frame[:, self.barlocs2[i]:self.barlocs2[i]+width, :]
        return
    
    def frame_tracers(self): #not working atm
        if np.random.rand()<self.tracer_fact and self.tracers_on == 0 and self.beat == 1:
            self.tracers_on = 1
            self.tracers_start=time.time()
            self.tracers_length = np.random.randint(10,30)
            self.num_tracers = np.random.randint(3,5)
            self.tracers_drift = 1
        if time.time()>self.tracers_start+ 2*self.tracers_length and self.beat == 1:
            self.tracers_on=0
        if self.tracers_on == 1 and time.time()<self.tracers_start+self.tracers_length:
#            self.tracers_drift = self.tracers_drift*1.2
#            for i in range(self.num_tracers):
            tracers_pos = self.old_bird_pos
            bird_draw_y = tracers_pos[0,:][(tracers_pos[0,:]>=0)&(tracers_pos[0,:]<self.frame_yres)&(tracers_pos[1,:]>=0)&(tracers_pos[1,:]<self.frame_xres)]
            bird_draw_x = tracers_pos[1,:][(tracers_pos[0,:]>=0)&(tracers_pos[0,:]<self.frame_yres)&(tracers_pos[1,:]>=0)&(tracers_pos[1,:]<self.frame_xres)]
            self.frame_birds[bird_draw_y,bird_draw_x,:] = np.transpose(np.vstack([np.vstack([self.rlist[int(self.order[0])],self.rlist[int(self.order[1])]]),self.rlist[int(self.order[2])]]))*1.5
        
        return
    
    def frame_shift(self):
        if np.random.rand()<self.shift_fact and self.shift_on == 0 and self.beat == 1:
            self.shift_on = 1
            self.shift_start=time.time()
            self.shift_length = np.random.randint(20,60)
            self.num_shift = np.random.randint(2,5)
            self.shift_drift = 1
            self.shift_axis = np.random.randint(2)
        if time.time()>self.shift_start+ 2*self.shift_length and self.beat == 1:
            self.shift_on=0
        if self.shift_on == 1 and time.time()<self.shift_start+self.shift_length:
            self.shift_drift *= 1.01
            for i in range(self.num_shift):
                self.frame += np.roll(self.frame,int(self.shift_drift*(i**2)),self.shift_axis)*0.5
        return
    
    def flip_hor(self):
        if self.center[1]<self.frame_xres/2:
            self.frame[:,int(self.frame_xres/2):-1,:] = np.flip(self.frame[:,0:int(self.frame_xres/2)-1,:],axis=1)
        else:
            self.frame[:,0:int(self.frame_xres/2)-1,:] = np.flip(self.frame[:,int(self.frame_xres/2):-1,:],axis=1)
    def flip_vert(self):
        if self.center[0]<self.frame_yres/2:
            self.frame[int(self.frame_yres/2):-1,:,:] = np.flip(self.frame[0:int(self.frame_yres/2)-1,:,:],axis=0)
        else:
            self.frame[0:int(self.frame_yres/2)-1,:,:] = np.flip(self.frame[int(self.frame_yres/2):-1,:,:],axis=0)      
    def flip_diag_45(self):
        lowlim = int(self.frame_xres/2 - self.frame_yres/2)
        highlim = lowlim+self.frame_yres
        if self.center[0]>self.center[1]-lowlim:
            self.frame[:,lowlim:highlim,0] = np.fliplr(np.triu(np.fliplr(self.frame[:,lowlim:highlim,0]))+np.transpose(np.triu(np.fliplr(self.frame[:,lowlim:highlim,0]))))
            self.frame[:,lowlim:highlim,1] = np.fliplr(np.triu(np.fliplr(self.frame[:,lowlim:highlim,1]))+np.transpose(np.triu(np.fliplr(self.frame[:,lowlim:highlim,1]))))
            self.frame[:,lowlim:highlim,2] = np.fliplr(np.triu(np.fliplr(self.frame[:,lowlim:highlim,2]))+np.transpose(np.triu(np.fliplr(self.frame[:,lowlim:highlim,2]))))
            
        else:
            self.frame[:,lowlim:highlim,0] = np.fliplr(np.tril(np.fliplr(self.frame[:,lowlim:highlim,0]))+np.transpose(np.tril(np.fliplr(self.frame[:,lowlim:highlim,0]))))
            self.frame[:,lowlim:highlim,1] = np.fliplr(np.tril(np.fliplr(self.frame[:,lowlim:highlim,1]))+np.transpose(np.tril(np.fliplr(self.frame[:,lowlim:highlim,1]))))
            self.frame[:,lowlim:highlim,2] = np.fliplr(np.tril(np.fliplr(self.frame[:,lowlim:highlim,2]))+np.transpose(np.tril(np.fliplr(self.frame[:,lowlim:highlim,2]))))
    
    def flip_diag_135(self):
        lowlim = int(self.frame_xres/2 - self.frame_yres/2)
        highlim = lowlim+self.frame_yres
        if self.center[0]>-self.center[1]+self.frame_yres+lowlim:
            self.frame[:,lowlim:highlim,0] = np.triu(self.frame[:,lowlim:highlim,0])+np.transpose(np.triu(self.frame[:,lowlim:highlim,0]))
            self.frame[:,lowlim:highlim,1] = np.triu(self.frame[:,lowlim:highlim,1])+np.transpose(np.triu(self.frame[:,lowlim:highlim,1]))
            self.frame[:,lowlim:highlim,2] = np.triu(self.frame[:,lowlim:highlim,2])+np.transpose(np.triu(self.frame[:,lowlim:highlim,2]))
        else:
            self.frame[:,lowlim:highlim,0] = np.tril(self.frame[:,lowlim:highlim,0])+np.transpose(np.tril(self.frame[:,lowlim:highlim,0]))
            self.frame[:,lowlim:highlim,1] = np.tril(self.frame[:,lowlim:highlim,1])+np.transpose(np.tril(self.frame[:,lowlim:highlim,1]))
            self.frame[:,lowlim:highlim,2] = np.tril(self.frame[:,lowlim:highlim,2])+np.transpose(np.tril(self.frame[:,lowlim:highlim,2]))        
        
    def frame_kal(self):
        if np.random.rand()<self.kal_fact and self.kal_on == 0 and self.beat == 1:
            self.kal_on = 1
            self.kal_start=time.time()
            self.kal_length = np.random.randint(20,60)
            self.kal_num_flips = np.random.choice([1,2,4],p=[.35*(1-(self.kal_fact*5)),.3,.35*(1+(self.kal_fact*5))])
            self.kal_which_flips = np.random.choice([0,1,2,3],size=self.kal_num_flips,replace=False)
        if time.time()>self.kal_start+ 2*self.kal_length and self.beat==1:
            self.kal_on=0
        if self.kal_on == 1 and time.time()<self.kal_start+self.kal_length:
            self.flips = [self.flip_hor,self.flip_vert,self.flip_diag_45,self.flip_diag_135]
            if self.kal_num_flips >= 1:
                self.flips[self.kal_which_flips[0]]()
            if self.kal_num_flips >= 2:
                self.flips[self.kal_which_flips[1]]()
            if self.kal_num_flips >= 4:
                self.flips[self.kal_which_flips[2]]()
                self.flips[self.kal_which_flips[3]]()
            if np.max(self.frame)==0:
                self.kal_on=0
        return
    
    def setpriority(self,pid=None,priority=1):
        priorityclasses = [win32process.IDLE_PRIORITY_CLASS,
                           win32process.BELOW_NORMAL_PRIORITY_CLASS,
                           win32process.NORMAL_PRIORITY_CLASS,
                           win32process.ABOVE_NORMAL_PRIORITY_CLASS,
                           win32process.HIGH_PRIORITY_CLASS,
                           win32process.REALTIME_PRIORITY_CLASS]
        if pid == None:
            pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, priorityclasses[priority])
#%%
v = Viz()
v.setpriority(priority=4)
v.draw_viz()
print('average time per frame = '+str((time.time()-v.starttime)/v.num_frames)+ ' s')

#%%

'''
to do:
    -coloring
        -birds going up are darker??
    -mapping sound to birds
    -progression of effects
        -clear - color
        -slowly increase sampling rate
        
        -kaleidoscope and bars eventually lol
'''


















