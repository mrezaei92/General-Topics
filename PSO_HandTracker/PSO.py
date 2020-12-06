# use case

#S=PSO(num_dimensions=25,lower_bound=0,higher_bound=10,num_particles=100)
#S.optimize(cost,100)
# to initialize: center_x,center_y=compute_center(IMG)
#nit=np.random.rand(1,ncomps+3 +2);nit[0,-2]=center_x.numpy();nit[0,-1]=center_y.numpy()

import random
import math
import numpy as np

#--- COST FUNCTION 
# function we are attempting to optimize (minimize)
def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[0,i]**2
    return total

def func2(x):
    return (x[0,0]-2)**2+x[0,1]**2



#--- MAIN 
class Particle:
    def __init__(self,n,ini=None):
        # n is the size of the vector, particles will be of size (1,n)
        
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1              # error individual
        self.particle_size=n
        if ini is not None:
            self.position_i=ini
            #print("yes")
        else:
            self.position_i=np.random.rand(1,n)
        self.velocity_i=np.random.uniform(-1,1,size=(1,n))

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        c1=2.8        # cognative constant
        c2=1.3        # social constant
        Gam=c1+c2
        w=2/abs(2-Gam-np.sqrt(Gam**2-4*Gam))

        
        r1=random.random()
        r2=random.random()

        vel_cognitive=c1*r1*(self.pos_best_i-self.position_i)
        vel_social=c2*r2*(pos_best_g-self.position_i)
        self.velocity_i=w*self.velocity_i+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,lower_bound,higher_bound):
        self.position_i=self.position_i+self.velocity_i
        self.position_i=np.clip(self.position_i,lower_bound,higher_bound)
        
    def set_position(self,pos):
        self.position_i=pos
    
                
class PSO():
    def __init__(self,num_dimensions,lower_bound,higher_bound,num_particles,ini=None):
        self.err_best_g=-1                   # best error for group
        self.pos_best_g=[]                   # best position for group
        self.num_particles=num_particles
        self.lower_bound=lower_bound
        self.higher_bound=higher_bound
        # establish the swarm
        self.swarm=[]
        for i in range(0,num_particles):
            self.swarm.append(Particle(num_dimensions,ini))

        
        
    def optimize(self,costFunc,num_iter,verbose=1,interval_rand=None,proportion_rand=0.25,dims=None,weights=None):
        # interval_rand denotes the the length of the inverval to randomly purturb proportion_rand of particles
        # along dimentions dims. dims should be a list of indices, like [1,2,3]
        # weights is a list where each element corresponds to the intensity by which the a certain dimension is randomized
        i=0
        save_interval=int(num_iter/verbose)
        randomized_population_size=np.int32(np.ceil(proportion_rand*self.num_particles))
        weights=np.array(weights).reshape(1,-1)
        for i in range(num_iter):
            
            if interval_rand is not None:
                if i%interval_rand==0:
                    randomly_selected=np.random.choice(self.num_particles,randomized_population_size,replace=False)
                    for j in range(0,len(randomly_selected)):
                        pos=self.swarm[randomly_selected[j]].position_i  # do some randomization
                        random_index=np.random.choice(len(dims),1)[0]
                        select_dim=dims[random_index]
                        pos[0,select_dim]=pos[0,select_dim]+np.random.rand(1,1)*weights[0,random_index]
                        self.swarm[randomly_selected[j]].position_i=pos
                        
                
            for j in range(0,self.num_particles):
                self.swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if self.swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g=self.swarm[j].position_i
                    self.err_best_g=self.swarm[j].err_i

            # cycle through swarm and update velocities and position
            for j in range(0,self.num_particles):
                self.swarm[j].update_velocity(self.pos_best_g)
                self.swarm[j].update_position(self.lower_bound,self.higher_bound)
#             print(i)
            if i%save_interval==0 and verbose is not None:
                print("iter {}, best error= {}".format(i,self.err_best_g))
    
        # print final results
        #print ('FINAL:')
        #print (self.pos_best_g)
        #print (self.err_best_g)


