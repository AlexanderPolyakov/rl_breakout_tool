import gymnasium as gym

env = gym.make('breakwall_clone:breakwall')
env.reset()
print("running ")
for i in range(1000): 
    env.step(1)
    env.render()

