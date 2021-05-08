# Breakout_DQN

Deep Q-Learning for Atari Breakout

Kafkalias A., Papayiannis K., Theodosiou M. Department of Computer Science University of Bath 2021

*We present a deep reinforcement learning model that learns to play the arcadegame Breakout.  Our agent is trained on the OpenAI Gym environment using avariant of Deep Q-Learning and a convolutional neural network that takes imagesof the game as input and outputs the expected reward for each action. We find thatour model performs 500% better than the average human score and receives analmost perfect score for the best runs.  We compare our model against a secondagent trained using the console RAM as input, and suggest several improvementsto further increase scores and decrease training times.*




<p align="justify">In this project we train a reinforcement learning agent to play the game of Breakout using Deep Q-Learning (DQN), drawing inspiration from the success of Mnih et al. (2013). DQN is an algorithm that combines the fields of Deep Learning and Reinforcement Learning, training agents to interact with complex environments such as those present in robotics and video games. Over the past years,video games have gained popularity as domains in which to apply Reinforcement Learning given their well defined rewards and self contained environments.Atari 2600 games are considered a challenging RL test-bed, providing a visual output of 210 x 160 RGB video at 60 Hz with millions of potential states. Learning to control agents from such complex environments is one of the open research areas within Reinforcement Learning, as traditional tabular methods cannot be used due to the large state space. We evaluate our agents performance against human benchmarks and results from literature, and compare performance against a second model trained using the Atari RAM the state representation.Our state representation is a 260x160x3 RBG image of the game screen, with rewards given for breaking bricks. Our action space has four possible actions: paddle left,paddle right, fire (resetting the ball after a lost life or environment reset) and no-op which keeps thepaddle stationary. An episode is terminal during evaluation when five lives are lost or all bricks are cleared, while for training we terminate episodes after the loss of one life. </p>
