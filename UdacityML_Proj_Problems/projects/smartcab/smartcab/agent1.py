import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    ALFA = 0.15     # Learning Rate
    GAMMA = 0.95     # Decay Rate

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.lastState = None
        self.reward = 0
        self.Q = {}
        self.EPSON = 1.8     # Explore Probability
        self.nTrials = 100
        self.kEpson = -0.02 # Epson dcrease slope
        self.t = 0          # Trial time
        self.track_epson = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.lastState = None
        self.reward = 0
        self.t += 1
        self.track_epson.append(100 * min(1, max(0, self.EPSON + self.kEpson * self.t)))

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Select action according to your policy
        action = self.takeAction([str(inputs), deadline, self.next_waypoint, ''])

        # TODO: Update state
        self.lastState = self.state
        self.state = (str(inputs), deadline, self.next_waypoint, str(action or ''))
        #print "State: ", self.state;
        
        # TODO: Learn policy based on state, action, reward
        self.learnDrive()

        # Execute action and get reward
        self.reward = self.env.act(self, action)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, self.reward)  # [debug]

    def learnDrive(self):
        ### Q(lastState + lastAction) <-Alfa- reward + Gamma * max{Q(currentState + AllActions)}
        lastQ = self.queryQ(self.lastState)
        
        # find maxQ for all actions for current state
        maxQ = 0
        l = list(self.state)
        for i_action in Environment.valid_actions:
            l[-1] = str(i_action or '')
            maxQ = max(maxQ, self.queryQ(tuple(l)))

        # update Q
        self.Q[self.lastState] = lastQ * (1 - LearningAgent.ALFA) + LearningAgent.ALFA * (self.reward + LearningAgent.GAMMA * maxQ)
        #print self.Q[self.lastState], " ---- Q of ----> ", self.lastState

    def queryQ(self, sin):
        if sin == None: return 0
        return self.Q.setdefault(sin, 0) # if not contain the key, init it with 0

    def takeAction(self, l):
        epsilon = self.EPSON + self.kEpson * self.t
        draw = random.random()
        #print "draw = ", draw
        if draw < epsilon:
            #print "choose to explore"
            return random.choice(Environment.valid_actions)
        else:
            #print "choose to used learned"
            maxQ = None
            maxAction = None
            first = True
            for i_action in Environment.valid_actions:
                l[-1] = str(i_action or '')
                curQ = self.queryQ(tuple(l))
                if first or maxQ < curQ :
                    maxQ = curQ
                    maxAction = i_action
                    first = False
            return maxAction


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    # Visualize the train result
    #sim = Simulator(e, update_delay=1, display=True)
    #sim.run(n_trials=3)

    # Analysis
    #print "DEADLINE TRACK: \n", e.track_deadline
    #print "NET REWARD TRACK: \n", e.track_net_reward

    #import matplotlib.pyplot as plt
    #plt.plot(e.track_deadline, label = "Deadline")
    #plt.plot(e.track_net_reward, label = "Net reward")
    #plt.plot(a.track_epson, label = "Epsilon [%]")
    #plt.legend()
    #plt.show()


if __name__ == '__main__':
    run()
