import numpy as np

class HealthyAgent:
    def __init__(self, alpha = 0.1, beta = 2.0):
        #core variables of behavioural loop

        self.alpha = alpha  #learning rate
        self.beta = beta  #inverse temperature for softmax action selection(impulsivity vs rationality)
        self.q_values = np.zeros(2)  # Memory. Initial expectations for two choices (0 and 1)

    def act(self):
        #mathematical translation of choice

        q_arr = np.array(self.q_values)  #Convert Q-values to a numpy array for vector mathematics

        exp_q = np.exp(self.beta * q_arr) #scaling by beta and exponentiating

        probabilities = exp_q / np.sum(exp_q)  # Softmax probabilities

        choice = np.random.choice([0, 1], p=probabilities)  # Choose action based on probabilities
        return choice

    def learn (self, choice, reward):

        # calculating dopamine (surprise factor)
        prediction_error = reward - self.q_values[choice]  # Calculate prediction error

        #updating memory
        self.q_values[choice] = self.q_values[choice] + (self.alpha * prediction_error)  # Update Q-value based on prediction error