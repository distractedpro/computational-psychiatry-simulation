import numpy as np

class HealthyAgent:
    def __init__(self, alpha = 0.1, beta = 2.0, stage = 1):
        #core variables of behavioural loop

        
        self.beta = beta  #inverse temperature for softmax action selection(impulsivity vs rationality)
        self.q_values = np.zeros(2)  # Memory. Initial expectations for two choices (0 and 1)

        # -- THE SEVERITY DIAL: 0 = Healthy, 1 = Ruminating, 2 = Anhedonic --
        self.stage = stage  # Stage of the agent

        # --STAGE 1: RUMINATION PARAMETER--
        # Inflates loss learning (Lateral Habenula hyperreactivity) if stage is 1 or higher
        self.alpha_loss = 0.4 if self.stage >= 1 else alpha 

        # -- STAGE 2: ANHEDONIA (LIKING) PARAMETER --
        # only blunts sensory reward input if stage is 2 or higher
        # If true, throttles sensory reward input to 40% intensity
        self.reward_blunt_factor = 0.4 if self.stage >= 2 else 1.0
        
        # -- STAGE 3: ANHEDONIA (WANTING) PARAMETER --
        # Only slows positive learning if stage is 2 or higher
        # If true, positive updates crawl at 0.02. Disappointments stay at 0.1
        self.alpha_gain = 0.02 if self.stage >= 2 else alpha 
         

    def act(self):
        #mathematical translation of choice

        q_arr = np.array(self.q_values)  #Convert Q-values to a numpy array for vector mathematics

        exp_q = np.exp(self.beta * q_arr) #scaling by beta and exponentiating

        probabilities = exp_q / np.sum(exp_q)  # Softmax probabilities

        choice = np.random.choice([0, 1], p=probabilities)  # Choose action based on probabilities
        return choice

    def learn (self, choice, reward):

        #1. THE LIKING FILTER (Serotonergic Glitch)
        perceived_reward = reward * self.reward_blunt_factor

        #2. THE DOPAMINE SPIKE (Muted prediction error)
        prediction_error = perceived_reward - self.q_values[choice]

        #3. THE WANTING UPDATE (Dopaminergic learning glitch)
        if prediction_error > 0:
            self.q_values[choice] += self.alpha_gain * prediction_error
        else:
            self.q_values[choice] += self.alpha_loss * prediction_error
