from environment import get_reward
from agent import HealthyAgent

def run_simulation():
    agent = HealthyAgent(alpha = 0.1, beta = 2.0)

    print("starting 100 trial simulation with healthy agent")

    for trial in range(1, 101):
        choice = agent.act()

        reward = get_reward(choice)

        agent.learn(choice, reward)

        if trial % 10 == 0:
            print(f"Trial {trial:03d} | Choice Made: {choice} | Brain Memory (Q-Values): [Opt0: {agent.q_values[0]:.2f}, Opt1: {agent.q_values[1]:.2f}]")

if __name__ == "__main__":
    run_simulation()