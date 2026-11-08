import random
def get_reward(choice):
    if choice == 0:
        if random.random() < 0.8:
            return 1
        else:
            return 0
    elif choice == 1:
        if random.random() < 0.2:
            return 1
        else:
            return 0
for i in range(10):
    choice = random.choice([0, 1])
    reward = get_reward(choice)
    print(f"Choice: {choice}, Reward: {reward}")