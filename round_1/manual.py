import argparse

# In the prompt, it is given that the reserve price is unknown but
# grows linearly from 900 to 1000.

# I assume that the pdf of the reserve price is parametrized by a (0 < a< 1)
# and it is given by
# f(x) = a/100 + (1-a)*(x-900)/50

# Let us denote the low bid as x1+900 and the high bid as x2+900
# Doing the integral, the expected reward is given by,

def reward_fun(a,x1,x2):
    result = 0
    result += a*(100-x1)*x1/100.0 +a*(100-x2)*(x2-x1)/100.0
    result += (1-a)*(100-x1)*(x1**2)/10000.0 +(1-a)*(100-x2)*(x2**2-x1**2)/10000.0
    return result

# Brute force the optimal solutions
def find_optimal(a):
    optim_pairs = 0
    optim_rewards = 0
    for x1 in range(0,101):
        for x2 in range(x1+1, 101):
            curr_reward = reward_fun(a, x1,x2)
            if curr_reward > optim_rewards:
                optim_rewards = curr_reward
                optim_pairs = (x1,x2)
    return optim_rewards, optim_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual 1")
    parser.add_argument('a')
    args = parser.parse_args()
    
    reward, bids  = find_optimal(float(args.a))
    print("reward : ", reward)
    print("(low bid, high bid) : ", bids)