from tools.bandits.sw_eps import SW_EpsilonGreedy
from tools.bandits.sw_ucb_u import SW_UCB_U
from tools.bandits.sw_uts import SW_UTS

### HARD CODED
# Definition of agents for the experiments
EPS_GREEDY_PARAMS = {
    "epsilon": 0.2,
    "no_actions": 100,
    "no_neighbors": 5,
    "sliding_window": 80,
}

UCB_U_PARAMS = {
    "alpha": 0.1,
    "no_actions": 100,
    "no_neighbors": 2,
    "sliding_window": 80,
}

UTS_PARAMS = {
    "std": 0.02,
    "no_actions": 100,
    "no_neighbors": 2,
    "sliding_window": 80,
}


def get_agent(agent_name):
    if agent_name == "epsilon_greedy":
        return SW_EpsilonGreedy(**EPS_GREEDY_PARAMS)
    elif agent_name == "ucb_u":
        return SW_UCB_U(**UCB_U_PARAMS)
    elif agent_name == "uts":
        return SW_UTS(**UTS_PARAMS)
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")
