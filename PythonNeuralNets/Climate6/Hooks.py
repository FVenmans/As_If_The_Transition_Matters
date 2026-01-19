import Parameters
from Parameters import policy, states, policy_states
import PolicyState
import State
import tensorflow as tf
import importlib

Definitions = importlib.import_module(Parameters.MODEL_NAME + ".Definitions")

def cycle_hook(state,i):
    policy_state = policy(state)
    for s in states:
        tf.summary.histogram("hist_" + s, getattr(State,s)(state), step=i)
        
    for p in policy_states:
        tf.summary.histogram("hist_" + p, getattr(PolicyState,p)(policy_state), step=i)

    return True

# Note we need to update output in initialization to be consistent
def post_init():
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "y_kf", Definitions.y_kf(Parameters.starting_state,None)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "y_kc", Definitions.y_kc(Parameters.starting_state,None)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "y_kd", Definitions.y_kd(Parameters.starting_state, None)))
    Parameters.starting_state.assign(State.update(Parameters.starting_state, "y_S", Definitions.y_S(Parameters.starting_state,None)))