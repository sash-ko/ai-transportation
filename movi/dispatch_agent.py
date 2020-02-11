
"""
T - time horison, e.g. predict demand from now T hours onwoard
M - number of regions (cells)

State:
    environment state at time t = (
        fleet state at time t,
        supply prediction (idling vehicles) per region from t till T,
        demand prediction (number of requests) per region from t till T,
    )

    fleet state at time t = [vehicle_1_state, vehicle_2_state, ..., vehicle_n_state]

    vehicle_n_state at time t = (
        - position
        - occupancy
        - destination
    )

Actions:
    >> Our  DQN  policy  learns  the  optimal  dispatch actions for individual
    >> vehicles. To do so, we suppose that all idle vehicles sequentially  decide
    >> where to go within a time  slot t.  Each vehicle’s  decision  accounts
    >> for the current locations  of  all other  vehicles, but does  not
    >> anticipate their future  actions. Since  drivers  have an app that
    >> updates  with other  drivers’ actions in real time, and it is unlikely
    >> that drivers would make decisions at the exact same times, they would have
    >> access to this knowledge

Reward:
    >> To  define reward, we wish  to  minimize  three  performance  criteria:  the  number  
    >> of service rejects, passenger waiting time and idle cruising time. A reject means a 
    >> ride request that could not be served within a given amount of time because of no 
    >> available vehicles neara customer. The waiting time is defined by the time between a 
    >> passenger’s placing a pickup request and the matched driver picking up the  
    >> passenger; even  if a request is not rejected, passengers would prefer to  
    >> be  picked up sooner rather thanlater. Finally, theidle cruising timeis the 
    >> time in which a taxiis unoccupied and therefore not generating revenue, while 
    >> still incurring costs like gasoline and wear on the vehicle.

    The DQN reward function for each vehicle n is a weighed sum of rides nth
    vehicle pickups up at time t and the total dispatch time c_t.

"""


"""
ARCHITECTURE

Models

q_network and q_network_target - dispatch network
demand_model
supply_model

Dispatch network

INPUT
51x51 size image with n-th vehicle in the middle
- predicted demand 51x51 matrix
- predicted supply 51x51x3 - number of available vehicles in 0, 15 and 30 minutes
- fleet state - number of idling vehicles in each region

OUTPUT
    15x15 matrix of actions: maximum 7 moves up, down, left or right


"""