
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

Action:
    move n vehicles from one region to another one
    u(t, i, j) - u vehicles from i to j at time t

Reward:
    >> To  define reward, we wish  to  minimize  three  performance  criteria:  the  number  
    >> of service rejects, passenger waiting time and idle cruising time. A reject means a 
    >> ride request that could not be served within a given amount of time because of no 
    >> available vehicles neara customer. Thewaiting timeis defined by the time between a 
    >> passenger’s placing a pickup request and the matched driverpicking up the  
    >> passenger; even  if a request is not rejected,passengers would prefer to  
    >> be  picked up sooner rather thanlater. Finally, theidle cruising timeis the 
    >> time in which a taxiis unoccupied and therefore not generating revenue, while 
    >> still incurring costs like gasoline and wear on the vehicle.

    reward(t, u(t)) = -lambda sum_i=1_M min(x(t, i) - w_hat(t, i), 0) - sum_i_j=1_M (eta(t ,i,j)*u(t, i,j))

    where:
        i and j - regions (cells)
        lambda - cost of reject
        M - number of regions (cells)
        w_hat(t, i) - number of predicted requests at time t

"""