### Research paper
[T. Oda and C. Joe-Wong, "Movi: A model-free approach to dynamic fleet management". 2018](https://arxiv.org/pdf/1804.04758.pdf)
>  MOVI, a Deep Q-network (DQN)-based framework that directly learns the optimal vehicle  dispatch  policy

### Tools and data

* [PyTorch](https://pytorch.org/)
* [NYC Taxi trip record data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
* [simobility](https://github.com/sash-ko/simobility)


### Basic idea

- create a service grid
- each vehicle can take 15x15 actions - move up to 7 cells up, down, left and right
- as an input Q-network takes (input shape 51x51):
  - predicted demand (demand model)
  - predicted supply - number of available vehicles per cell (supply model)
  - idling vehicles per region in 0, 15 and 30 minutes
- Q-network input centered around a vehicle
