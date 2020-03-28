from datetime import timedelta
import math
import argparse

from simobility.core import Clock
import simobility.routers as routers
from simobility.core import Fleet
from simobility.core import BookingService
from simobility.core import Dispatcher
from simobility.core.tools import ReplayDemand
from simobility.core import Booking


def create_demand_model(config, clock):

    from_datetime = clock.to_datetime()
    duration_mins = config["simulation"]["duration"]

    to_datetime = from_datetime + timedelta(minutes=duration_mins)

    num_bookings = (
        math.ceil(duration_mins / 60) * config["bookings"]["bookings_per_hour"]
    )

    round_to = clock.to_pandas_units()
    demand = ReplayDemand(
        clock,
        config["demand"]["data_file"],
        from_datetime,
        to_datetime,
        round_to,
        num_bookings,
        seed=config["simulation"].get("demand_seed"),
    )
    return demand


def create_config(data_file):
    return {
        "simulation": {
            # simulated duration in minutes
            "duration": 60,
            "demand_seed": 234,
            "fleet_seed": 2323,
            "clock_step": 25,
            "starting_time": "2015-02-01 12:00:00",
            
        },
        "demand": {"data_file": data_file},
        "fleet": {"vehicles": 10},
        "bookings": {"max_pending_time": 10, 'bookings_per_hour': 100},
    }


class TaxiService:
    def __init__(self, clock, dispatcher, router):
        self.clock = clock
        self.router = router
        self.dispatcher = dispatcher

        self.bookings_queue = []

    def add_booking_request(self, booking: Booking):
        self.bookings_queue.append(booking)

    def step(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("--demand-file", help="feather file with rides")
    parser.add_argument("--geofence-file", help="GeoJSON file")
    parser.add_argument("--stations-file", help="GeoJSON file")
    args = parser.parse_args()

    config = create_config(args.demand_file)

    clock = Clock(
        time_step=config["simulation"]["clock_step"],
        time_unit="s",
        starting_time=config["simulation"]["starting_time"],
    )

    router = routers.LinearRouter(clock=clock)

    fleet = Fleet(clock, router)
    fleet.infleet_from_geojson(
        config["fleet"]["vehicles"],
        args.stations_file,
        geofence_file=args.geofence_file,
        seed=config["simulation"].get("fleet_seed"),
    )

    max_pending_time = clock.time_to_clock_time(
        config["bookings"]["max_pending_time"], "m"
    )
    booking_service = BookingService(clock, max_pending_time)

    demand = create_demand_model(config, clock=clock)

    dispatcher = Dispatcher()

    taxi_service = TaxiService(clock, dispatcher, router)

    num_steps = clock.time_to_clock_time(config["simulation"]["duration"], "m")

    for i in range(num_steps):

        bookings = demand.next()
        for booking in bookings:
            taxi_service.add_booking_request(booking)

        taxi_service.step()

        fleet.step()

        dispatcher.step()

        clock.tick()
