import numpy as np
import time
from data import cc, tt
np.float_ = np.float64
import os
import csv


class Route:
    """
    Represents a route with pickup and delivery locations, timing, and distance information.
    """

    def __init__(self):
        self.nodes = []  # List of node IDs (pickup/delivery locations)
        self.requests = []  # List of request IDs (corresponding to k in original code)
        self.distance = 0  # Total distance of the route
        self.duration = 0  # Total duration of the route
        self.service_times = []  # List of service times at each node
        self.waiting_times = []  # List of waiting times at each node
        self.slacks = []  # List of slacks at each node
        self.pickup_delivery = []  # List of 'p' for pickup and 'd' for delivery
        self.capacity_slacks = [] # list of capacity slacks at each node

    def insert(self, request_id, insert_pickup_at, insert_delivery_at, data, dt):
        """
        Inserts a pickup and delivery pair into the route.

        Args:
            request_id: The ID of the request to insert.
            insert_pickup_at: The index where the pickup node should be inserted.
            insert_delivery_at: The index where the delivery node should be inserted.
            data: The data object containing problem information.
            dt: Time-related parameter.
        """
        pickup_node = data.k.loc[request_id].oid
        delivery_node = data.k.loc[request_id].did

        if insert_delivery_at == insert_pickup_at + 1:
            self.nodes.insert(insert_pickup_at, pickup_node)
            self.nodes.insert(insert_delivery_at, delivery_node)
            self.requests.insert(insert_pickup_at, request_id)
            self.requests.insert(insert_delivery_at, request_id)
            self.pickup_delivery.insert(insert_pickup_at, 'p')
            self.pickup_delivery.insert(insert_delivery_at, 'd')

        else:
            self.nodes.insert(insert_pickup_at, pickup_node)
            self.nodes.insert(insert_delivery_at + 1, delivery_node)
            self.requests.insert(insert_pickup_at, request_id)
            self.requests.insert(insert_delivery_at + 1, request_id)
            self.pickup_delivery.insert(insert_pickup_at, 'p')
            self.pickup_delivery.insert(insert_delivery_at + 1, 'd')

        self.service_times = self.calculate_service_times(data, dt)
        self.waiting_times = self.calculate_waiting_times(data, dt)
        self.distance = self.calculate_distance(data)
        self.duration = self.calculate_duration(data, dt)
        self.slacks = self.calculate_slack(data)
        self.capacity_slacks = self.calculate_capacity_slack(data)

    def calculate_service_times(self, data, dt):
        """Calculates the service time at each node."""
        service_times = [data.k.loc[self.requests[0]].l]
        for i in range(1, len(self.requests)):
            service_times.append(
                max(
                    data.k.loc[self.requests[i]].l,
                    service_times[i - 1]
                    + tt(self.nodes[i - 1], self.nodes[i], data.tt_m, dt),
                )
            )
        return service_times

    def calculate_waiting_times(self, data, dt):
        """Calculates the waiting time at each node."""
        waiting_times = [0]
        for i in range(1, len(self.requests)):
            waiting_times.append(
                max(
                    0,
                    data.k.loc[self.requests[i]].l
                    - (
                            self.service_times[i - 1]
                            + tt(self.nodes[i - 1], self.nodes[i], data.tt_m, dt)
                    ),
                )
            )
        return waiting_times

    def calculate_distance(self, data):
        """Calculates the total distance of the route."""
        distance = 0
        for i in range(1, len(self.nodes)):
            distance += cc(self.nodes[i - 1], self.nodes[i], data.d_m)
        return distance

    def calculate_duration(self, data, dt):
        """Calculates the total duration of the route."""
        duration = 0
        for i in range(1, len(self.nodes)):
            duration += tt(self.nodes[i - 1], self.nodes[i], data.tt_m, dt)
        return duration + sum(self.waiting_times)

    def calculate_slack(self, data):
        """Calculates the slack at each node."""
        slacks = [0] * len(self.requests)
        slacks[-1] = max(
            data.k.loc[self.requests[-1]].u
            - self.service_times[-1]
            + self.waiting_times[-1],
            0,
        )
        for i in range(len(slacks) - 2, -1, -1):
            forward_slack = slacks[i + 1]
            local_slack = (
                    data.k.loc[self.requests[i]].u
                    - self.service_times[i]
                    + self.waiting_times[i]
            )
            slacks[i] = max(min(local_slack, forward_slack), 0)
        return slacks
    def calculate_capacity_slack(self,data):
        capacity_slack = []
        for i in range(len(self.requests)):
            if i == 0:
                capacity_slack.append(100)
            else:
                if self.pickup_delivery[i-1] == 'd':
                    demand = -1*data.k.loc[self.requests[i-1]].nk
                else:
                    demand = data.k.loc[self.requests[i-1]].nk
                capacity_slack.append(capacity_slack[i-1] - demand)
        return capacity_slack

    def evaluate_insert(self, request_id, insert_pickup_at, insert_delivery_at, data, dt):
        """
        Evaluates if a request can be inserted at the given positions.
        Returns the slack increase if possible, -1 otherwise.
        """
        pickup_node = data.k.loc[request_id].oid
        delivery_node = data.k.loc[request_id].did
        lower_time_window = data.k.loc[request_id].l
        upper_time_window = data.k.loc[request_id].u

        if insert_pickup_at > len(self.requests):
            return -1

        if insert_pickup_at == len(self.requests):
            service_pickup = max(
                self.service_times[insert_pickup_at - 1]
                + tt(self.nodes[insert_pickup_at - 1], pickup_node, data.tt_m, dt),
                lower_time_window,
            )
            service_delivery = max(
                service_pickup + tt(pickup_node, delivery_node, data.tt_m, dt),
                lower_time_window,
            )
            if service_pickup > upper_time_window or service_delivery > upper_time_window:
                return -1
            else:
                return 0

        # Evaluate capacity
        if insert_delivery_at != 1:
            vehicle_capacity = min(self.capacity_slacks[insert_pickup_at:insert_delivery_at])
            if vehicle_capacity < data.k.loc[request_id].nk:
                return -1

        if insert_pickup_at == 0:
            service_pickup = lower_time_window
        else:
            service_pickup = max(
                self.service_times[insert_pickup_at - 1]
                + tt(self.nodes[insert_pickup_at - 1], pickup_node, data.tt_m, dt),
                lower_time_window,
            )

        if service_pickup > upper_time_window:
            return -1

        if insert_delivery_at == insert_pickup_at + 1:
            service_delivery = max(
                service_pickup + tt(pickup_node, delivery_node, data.tt_m, dt),
                lower_time_window,
            )
            if service_delivery > upper_time_window:
                return -1
            time_shift = max(
                0,
                service_delivery
                + tt(delivery_node, self.nodes[insert_pickup_at], data.tt_m, dt)
                - self.service_times[insert_pickup_at],
            )
            if time_shift > self.slacks[insert_pickup_at]:
                return -1
            else:
                return time_shift

        time_shift = max(
            0,
            service_pickup
            + tt(pickup_node, self.nodes[insert_pickup_at], data.tt_m, dt)
            - self.service_times[insert_pickup_at],
        )

        if time_shift > self.slacks[insert_pickup_at]:
            return -1

        if insert_delivery_at == len(self.requests):
            service_delivery = max(
                time_shift
                + self.service_times[insert_delivery_at - 1]
                + tt(self.nodes[insert_delivery_at - 1], delivery_node, data.tt_m, dt),
                lower_time_window,
            )
            if service_delivery > upper_time_window:
                return -1
            else:
                return time_shift

        service_delivery = max(
            time_shift
            + self.service_times[insert_delivery_at - 1]
            + tt(self.nodes[insert_delivery_at - 1], delivery_node, data.tt_m, dt),
            lower_time_window,
        )
        if service_delivery > upper_time_window:
            return -1

        time_shift2 = max(
            0,
            service_delivery
            + tt(delivery_node, self.nodes[insert_delivery_at], data.tt_m, dt)
            - self.service_times[insert_delivery_at],
        )
        if time_shift2 > self.slacks[insert_delivery_at]:
            return -1

        return time_shift + time_shift2

    def evaluate_cost(self, request_id, insert_pickup_at, insert_delivery_at, data, dt):
        """Evaluates the cost of inserting a request at given positions."""
        if self.evaluate_insert(
                request_id, insert_pickup_at, insert_delivery_at, data, dt
        ) == -1:
            return np.inf

        pickup_node = data.k.loc[request_id].oid
        delivery_node = data.k.loc[request_id].did

        if insert_pickup_at == len(self.requests):
            return cc(
                self.nodes[insert_pickup_at - 1], pickup_node, data.d_m
            ) + cc(pickup_node, delivery_node, data.d_m)

        if insert_pickup_at == 0:
            if insert_delivery_at == insert_pickup_at + 1:
                return cc(pickup_node, delivery_node, data.d_m) + cc(
                    delivery_node, self.nodes[insert_pickup_at], data.d_m
                )
            elif insert_delivery_at == len(self.requests):
                return cc(pickup_node, self.nodes[insert_pickup_at], data.d_m) + cc(
                    self.nodes[insert_delivery_at - 1], delivery_node, data.d_m
                )
            else:
                return cc(pickup_node, self.nodes[insert_pickup_at], data.d_m) + cc(
                    self.nodes[insert_delivery_at - 1], delivery_node, data.d_m
                ) + cc(
                    delivery_node, self.nodes[insert_delivery_at], data.d_m
                ) - cc(
                    self.nodes[insert_delivery_at - 1],
                    self.nodes[insert_delivery_at],
                    data.d_m,
                )

        else:
            if insert_delivery_at == insert_pickup_at + 1:
                return cc(
                    self.nodes[insert_pickup_at - 1], pickup_node, data.d_m
                ) + cc(pickup_node, delivery_node, data.d_m) + cc(
                    delivery_node, self.nodes[insert_pickup_at], data.d_m
                ) - cc(
                    self.nodes[insert_pickup_at - 1],
                    self.nodes[insert_pickup_at],
                    data.d_m,
                )
            elif insert_delivery_at == len(self.requests):
                return cc(
                    self.nodes[insert_pickup_at - 1], pickup_node, data.d_m
                ) + cc(pickup_node, self.nodes[insert_pickup_at], data.d_m) + cc(
                    self.nodes[insert_delivery_at - 1], delivery_node, data.d_m
                ) - cc(
                    self.nodes[insert_pickup_at - 1],
                    self.nodes[insert_pickup_at],
                    data.d_m,
                )
            else:
                return cc(
                    self.nodes[insert_pickup_at - 1], pickup_node, data.d_m
                ) + cc(pickup_node, self.nodes[insert_pickup_at], data.d_m) + cc(
                    self.nodes[insert_delivery_at - 1], delivery_node, data.d_m
                ) + cc(
                    delivery_node, self.nodes[insert_delivery_at], data.d_m
                ) - cc(
                    self.nodes[insert_pickup_at - 1],
                    self.nodes[insert_pickup_at],
                    data.d_m,
                ) - cc(
                    self.nodes[insert_delivery_at - 1],
                    self.nodes[insert_delivery_at],
                    data.d_m,
                )

    def find_best_insert(self, request_id, data, dt):
        """
        Finds the best place to insert a request into the route.

        Returns:
            A list [best_cost, request_id, best_pickup_index, best_delivery_index]
            or [] if no valid insertion is found.
        """
        insert_options = []
        for pickup_index in range(len(self.requests) + 1):
            for delivery_index in range(pickup_index + 1, len(self.requests) + 2):
                if pickup_index < len(self.requests) and delivery_index > len(self.requests):
                    continue
                cost = self.evaluate_cost(
                    request_id, pickup_index, delivery_index, data, dt
                )
                insert_options.append([cost, request_id, pickup_index, delivery_index])

        insert_options.sort()

        if len(insert_options) > 0:
            return insert_options[0]
        return []


def single_route_cost(k_index,data, dt):
    """Constructs an initial solution using a greedy approach for a single route."""
    request_ids = list(data.k.index)
    route = Route()
    route.insert(k_index, 0, 1, data, dt)
    requests_to_insert = [k for k in request_ids if k != k_index]

    while requests_to_insert:
        best_inserts = [
            route.find_best_insert(k, data, dt)
            for k in requests_to_insert
            if route.find_best_insert(k, data, dt)
        ]

        # Filter out inf values
        valid_inserts = [insert for insert in best_inserts if insert[0] != np.inf]

        if valid_inserts:
            valid_inserts.sort(key=lambda x: x[0])
            best_insert = valid_inserts[0]
            route.insert(
                best_insert[1], best_insert[2], best_insert[3], data, dt
            )
            requests_to_insert.remove(best_insert[1])
        else:
            break
    cost = 0
    for i in range(1, len(route.nodes)):
        cost += cc(route.nodes[i - 1], route.nodes[i], data.d_m)
    cost = cost/len(set(route.requests))
    return cost

def construct_solution(data, dt):
    """Constructs an initial solution using a greedy randomized approach."""
    # Use data.k.index directly for request IDs
    request_ids = list(data.k.index)
    kk = np.random.choice(request_ids)
    routes = [Route()]
    routes[0].insert(kk, 0, 1, data, dt)
    requests_to_insert = [k for k in request_ids if k != kk]

    while requests_to_insert:
        best_inserts = [
            routes[-1].find_best_insert(k, data, dt)
            for k in requests_to_insert
            if routes[-1].find_best_insert(k, data, dt)
        ]

        # Filter out inf values
        valid_inserts = [insert for insert in best_inserts if insert[0] != np.inf]

        if valid_inserts:
            valid_inserts.sort(key=lambda x: x[0])
            if len(valid_inserts) == 1:
                best_insert = valid_inserts[0]
            else:
                best_insert = valid_inserts[
                    np.random.choice(min(5, len(valid_inserts)))
                ]
            routes[-1].insert(
                best_insert[1], best_insert[2], best_insert[3], data, dt
            )
            requests_to_insert.remove(best_insert[1])
        else:
            routes.append(Route())
            kk = np.random.choice(requests_to_insert)
            routes[-1].insert(kk, 0, 1, data, dt)
            requests_to_insert.remove(kk)

    return routes


def calculate_solution_cost(solution, data):
    """Calculates the total cost of a solution."""
    cost = 0
    for route in solution:
        for i in range(1, len(route.nodes)):
            cost += 2 * np.ceil(cc(route.nodes[i - 1], route.nodes[i], data.d_m))
    return 500 * len(solution) + cost


def rgch(iterations, data, dt, time_limit=3600):
    """
    Runs the Randomized Greedy Construction Heuristic (RGCH).

    Args:
        iterations: The number of iterations to run.
        data: The problem data.
        dt: Time-related parameter (e.g., current time).

    Returns:
        A list containing the best solution found and its cost, sorted by cost (ascending).
        Format : [best_cost, best_solution]
    """
    solutions = []
    initial_time = time.time()
    for _ in range(iterations):
        solution = construct_solution(data, dt)
        cost = calculate_solution_cost(solution, data)
        solutions.append([cost, solution])
        if time.time() - initial_time > time_limit:
            break

    solutions.sort(key=lambda x: x[0])  # Sort by cost (ascending)
    return solutions[0]  # Return only the best solution

def save_log_to_csv(log_data, filepath):
    """
    Appends a dictionary of log data as a new row to a CSV file.
    Creates the file and header if it doesn't exist.
    """
    # Define a superset of all possible headers to ensure consistency
    fieldnames = [
        'timestamp', 'instance_id', 'log_type',
        'upper_bound', 'lower_bound', 'optimality_gap', 'wall_clock_time',
        'with_solution'
    ]

    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_data)

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from data import load_data

    # --- Define and Parse Script Parameters ---
    results_logfile = "upper_bound_rgch_results.csv"

    parser = argparse.ArgumentParser(
        description="Obtain a valid solution for the OFNDP from the unrestricted decomposition approach",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--instance_id', type=str, default='2',
                        help='The ID of the data instance to load (e.g., "1", "2").')
    args = parser.parse_args()
    instance_id = args.instance_id
    overall_start_time = time.time()
    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir / 'cbl_inst'
    data = load_data(data_folder, instance_id)
    rgch_solution = rgch(50, data, 15)
    wall_clock_time = round(time.time() - overall_start_time, 2)
    solution_summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'instance_id': instance_id,
        'log_type': 'rgch',
        'upper_bound': rgch_solution[0],
        'lower_bound': np.nan,
        'optimality_gap': np.nan,
        'wall_clock_time': wall_clock_time,
        'with_solution': True
    }
    print("\n" + "=" * 40 + "\n")
    print("--- Randomized Greedy Construction Heuristic Solution ---\n")
    for name, value in solution_summary.items():
        print(f"{name}: {value}")
    print("\n" + "=" * 40 + "\n")
    save_log_to_csv(solution_summary, results_logfile)