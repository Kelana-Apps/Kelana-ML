from haversine import haversine, Unit
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Function to create a distance matrix
def create_distance_matrix(places):
    distance_matrix = []
    coords = list(places.values())
    
    for i in range(len(coords)):
        row = []
        for j in range(len(coords)):
            if i == j:
                row.append(0)
            else:
                # Using the Haversine formula to calculate distance in meters
                distance = haversine(coords[i], coords[j], unit=Unit.METERS)
                row.append(int(distance))  # Convert distance to integer
        distance_matrix.append(row)
    return distance_matrix

# Function to create data model
def create_data_model(places):
    distance_matrix = create_distance_matrix(places)
    data = {
        'distance_matrix': distance_matrix,
        'num_vehicles': 1,  # One vehicle
        'depot': 0  # Start from the first place
    }
    return data

# Function to print the solution
def print_solution(manager, routing, solution, places):
    route = []
    total_distance = 0
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        total_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    route.append(manager.IndexToNode(index))

    # List of tourist attractions based on the index in the distance matrix
    route_names = [list(places.keys())[i] for i in route]
    total_distance_km = total_distance / 1000  # Convert meters to kilometers

    return {
        'route': route_names,
        'total_distance': total_distance_km
    }

# Function to solve the TSP
def solve_tsp(places):
    data = create_data_model(places)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # Callback for the distance between places
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        return print_solution(manager, routing, solution, places)
    else:
        print('Tidak ada solusi ditemukan!')
        return None
