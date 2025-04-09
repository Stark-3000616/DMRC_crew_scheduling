import csv
from datetime import timedelta
import networkx as nx
import random
import matplotlib.pyplot as plt
import inspect
import numpy as np
from collections import defaultdict
import gurobipy as gp 
from gurobipy import GRB

class Service:
    def __init__(self, attrs):
        self.serv_num = int(attrs[0])
        self.train_num = attrs[1]
        self.start_stn = attrs[2]
        self.start_time = hhmm2mins(attrs[3])
        self.end_stn = attrs[4]
        self.end_time = hhmm2mins(attrs[5])
        self.direction = attrs[6]
        self.serv_dur = int(attrs[7])
        self.jurisdiction = attrs[8]
        self.stepback_train_num = attrs[9]
        self.serv_added = False
        self.break_dur = 0
        self.trip_dur = 0

def hhmm2mins(hhmm):
    ''' Convert time from HH:MM format to minutes '''
    h, m = map(int, hhmm.split(':'))
    return h*60 + m

def mins2hhmm(mins):
    ''' Convert time from minutes to HH:MM format '''
    h = mins // 60
    m = mins % 60
    return f"{h:02}:{m:02}"

def fetch_data(filename, partial=False, rakes=10):
    ''' Fetch data from the given CSV file '''
    services = []
    services_dict = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            serv_obj = Service(row)
            if partial:
                if serv_obj.train_num in [f"{700+i}" for i in range(rakes+1)]:
                    services.append(serv_obj)
                    services_dict[serv_obj.serv_num] = serv_obj
            else:
                services.append(serv_obj)
                services_dict[serv_obj.serv_num] = serv_obj
    return services, services_dict

def draw_graph_with_edges(graph, n=50):
    ''' Draw the first n edges of the given graph '''
    # Create a directed subgraph containing only the first n edges
    subgraph = nx.DiGraph()
    
    # Add the first n edges and associated nodes to the subgraph
    edge_count = 0
    for u, v in graph.edges():
        if edge_count >= n:
            break
        if u != -2 and v != -1:
            subgraph.add_edge(u, v)
            edge_count += 1

    # Plotting the directed subgraph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(subgraph)  # Position nodes using the spring layout
    nx.draw_networkx_nodes(subgraph, pos, node_size=50, node_color='red')
    nx.draw_networkx_labels(subgraph, pos, font_size=15)
    nx.draw_networkx_edges(
        subgraph, pos, arrowstyle='->', arrowsize=20, edge_color='blue'
    )
    
    plt.title(f"First {n} Directed Edges of the Network")
    # plt.show()
    plt.savefig(f'first{n}edges.png')

def node_legal(service1, service2):
    ''' Check if two services can be connected '''

    if service1.stepback_train_num == "No Stepback":
        if service2.train_num == service1.train_num:
            if service1.end_stn == service2.start_stn and service2.start_time >= service1.end_time:
                return True
        else:
            if (service1.end_stn[:4] == service2.start_stn[:4]) and (service2.start_time >= service1.end_time):
                return True
        
    else:
        if service2.train_num == service1.stepback_train_num:
            if (service1.end_stn == service2.start_stn) and (service2.start_time >= service1.end_time ):
                return True
        else:
            if (service1.end_stn[:4] == service2.start_stn[:4]) and (service2.start_time >= service1.end_time):
                return True
    return False

def no_overlap(service1, service2):
    ''' Check if two services overlap in time '''
    return service1.end_time <= service2.start_time

def create_duty_graph(services):
    ''' 
        Creates a directed graph of services, with source and sink nodes at -2, -1 respectively

        Arguments: services - list of Service objects

        Returns: a directed graph G
    '''
    G = nx.DiGraph()

    for i, service1 in enumerate(services):
        G.add_node(service1.serv_num)

    G.add_node(-1) #end_node
    G.add_node(-2) #start_node

    for i, service1 in enumerate(services):
        for j, service2 in enumerate(services):
            if i != j:
                if node_legal(service1, service2):
                    G.add_edge(service1.serv_num, service2.serv_num, weight=service1.serv_dur)

    #end node edges
    for i, service in enumerate(services):
        G.add_edge(service.serv_num, -1, weight=service.serv_dur)
        # G.add_edge(service.serv_num, -1, weight=0)

    #start node edges
    for i, service in enumerate(services):
        G.add_edge(-2, service.serv_num, weight=0)
        
    return G

def extract_nodes(var_name):

    parts = var_name.split('_')
    if len(parts) != 3 or parts[0] != 'x':
        raise ValueError(f"Invalid variable name format: {var_name}")
    
    start_node = int(parts[1])
    end_node = int(parts[2])
    
    return start_node, end_node

def generate_paths(outgoing_var, show_paths = False):

    paths = []
    paths_decision_vars = []
    current = -2
    for start_path in outgoing_var[-2]:
        current_path = []
        current_path_decision_vars = []
        if start_path.x !=1:continue
        else:
            start, end = extract_nodes(start_path.VarName)
            # current_path.append(start_path.VarName)
            current_path.append(end)
            current_path_decision_vars.append(start_path)
            # start, end = extract_nodes(start_path.VarName)
            current = end
            while current != -1:
                for neighbour_edge in outgoing_var[current]:
                    if neighbour_edge.x !=1:continue
                    else:
                        start, end = extract_nodes(neighbour_edge.VarName)
                        current_path.append(end)
                        # current_path.append(neighbour_edge.VarName)
                        current_path_decision_vars.append(neighbour_edge)
                        # start, end = extract_nodes(neighbour_edge.VarName)
                        current = end
            paths.append(current_path)
            current_path.pop()
            paths_decision_vars.append(current_path_decision_vars)
            if show_paths:
                print(current_path)
    return paths, paths_decision_vars
# def generate_paths(outgoing_var, show_paths=False):
#     paths = []
#     paths_decision_vars = []
#     for start_path in outgoing_var[-2]:
#         current_path = []
#         current_path_decision_vars = []
#         # Try retrieving the solution value robustly
#         try:
#             val = start_path.x
#         except AttributeError:
#             val = start_path.getAttr("x")
#         if val != 1:
#             continue
#         else:
#             start, end = extract_nodes(start_path.VarName)
#             current_path.append(end)
#             current_path_decision_vars.append(start_path)
#             current = end
#             while current != -1:
#                 found = False
#                 for neighbour_edge in outgoing_var[current]:
#                     try:
#                         neigh_val = neighbour_edge.x
#                     except AttributeError:
#                         neigh_val = neighbour_edge.getAttr("x")
#                     if neigh_val != 1:
#                         continue
#                     else:
#                         start, end = extract_nodes(neighbour_edge.VarName)
#                         current_path.append(end)
#                         current_path_decision_vars.append(neighbour_edge)
#                         current = end
#                         found = True
#                         break
#                 if not found:
#                     break
#             paths.append(current_path)
#             # Remove the sink (-1) if present at the end (optional)
#             if current_path and current_path[-1] == -1:
#                 current_path.pop()
#             paths_decision_vars.append(current_path_decision_vars)
#             if show_paths:
#                 print(current_path)
#     return paths, paths_decision_vars


def solution_verify(services, duties):
    ''' 
    Checks if all services are assigned to a duty 
    '''
    flag = True
    for service in services:
        service_check = False
        for duty in duties:
            if service.serv_num in duty:
                service_check = True
                break
        if service_check == False:
            print(f"Service {service.serv_num} not assigned to any duty")
            flag= False
            break
    return flag

def roster_statistics(paths, service_dict):

    """
    service_dict: The dictionary of service times    
    """

    #1 Number of duties
    print("\nRoster Statistics:")
    print("Number of duties: ", len(paths))

    #2 Maximum number of services in a duty
    max_len_duty = 0
    min_len_duty = 1e9
    for duty in paths:
        if len(duty)>max_len_duty:
            max_len_duty = len(duty)
        if len(duty)<min_len_duty:
            min_len_duty = len(duty)

    print("Maximum number of services in a duty: ", max_len_duty-1)
    print("Minimum number of services in a duty: ", min_len_duty-1)

    #3 Maximum duration of a duty
    max_duration = 0
    min_duration = 1e9
    serv_dur_6 = 0
    serv_dur_8 = 0
    for duty in paths:
        current_duration = 0
        for service in duty:
            # start, end = extract_nodes(edge)
            if service != -2: current_duration += service_dict[service].serv_dur
        if current_duration > max_duration:
            max_duration = current_duration
        if current_duration < min_duration:
            min_duration = current_duration
        if current_duration > (6*60):
            serv_dur_6+=1
        if current_duration > (8*60):
            serv_dur_8+=1
            
    print("Maximum duration of duty: ", mins2hhmm(max_duration))
    print("Minimum duration of duty: ", mins2hhmm(min_duration))
    print("Duties with driving time more than 6hrs: ",  serv_dur_6)
    print("Duties with driving time more than 8hrs: ",  serv_dur_8)

def get_bad_paths(paths, paths_decision_vars, service_dict):
    bad_paths = []
    bad_paths_decision_vars = []
    for i in range(len(paths)):
        current_duration = 0
        for node in paths[i]:
            # start, end = extract_nodes(edge)
            # if node != -2: current_duration += service_dict[node].serv_dur
            if node != -2: current_duration += service_dict[node].serv_dur
        if current_duration > 6*60:
            bad_paths.append(paths[i])
            bad_paths_decision_vars.append(paths_decision_vars[i])

    return bad_paths, bad_paths_decision_vars

def get_lazy_constraints(bad_paths, bad_paths_decision_vars, service_dict):
    lazy_constraints = []
    for i in range(len(bad_paths)):
        current_duration = 0
        current_lazy_constr = []
        bad_paths[i].append(-2)  #to make the size of paths and path_decision_vars equal
        for j in range(len(bad_paths[i])):
            # start, end = extract_nodes(bad_paths[i][j])
            node = bad_paths[i][j]
            if node != -2: current_duration += service_dict[node].serv_dur
            current_lazy_constr.append(bad_paths_decision_vars[i][j])
            if current_duration > 6*60:
                lazy_constraints.append(current_lazy_constr)
                break

    return lazy_constraints

def can_append(duty, service):
    ''' Checking if service can be appended to duty or not '''
    last_service = duty[-1]
    
    start_end_stn_tf = last_service.end_stn == service.start_stn
    # print(service.start_time, last_service.end_time)
    start_end_time_tf = 5 <= (service.start_time - last_service.end_time) <= 15
    start_end_stn_tf_after_break = last_service.end_stn[:4] == service.start_stn[:4]
    start_end_time_within = 50 <= (service.start_time - last_service.end_time) <= 150

    if last_service.stepback_train_num == "No StepBack":
        start_end_rake_tf = last_service.train_num == service.train_num
    else:
        start_end_rake_tf = last_service.stepback_train_num == service.train_num
    
    # Check for valid conditions and time limits
    if start_end_rake_tf and start_end_stn_tf and start_end_time_tf:
        time_dur = service.end_time - duty[0].start_time
        cont_time_dur = sum([serv.serv_dur for serv in duty])
        if cont_time_dur <= 180 and time_dur <= 445:
            return True
    elif start_end_time_within and start_end_stn_tf_after_break:
        time_dur = service.end_time - duty[0].start_time
        if time_dur <= 445:
            return True
    return False

def solve_RMLP(services, duties, threshold=0):
    '''
    Solves the RMLP 

    Arguments: services - list of Service objects,
            duties - list of duties
    
    Returns: selected_duties - list of selected duties,
            dual_values - list of dual values for each service,
            selected_duties_vars - list of selected duty variables
            objective_value = objective value of the iteration
    '''
    objective = 0
    model = gp.Model("CrewScheduling")
    model.setParam('OutputFlag', 0)
    
    duty_vars = []
    for i in range(len(duties)):
        duty_vars.append(model.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=0, name=f"x{i}"))

    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    service_constraints = []
    for service_idx, service in enumerate(services):
        constr = model.addConstr(
            # gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty) >= 1,
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty) == 1,

            name=f"service_{service.serv_num}")
        service_constraints.append(constr)

    model.optimize()

    # Step 5: Check the solution and retrieve dual values and selected duties
    if model.status == GRB.INFEASIBLE:
        print('Infeasible problem!')
    elif model.status == GRB.OPTIMAL:
        objective = model.getObjective()
        # print("Optimal solution found")
        
        # Get the dual variables for each service constraint
        # dual_values = [constr.Pi for constr in service_constraints] 
        dual_values = {f"service_{service.serv_num}": constr.Pi for service, constr in zip(services, service_constraints)}

        selected_duties_vars = [v.varName for v in model.getVars() if v.x > threshold]
        selected_duties = [v for v in model.getVars() if v.x > threshold]
        
        return selected_duties, dual_values, selected_duties_vars, objective.getValue()
    else:
        print("No optimal solution found.")
        return None, None, None, None

# //////////////rmlp with model/////////////////////////////////
def solve_RMLP_test(services, duties, threshold=0):
    """
    Solves the Restricted Master LP (RMLP) for the crew scheduling problem.

    Arguments:
        services: List of Service objects.
        duties:   List of duties (each duty is a list of service numbers).
        threshold: A threshold to decide if a duty variable is “active” (default 0).

    Returns:
        selected_duties: List of indices for duties with variable value > threshold.
        dual_values: Dictionary mapping each service constraint name (e.g., "cover_service_X") to its dual value.
        selected_duty_vars: List of duty variable names (those with value > threshold).
        obj: The LP objective value.
        model: The Gurobi model object (so you can extract variable values later).
    """
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model.
    model = gp.Model("RMLP")
    # Optionally suppress output:
    model.setParam('OutputFlag', 0)

    # Create a binary variable for each duty (column).
    duty_vars = {}
    for i, duty in enumerate(duties):
        # Here, "duty_i" is the variable name.
        duty_vars[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f"duty_{i}")
    model.update()

    # For each service, add a coverage constraint:
    # The idea is that for each service, the sum of duty variables for those duties that cover it equals 1.
    for service in services:
        # Gather all duty variables that cover this service.
        relevant_vars = [duty_vars[i] for i, duty in enumerate(duties) if service.serv_num in duty]
        if not relevant_vars:
            print(f"Warning: Service {service.serv_num} is not covered by any duty!")
        else:
            model.addConstr(gp.quicksum(relevant_vars) == 1, name=f"cover_service_{service.serv_num}")

    # Set the objective to minimize the total number of duties selected.
    # (Often the LP relaxation uses continuous variables, so the LP objective will be a lower bound.)
    model.setObjective(gp.quicksum(duty_vars[i] for i in duty_vars), GRB.MINIMIZE)

    # Solve the model.
    model.optimize()

    # Check for optimality.
    if model.status != GRB.OPTIMAL:
        print("No optimal solution found in solve_RMLP. Model status:", model.status)
        return None, None, None, None, model

    # Extract selected duties based on the threshold.
    selected_duties = [i for i in duty_vars if duty_vars[i].X > threshold]

    # Extract dual values from the coverage constraints.
    dual_values = {}
    for service in services:
        constr = model.getConstrByName(f"cover_service_{service.serv_num}")
        if constr is not None:
            dual_values[f"service_{service.serv_num}"] = constr.Pi

    # Also extract the names of the duty variables that have a positive value.
    selected_duty_vars = [duty_vars[i].VarName for i in duty_vars if duty_vars[i].X > threshold]

    obj = model.ObjVal

    return selected_duties, dual_values, selected_duty_vars, obj, model

def greedy_crew_schedule_lp(services, duties, lp_values):
    """
    A greedy heuristic that uses LP fractional values to warm-start the selection.
    
    Arguments:
        services: List of Service objects, each with a unique serv_num.
        duties:   List of duties (each duty is a list of service numbers).
        lp_values: Dictionary mapping duty index to its LP fractional value.
                   
    Returns:
        selected_duties: List of duty indices chosen by the heuristic.
        uncovered:       Set of service numbers that remain uncovered.
    """
    # Create a set of all service numbers that need to be covered.
    uncovered = set(service.serv_num for service in services)
    
    # Precompute each duty's coverage as a set for faster lookup.
    duty_coverage = {i: set(duty) for i, duty in enumerate(duties)}
    
    # Rank the duties by their LP value (higher first).
    ranked_duties = sorted(range(len(duties)), key=lambda i: lp_values.get(i, 0), reverse=True)
    
    selected_duties = []
    
    # Iterate over the ranked duties and select those that cover uncovered services.
    for i in ranked_duties:
        current_cov = duty_coverage[i].intersection(uncovered)
        if current_cov:
            selected_duties.append(i)
            uncovered -= duty_coverage[i]
        if not uncovered:
            break
    
    return selected_duties, uncovered




def new_duty_with_bellman_ford(graph, dual_values):

    '''
    Finds a new duty using NetworkX Bellman-Ford algorithm

    Arguments: graph - directed graph of services,
            dual_values - list of dual values for each service

    Returns: path - list of services in the new duty,
            length - length of the new duty,
            graph_copy - copy of the graph with adjusted edge weights
    '''
    graph_copy = graph.copy()
    for u, v in graph_copy.edges():
        if u != -2:
            service_idx_u = u
            # dual_u = dual_values[service_idx_u]
            dual_u = dual_values[f"service_{service_idx_u}"]

            graph_copy[u][v]['weight'] = -(dual_u)  # Adjust edge weight by dual value
        # else:
        #     service_idx_u = -2
        #     dual_u = dual_values[service_idx_u]
        #     graph_copy[u][v]['weight'] = -(dual_u)
    

    path = nx.bellman_ford_path(graph_copy, -2, -1, weight='weight')
    length = nx.bellman_ford_path_length(graph_copy, -2, -1, weight='weight')

    return path, length, graph_copy

def new_duty_with_bellman_ford_2(graph, dual_values, service_dict, max_duration=360):
    """
    Finds a new duty using NetworkX Bellman-Ford algorithm while ensuring total duty duration <= max_duration.

    Arguments:
        graph - directed graph of services
        dual_values - dictionary of dual values for each service (key: "Service_{service_id}")
        max_duration - maximum allowed duration for the new duty

    Returns:
        path - list of services in the new duty
        length - length of the new duty (shortest path distance)
        graph_copy - copy of the graph with adjusted edge weights
    """
    graph_copy = graph.copy()

    # Adjust edge weights based on dual values
    for u, v in graph_copy.edges():
        if u != -2:  # Exclude the source node from dual value mapping
            service_idx_u = u
            dual_u = dual_values.get(f"Service_{service_idx_u}", 0)  # Default to 0 if not found

            graph_copy[u][v]['weight'] = -dual_u  # Adjust edge weight by dual value

    # Bellman-Ford with duration constraint
    dist = {node: float('inf') for node in graph_copy}
    duration = {node: float('inf') for node in graph_copy}  # Track cumulative duty duration
    predecessor = {node: None for node in graph_copy}

    dist[-2] = 0  # Source node
    duration[-2] = 0

    # Relaxation for |V| - 1 iterations
    for _ in range(len(graph_copy) - 1):
        for u, v, data in graph_copy.edges(data=True):
            weight = data['weight']  # Dual-adjusted cost
            time = service_dict[u].serv_dur if u not in [-2, -1] else 0  # Edge duration attribute

            # Relaxation step with duty duration constraint
            if dist[u] + weight < dist[v] and duration[u] + time <= max_duration:
                dist[v] = dist[u] + weight
                duration[v] = duration[u] + time
                predecessor[v] = u  # Store path

    # Reconstruct path from sink (-1) to source (-2)
    path = []
    current = -1
    while current is not None:
        path.append(current)
        current = predecessor[current]
    path.reverse()

    # Ensure path is valid and meets duration constraint
    if duration[-1] > max_duration:
        print("No valid duty found within duration constraint.")
        return None, None, graph_copy

    return path, dist[-1], graph_copy
# def new_duty_with_bellman_ford_2(graph, dual_values, max_duration=360,):
#     """
#     Finds a new duty using NetworkX Bellman-Ford algorithm while ensuring total duty duration <= max_duration.

#     Arguments:
#         graph - directed graph of services
#         dual_values - dictionary of dual values for each service (key: "Service_{service_id}")
#         max_duration - maximum allowed duration for the new duty

#     Returns:
#         path - list of services in the new duty
#         length - length of the new duty (shortest path distance)
#         graph_copy - copy of the graph with adjusted edge weights
#     """
#     graph_copy = graph.copy()

#     # Adjust edge weights based on dual values
#     for u, v in graph_copy.edges():
#         if u != -2:  # Exclude the source node from dual value mapping
#             service_idx_u = u
#             dual_u = dual_values.get(f"Service_{service_idx_u}", 0)  # Default to 0 if not found

#             graph_copy[u][v]['weight'] = -dual_u  # Adjust edge weight by dual value

#     # Bellman-Ford with duration constraint
#     dist = {node: float('inf') for node in graph_copy}
#     duration = {node: float('inf') for node in graph_copy}  # Track cumulative duty duration
#     predecessor = {node: None for node in graph_copy}

#     dist[-2] = 0  # Source node
#     duration[-2] = 0
#     time = 0

#     # Relaxation for |V| - 1 iterations
#     for _ in range(len(graph_copy) - 1):
#         for u, v, data in graph_copy.edges(data=True):
#             # weight = data.get("weight", 1)  # Dual-adjusted cost
#             weight = graph[u][v]["weight"]  # Fallback to default weight if not found
#             time = service_dict[u].serv_dur if u not in [-2, -1] else 0  # Service duration for u

#             # Relaxation step with duty duration constraint
#             if dist[u] + weight < dist[v] and time + duration[u] <= max_duration:
#                 dist[v] = dist[u] + weight
#                 duration[v] = duration[u] + time
#                 predecessor[v] = u  # Store path

#     # Reconstruct path from sink (-1) to source (-2)
#     path = []
#     current = -1
#     while current is not None:
#         path.append(current)
#         current = predecessor[current]
#     path.reverse()

#     # Ensure path is valid and meets duration constraint
#     if duration[-1] > max_duration:
#         print("No valid duty found within duration constraint.")
#         return None, None, graph_copy

#     return path, dist[-1], graph_copy

        
def updated_new_duty_with_bellman_ford(graph, dual_values, service_dict, max_duration=360):
    """
    Finds a new duty (a path from -2 to -1) using a plain Bellman-Ford algorithm
    that minimizes cost (using adjusted edge weights from dual values) while ensuring
    that the cumulative duty duration (sum of service durations) does not exceed max_duration.

    Arguments:
        graph       - Directed graph of services (nodes), with source (-2) and sink (-1).
        dual_values - Dictionary of dual values (e.g., {"service_123": value, ...}).
        service_dict- Dictionary mapping service numbers to Service objects (each having serv_dur).
        max_duration- Maximum allowed cumulative duty duration (default is 360 minutes).

    Returns:
        path       - List of nodes representing the duty (from -2 to -1), or None if no feasible path.
        best_cost  - Total (reduced) cost of the found path (or infinity if no path found).
        graph_copy - A copy of the graph with adjusted edge weights.
    """
    # import math
    # Create a copy of the graph and adjust edge weights based on dual values.
    graph_copy = graph.copy()
    sink_edge_weights = []
    for u, v in graph_copy.edges():
        # graph_copy[u][-1]['weight'] = 0
        service_idx_u = u
        if u != -2:
            dual_u = dual_values[f"service_{service_idx_u}"]
            graph_copy[u][v]['weight'] = -(dual_u)
        # if v == -1:
        #     sink_edge_weights.append(graph_copy[u][v]['weight'])

    # print("edge weights to sink",sink_edge_weights)
       
    # Initialize dictionaries for cost, cumulative duration, and predecessor pointers.
    nodes = list(graph_copy.nodes())
    INF = float('inf')
    cost = {node: INF for node in nodes}
    duration = {node: INF for node in nodes}
    pred = {node: None for node in nodes}
    
    # The source node (-2) has cost 0 and duration 0.
    cost[-2] = 0
    duration[-2] = 0
    
    # Perform up to (|V| - 1) relaxations.
    for _ in range(len(nodes) - 1):
        updated = False
        # For each edge, try to relax.
        for u, v in graph_copy.edges():
            # If u is reachable...
            if cost[u] < INF:
                # Additional duration for node v: if v is a service node, add its service duration.
                add_dur = service_dict[v].serv_dur if v not in [-2, -1] else 0
                new_dur = duration[u] + add_dur
                # Only relax if the new cumulative duration is within allowed limit.
                if new_dur <= max_duration:
                    new_cost = cost[u] + graph_copy[u][v]['weight']
                    if new_cost < cost[v]:
                        cost[v] = new_cost
                        duration[v] = new_dur
                        pred[v] = u
                        updated = True
        # No updates in this iteration means we can stop early.
        if not updated:
            break

    # If the sink (-1) is unreachable within the duration constraint, return None.
    if cost[-1] == INF:
        return None, INF, graph_copy

    # Reconstruct the path from sink (-1) back to source (-2) using predecessor pointers.
    path = []
    current = -1
    while current is not None:
        path.append(current)
        current = pred[current]
    path.reverse()
    cost_final = 1 + cost[-1]
    return path, cost_final, graph_copy

# def updated_new_duty_with_bellman_ford(graph, dual_values, service_dict, max_duty_duration=360):
#     """
#     Finds a new duty using a modified Bellman–Ford algorithm that considers both cost and duty duration.
#     Instead of returning None if the lowest-cost path violates the maximum duty duration, it iteratively
#     removes the edge that first causes the cumulative duration to exceed the allowed limit and re-runs Bellman–Ford.
#     Logs key steps along the way.
    
#     Parameters:
#       graph: a directed graph of services (nodes represent services; -2 is the source and -1 is the sink)
#       dual_values: a dictionary mapping keys like "service_<service_id>" to dual values
#       service_dict: a dictionary mapping service id to a Service object (which has attribute serv_dur)
#       max_duty_duration: maximum allowed duty duration in minutes (default is 360)
      
#     Returns:
#       (path, total_cost, total_duration, graph_used)
#         - path: list of node IDs (services) forming the duty, or None if no feasible duty is found.
#         - total_cost: the computed cost (using the adjusted weights)
#         - total_duration: cumulative service duration along the path (excluding source and sink)
#         - graph_used: the modified graph used in the computation.
#     """
#     # Create a working copy of the graph.
#     current_graph = graph.copy()
    
#     # Adjust edge weights using dual values for non-source nodes.
#     for u, v in current_graph.edges():
#         if u != -2:
#             dual_u = dual_values.get(f"service_{u}", 0)
#             current_graph[u][v]['weight'] = -dual_u
#     print("[LOG] Initialized graph with adjusted weights based on dual values.")

#     while True:
#         try:
#             candidate_path = nx.bellman_ford_path(current_graph, -2, -1, weight='weight')
#             candidate_cost = nx.bellman_ford_path_length(current_graph, -2, -1, weight='weight')
#             print(f"[LOG] Candidate path found: {candidate_path} with cost {candidate_cost:.2f}")
#         except (nx.NetworkXNoPath, nx.NetworkXUnbounded):
#             print("[LOG] No candidate path exists in the modified graph.")
#             return None, None, None, current_graph

#         # Calculate the cumulative duration along the candidate path (ignoring source -2 and sink -1).
#         total_duration = sum(service_dict[node].serv_dur for node in candidate_path if node not in [-2, -1])
#         print(f"Total duration {total_duration} minutes.")

#         if total_duration <= max_duty_duration:
#             print("[LOG] Candidate path meets the duration constraint.")
#             return candidate_path, candidate_cost, total_duration, current_graph
#         else:
#             print("[LOG] Candidate path exceeds the maximum duration. Searching for the problematic edge.")
#             # Identify the first service in the candidate path that makes the cumulative duration exceed the limit.
#             cumulative = 0
#             removal_index = None
#             # candidate_path[0] is expected to be -2 (the source), so we start from index 1.
#             for i in range(1, len(candidate_path) - 1):  # skip the sink (-1)
#                 node = candidate_path[i]
#                 cumulative += service_dict[node].serv_dur
#                 if cumulative > max_duty_duration:
#                     removal_index = i
#                     print(f"[LOG] Cumulative duration reached {cumulative} minutes at node {node}, which exceeds the limit.")
#                     break

#             # If no removal index was found (unexpected), return no path.
#             if removal_index is None:
#                 print("[LOG] No problematic edge found even though duration exceeded. Exiting.")
#                 return None, candidate_cost, total_duration, current_graph

#             # Remove the edge from the previous node to the problematic node.
#             u = candidate_path[removal_index - 1]
#             v = candidate_path[removal_index]
#             if current_graph.has_edge(u, v):
#                 current_graph.remove_edge(u, v)
#                 print(f"[LOG] Removed edge from {u} to {v} to force an alternative path.")
#             else:
#                 print(f"[LOG] Edge from {u} to {v} already missing. Exiting.")
#                 return None, candidate_cost, total_duration, current_graph


# /////////////////////////// 2nd approach -a resource‐constrained shortest path (RCSP) algorithm. //////////////////////////////

# def new_duty_with_RCSP(graph, dual_values, service_dict, max_resource):
#     """
#     Finds a new duty (path from -2 to -1) using a Resource-Constrained Shortest Path algorithm.
    
#     Parameters:
#       graph         - The directed graph (NetworkX DiGraph) of services (nodes) with start (-2) and sink (-1).
#       dual_values   - A dictionary of dual values, e.g. { "service_123": value, ... }.
#       service_dict  - A dictionary mapping service number to Service objects (providing serv_dur).
#       max_resource  - The maximum allowed resource consumption (e.g., maximum duty duration in minutes).
      
#     Returns:
#       best_path     - The list of nodes representing the new duty (including -2 at start and -1 at end).
#       best_cost     - The associated cost (reduced cost) of that path.
#       labels        - The complete dictionary of labels (for debugging purposes).
#     """
#     # Each label is a tuple: (current_node, total_cost, total_resource, path)
#     # Initialize labels at the source node (-2)
#     labels = { -2: [(-2, 0, 0, [-2])] }
    
#     # Use a simple queue to process nodes (could be improved with a priority queue)
#     queue = [-2]
    
#     while queue:
#         u = queue.pop(0)
#         for label in labels[u]:
#             current_node, current_cost, current_resource, current_path = label
            
#             # For each successor of u
#             for v in graph.successors(u):
#                 # If v is a service (not source or sink), add its duration
#                 additional_resource = 0
#                 if v not in [-2, -1]:
#                     additional_resource = service_dict[v].serv_dur
#                 new_resource = current_resource + additional_resource
#                 # Skip this extension if resource limit is exceeded
#                 if new_resource > max_resource:
#                     continue

#                 # Define the additional cost.
#                 # if u is not the source, you might set:
#                 additional_cost = 0
#                 if u != -2:
#                     # Use the dual value for service u
#                     additional_cost = -dual_values.get(f"service_{u}", 0)
#                 new_cost = current_cost + additional_cost
#                 new_path = current_path + [v]
#                 new_label = (v, new_cost, new_resource, new_path)
                
#                 # Dominance check: prune labels at node v.
#                 dominated = False
#                 non_dominated = []
#                 for existing in labels.get(v, []):
#                     # If existing label is as good or better in both cost and resource, discard new label.
#                     if existing[1] <= new_cost and existing[2] <= new_resource:
#                         dominated = True
#                         break
#                     # If new_label dominates existing, we skip keeping the existing label.
#                     if new_cost <= existing[1] and new_resource <= existing[2]:
#                         continue
#                     non_dominated.append(existing)
#                 if dominated:
#                     continue
#                 # Update label list at v
#                 updated_labels = non_dominated + [new_label]
#                 labels[v] = updated_labels
#                 if v not in queue:
#                     queue.append(v)
    
#     # Check if any labels reached the sink (-1)
#     if -1 in labels:
#         # Select the label with the smallest cost at the sink
#         best_label = min(labels[-1], key=lambda x: x[1])
#         best_path = best_label[3]
#         best_cost = best_label[1]
#         return best_path, best_cost, labels
#     else:
#         # No feasible path found under the resource constraint
#         return None, None, labels


def new_duty_with_RCSP(graph, dual_values, service_dict, max_resource):
    """
    Finds a new duty (path from -2 to -1) using an RCSP algorithm.
    
    Parameters:
      graph         - The directed graph (NetworkX DiGraph).
      dual_values   - Dictionary with dual values (e.g., {"service_1": value, ...}).
      service_dict  - Dictionary mapping service number to Service objects.
      max_resource  - Maximum allowed resource (e.g., maximum duty duration in minutes).
      
    Returns:
      best_path     - The path (list of nodes) for the new duty.
      best_cost     - The associated reduced cost.
      labels        - Dictionary of labels at each node.
    """
    # Initialize label at source (-2)
    labels = {-2: [(-2, 0, 0, [-2])]}  # (current_node, cost, resource, path)
    queue = [-2]
    
    while queue:
        u = queue.pop(0)
        for label in labels[u]:
            current_node, current_cost, current_resource, current_path = label
            for v in graph.successors(u):
                # Calculate the transition time from u to v:
                transition_time = 0
                if u not in [-2, -1] and v not in [-2, -1]:
                    # Ensure service u has an end time and service v has a start time.
                    transition_time = max(0, service_dict[v].start_time - service_dict[u].end_time)
                # Otherwise, for edges involving source/sink, you may define transition_time as 0.
                
                # Calculate the new service duration at v (if v is a service)
                additional_duration = service_dict[v].serv_dur if v not in [-2, -1] else 0
                
                # new_resource = current_resource + transition_time + additional_duration
                new_resource = current_resource + additional_duration
                # If new resource exceeds limit, skip
                if new_resource > max_resource:
                    continue
                
                # Update cost; here we subtract the dual value for node u if applicable.
                additional_cost = -dual_values.get(f"service_{u}", 0) if u not in [-2] else 0
                new_cost = current_cost + additional_cost
                new_path = current_path + [v]
                new_label = (v, new_cost, new_resource, new_path)
                
                # Dominance check at node v
                dominated = False
                non_dominated = []
                for existing in labels.get(v, []):
                    if existing[1] <= new_cost and existing[2] <= new_resource:
                        dominated = True
                        break
                    if not (new_cost <= existing[1] and new_resource <= existing[2]):
                        non_dominated.append(existing)
                if dominated:
                    continue
                labels[v] = non_dominated + [new_label]
                if v not in queue:
                    queue.append(v)
                    
    if -1 in labels:
        best_label = min(labels[-1], key=lambda x: x[1])
        return best_label[3], best_label[1], labels
    else:
        return None, None, labels


# ///////////////////////////////////////////////////////////////////////////////////////////////
# topo then dp optimized
import networkx as nx
import bisect

def pricing_rcsp_dag_full(graph, duals, service_dict, R=360):
    """
    DAG pricing: maximize dual-sum subject to duration <= R.
    Tail-charging, full dominance, labels = (d, r, pred_u, pred_label_idx).
    Returns (best_path, reduced_cost) or (None, inf) if no improving column.
    """

    # 1. Topo sort + index
    topo = list(nx.topological_sort(graph))
    idx  = {node:i for i,node in enumerate(topo)}
    N    = len(topo)

    # 2. Precompute dur[] and succ[]
    dur  = [0]*N
    succ = [[] for _ in range(N)]
    for node in topo:
        i = idx[node]
        if node not in (-2, -1):
            dur[i] = service_dict[node].serv_dur
        for nbr in graph.successors(node):
            succ[i].append(idx[nbr])

    # 3. Precompute minRemDur[u]
    minRemDur = [float('inf')]*N
    sink_i    = idx[-1]
    minRemDur[sink_i] = 0
    for node in reversed(topo):
        ui = idx[node]
        best = float('inf')
        for vi in succ[ui]:
            best = min(best, dur[ui] + minRemDur[vi])
        if best < float('inf'):
            minRemDur[ui] = best

    # 4. Labels: at each i, list of (d, r, pred_u, pred_label_idx), sorted by r
    labels = [[] for _ in range(N)]
    labels[idx[-2]] = [(0.0, 0, -1, -1)]  # source label

    # Track best sink label
    best_sink = ( -1.0, -1, None, None )  # (d, r, pred_u, pred_label_idx)

    # 5. Topo sweep
    for node in topo:
        ui = idx[node]
        for label_idx, (d_u, r_u, pred_u, pred_lab) in enumerate(labels[ui]):
            # 5a. Duration prune
            if r_u + minRemDur[ui] > R:
                continue

            # 5b. Tail-charge at u
            pi_u  = duals.get(f"service_{node}", 0.0) if node not in (-2,-1) else 0.0
            dur_u = dur[ui]

            # 5c. Relax edges u->v
            for vi in succ[ui]:
                r_v = r_u + dur_u
                if r_v > R:
                    continue
                d_v = d_u + pi_u

                # 5d. If v is sink, update best_sink
                if vi == sink_i:
                    if d_v > best_sink[0]:
                        best_sink = (d_v, r_v, ui, label_idx)
                    continue

                # 5e. Full dominance at v
                L = labels[vi]
                dominated = False
                new_L = []
                for (d_old, r_old, pu, pli) in L:
                    # existing dominates new?
                    if r_old <= r_v and d_old >= d_v:
                        dominated = True
                        new_L.append((d_old, r_old, pu, pli))
                        continue
                    # new dominates old?
                    if r_old >= r_v and d_old <= d_v:
                        continue
                    # otherwise keep old
                    new_L.append((d_old, r_old, pu, pli))

                if dominated:
                    continue

                # Insert new label into new_L sorted by r
                # extract r's for bisect
                r_list = [r for (_, r, _, _) in new_L]
                pos = bisect.bisect_left(r_list, r_v)
                new_L.insert(pos, (d_v, r_v, ui, label_idx))
                labels[vi] = new_L

    # 6. No feasible path?
    if best_sink[0] < 0:
        return None, float('inf')

    # 7. Reconstruct path
    path = []
    cur_i, cur_lab = sink_i, best_sink
    # Always start with the sink
    path.append(-1)

    # Walk backwards until we reach the source
    while True:
        d_cur, r_cur, pred_u, pred_label_idx = cur_lab
        if pred_u == idx[-2]:
            # predecessor is the source: prepend it and stop
            path.append(-2)
            break
        # Otherwise prepend the actual node
        node = topo[pred_u]
        path.append(node)
        # Move to the predecessor label
        cur_lab = labels[pred_u][pred_label_idx]
        cur_i   = pred_u

    # path is built in reverse: [-1, …, -2], so reverse it
    path.reverse()
    # 8. Reduced cost = 1 - total dual
    total_dual = sum(duals.get(f"service_{u}", 0.0)
                     for u in path if u not in (-2, -1))
    reduced_cost = 1.0 - total_dual

    return path, reduced_cost

# topo then dp
def pricing_rcsp_topolabel(graph, duals, service_dict, R=360):
    """
    Resource‐Constrained Shortest Path (longest dual‐sum) in a DAG with tail‐charging.
    Edge (u->v) charges duration of u and dual of u.
    """


    # 1. Topological sort
    topo = list(nx.topological_sort(graph))

    # 2. Initialize labels at each node: (dual_sum, duration, predecessor_label, predecessor_node)
    labels = {v: [] for v in graph.nodes()}
    labels[-2] = [(0.0, 0.0, None, None)]  # At source, no cost, no duration

    # 3. Propagate labels in topological order
    for u in topo:
        for (d_u, r_u, pred_label, pred_node) in labels[u]:
            # For each outgoing edge (u -> v)
            for v in graph.successors(u):
                # Tail‐charging: add u's duration and u's dual
                dur_u = service_dict[u].serv_dur if u not in (-2, -1) else 0
                pi_u  = duals[f"service_{u}"] if u not in (-2, -1) else 0.0

                r_v = r_u + dur_u
                if r_v > R:
                    continue

                d_v = d_u + pi_u

                # Dominance check at v
                dominated = False
                new_labels = []
                for (d_old, r_old, pl, pn) in labels[v]:
                    # If an existing label dominates the new one, skip adding
                    if d_old >= d_v and r_old <= r_v:
                        dominated = True
                        break
                    # If new label dominates old one, drop the old one
                    if d_v >= d_old and r_v <= r_old:
                        continue
                    new_labels.append((d_old, r_old, pl, pn))
                if dominated:
                    continue

                # Append new label with back‐pointer to u
                new_labels.append((d_v, r_v, (d_u, r_u, pred_label, pred_node), u))
                labels[v] = new_labels

    # 4. Extract best label at sink (-1)
    best = max(labels[-1], key=lambda x: x[0], default=None)
    if best is None:
        return None, float('inf')

    d_best, r_best, pred_label, pred_node = best

    # 5. Reconstruct path
    path = []
    cur = best
    node = -1
    while node is not None:
        path.append(node)
        _, _, cur, node = cur  # cur holds predecessor label, node holds predecessor node
    path.reverse()

    # Reduced cost = 1 - total dual_sum
    reduced_cost = 1.0 - d_best
    return path, reduced_cost

# update -> maximize dual sum and then  1 - dual sum is the reduced cost

# //////////////////////////////////////Priority queue 3rd approach////////////////////////////////////////////
import heapq

# latest approacj 07/04 , promising 
# priority cost then duratin , and tent and permanent labels
#  cost = sum of - duals ,  instead of 1- sum of duals
import heapq

def reconstruct_path(label):
    """
    Reconstruct the full path by backtracking through the 'pred' pointers.
    Each label is a tuple: (cost, resource_vector, node, pred).
    Returns the full path as a list of nodes.
    """
    path = []
    while label is not None:
        cost, res_vec, node, pred = label
        path.append(node)
        label = pred
    return list(reversed(path))

def dominates(lab1, lab2):
    """
    Returns True if lab1 dominates lab2.
    Each label is of the form: (cost, resource_vector, node, pred).
    lab1 dominates lab2 if:
       - lab1.cost <= lab2.cost, and
       - For every index i, lab1.resource_vector[i] <= lab2.resource_vector[i],
         with at least one strict inequality.
    (A safeguard converts an int resource into a one-element tuple.)
    """
    cost1, res1, node1, _ = lab1
    cost2, res2, node2, _ = lab2
    if not isinstance(res1, tuple):
        res1 = (res1,)
    if not isinstance(res2, tuple):
        res2 = (res2,)
    if cost1 > cost2:
        return False
    all_le = all(res1[i] <= res2[i] for i in range(len(res1)))
    any_lt = any(res1[i] < res2[i] for i in range(len(res1)))
    return all_le
    # return all_le and any_lt

def new_duty_rcsp_tent_perm(graph, dual_values, service_dict, max_resource):
    """
    RCSP algorithm with multiple vectorial labeling using explicit tentative (L) vs.
    permanent (P) label management and iterative refinement, as described in Aneja et al. (1983).
    
    Each label is represented as:
         (cost, resource_vector, node, pred)
    where resource_vector is a tuple (currently (duration,)) and pred is a pointer to the predecessor label.
    
    The algorithm:
      1. Initializes L(source) with (0, (0,), source, None) and sets P(source) = {}.
      2. Uses a priority queue (heap) to extract the best tentative label (lexicographic order: cost then resource_vector).
      3. When a label is popped from the heap, if it is still in L(u) it is finalized (moved to P(u)).
      4. The permanent label is then extended to each successor v, generating new labels that are added to L(v)
         only if they are not dominated (by either tentative or permanent labels already at v).
      5. An iterative refinement step is then performed on all nodes: for each node, any label in L(node) that is dominated
         by any label in P(node) is removed.
      6. The process continues until the heap is empty.
      7. At termination, the best label at the sink (node -1) is selected and its path is reconstructed.
      
    Parameters:
       graph         - A NetworkX DiGraph (with designated source = -2 and sink = -1).
       dual_values   - Dictionary mapping "service_{u}" to dual values.
       service_dict  - Dictionary mapping service numbers to Service objects (with attributes start_time, end_time, serv_dur).
       max_resource  - Maximum allowed resource (e.g. maximum duty duration) as a number.
    
    Returns:
       best_path      - The best path (list of nodes) from source (-2) to sink (-1) if one exists.
       best_cost      - The cost of that path.
       tentative      - Dictionary of tentative labels per node.
       permanent      - Dictionary of permanent labels per node.
    """
    source = -2
    sink = -1
    # //////////////////////////////  extrra check for prunning//////////////////////////////
    
    # Enforce tail charging resource values on edges using the "weight" attribute.
    # For an edge (x, y):
    # - If x == source, set weight = 0.
    # - If y == sink, set weight = service_dict[x].serv_dur (if available).
    # - Otherwise, set weight = service_dict[x].serv_dur.
    for x, y in list(graph.edges()):
        if x == source:
            graph[x][y]['weight'] = 0
        elif y == sink:
            try:
                graph[x][y]['weight'] = service_dict[x].serv_dur
            except KeyError:
                graph[x][y]['weight'] = 0
        else:
            try:
                graph[x][y]['weight'] = service_dict[x].serv_dur
            except KeyError:
                graph[x][y]['weight'] = 0
    
    # Precompute the lower bound on resource from any node to the sink.
    # Here we use the "weight" attribute for the reversed graph.
    try:
        rev_graph = graph.reverse(copy=True)
        g_comp_res = nx.single_source_dijkstra_path_length(rev_graph, sink, weight="weight")
    except Exception as e:
        print("Error computing g_comp_res:", e)
        g_comp_res = {}
    # Initialize tentative set at source: store resource as tuple (0,)
    init_label = (0, (0,), source, None)
    tentative = {source: [init_label]}
    permanent = {source: []}
    
    # Priority queue (heap) contains tentative labels.
    heap = [init_label]
    
    while heap:
        # Pop the best tentative label.
        label = heapq.heappop(heap)  # label: (cost, res_vec, u, pred)
        cost, res_vec, u, pred = label
        # If this label is no longer in L(u), skip it.
        if u not in tentative or label not in tentative[u]:
            continue
        
        # Finalize the label: remove it from tentative and add to permanent.
        tentative[u].remove(label)
        permanent.setdefault(u, []).append(label)
        
        # Extend this finalized label to every successor v.
        for v in graph.successors(u):
            # Compute transition time if both u and v are real service nodes.
            # trans_time = 0
            # if u not in [source, sink] and v not in [source, sink]:
            #     trans_time = max(0, service_dict[v].start_time - service_dict[u].end_time)
            # Compute additional resource: service duration at v.
            # add_dur = service_dict[v].serv_dur if v not in [source, sink] else 0
            add_dur = service_dict[u].serv_dur if u not in [source, sink] else 0
            new_duration = res_vec[0] +add_dur
            # new_duration = res_vec[0] + trans_time + add_dur
            # Look-ahead: check that even with an optimistic lower bound from v, total resource doesn't exceed max_resource.
            if v in g_comp_res and new_duration + g_comp_res[v] > max_resource:
                continue

            if new_duration > max_resource:
                continue  # Skip extension if resource constraint is violated.
            new_res_vec = (new_duration,)
            
            # Update cost: subtract dual value for u if applicable.
            if u != source:
                # dual_value = dual_values.get(f"service_{u}", 0)
                dual_value = dual_values.get(f"service_{u}", 0)
                add_cost = -(dual_value)
            else:
                add_cost= 0
            new_cost = cost + add_cost
            
            # Create new label with predecessor pointer.
            new_label = (new_cost, new_res_vec, v, label)
            
            # Dominance check at node v against tentative labels.
            dominated = False
            non_dominated = []
            for lab in tentative.get(v, []):
                if (lab[0] <= new_cost and lab[1] <= new_res_vec):
                    dominated = True
                    break
                if not ((new_cost <= lab[0] and new_res_vec < lab[1])):
                    non_dominated.append(lab)
            if dominated:
                continue
            # Also check against permanent labels at v.
            for lab in permanent.get(v, []):
                if (lab[0] < new_cost) or (lab[0] == new_cost and lab[1] <= new_res_vec):
                    dominated = True
                    break
            if dominated:
                continue
            
            # If not dominated, update tentative set for v.
            tentative.setdefault(v, [])
            tentative[v] = non_dominated + [new_label]
            heapq.heappush(heap, new_label)
        
        # # Iterative Refinement: for every node, remove from tentative any label dominated by a permanent label.
        # for node in list(tentative.keys()):
        #     refined = []
        #     for lab in tentative[node]:
        #         if not any(dominates(perm_lab, lab) for perm_lab in permanent.get(node, [])):
        #             refined.append(lab)
        #     tentative[node] = refined
    
    # End loop: Gather all labels at sink (both tentative and permanent).
    sink_labels = tentative.get(sink, []) + permanent.get(sink, [])
    if sink_labels:
        best_label = min(sink_labels, key=lambda lab: (lab[0], lab[1]))
        best_path = reconstruct_path(best_label)
        # The accumulated dual sum is -best_label[0].
        best_cost = 1 - (-best_label[0]) 
        # best_cost = best_label[0]
        return best_path, best_cost, tentative, permanent
    else:
        return None, None, tentative, permanent

# Example usage:
# Suppose 'graph' is a NetworkX DiGraph with source -2 and sink -1,
# 'duals' is a dictionary mapping "service_{u}" to dual values,
# 'service_dict' maps service numbers to Service objects,
# and max_duration is a number (e.g., 360 minutes).
#
# best_path, best_cost, tentative_labels, permanent_labels = RCSP_tentative_permanent(graph, duals, service_dict, max_duration)



# ////////////////////////////////////////////////////////////////////////////////////////////////
# ///  label + path
def updated_new_duty_with_RCSP_priority(graph, dual_values, service_dict, max_resource):

    """
    Finds a new duty (path from source -2 to sink -1) using an RCSP algorithm with a 
    refined label management scheme (tentative vs. permanent labels) via a priority queue.
    Label tuple: (cost, current_node, resource, path) where path is stored as a tuple.
    
    Parameters:
      graph         - A NetworkX DiGraph.
      dual_values   - Dictionary of dual values (e.g., {"service_1": value, ...}).
      service_dict  - Dictionary mapping service numbers to Service objects.
      max_resource  - Maximum allowed resource (e.g., maximum duty duration in minutes).
    
    Returns:
      best_path     - The best path (list of nodes) from source (-2) to sink (-1).
      best_cost     - The associated cost (reduced cost) of that path.
      final_labels  - Dictionary with keys for each node containing all labels (tentative and permanent).
    """
    # For each node, maintain two sets:
    # tentative_labels[node] : labels that are open for extension.
    # permanent_labels[node] : labels that are finalized.
    tentative_labels = {}
    permanent_labels = {}
    final_labels = {}  # For debugging: union of tentative and permanent labels
    
    # Initialize for source (we denote source as -2, sink as -1)
    source = -2
    init_label = (0, 0, source, (source,))  # store path as a tuple
    tentative_labels[source] = [init_label]
    permanent_labels[source] = []  # no permanent label yet
    final_labels[source] = [init_label]
    
    # Initialize the priority queue with the initial label.
    heap = [init_label]  # Heap items are tuples: (cost,resource, node,  path)
    
    while heap:
        # Pop the label with the lowest cost (i.e. best priority)
        cost,current_resource,u, current_path = heapq.heappop(heap)
        
        # Check if this label is already permanent at u.
        is_permanent = False
        for lab in permanent_labels.get(u, []):
            if lab == (cost, current_resource, u,current_path):
                is_permanent = True
                break
        if not is_permanent:
            # Finalize the label: move it from tentative to permanent.
            permanent_labels.setdefault(u, []).append((cost, current_resource, u,current_path))
            # Remove it from tentative_labels[u] if present.
            if u in tentative_labels:
                tentative_labels[u] = [lab for lab in tentative_labels[u] if lab != (cost, current_resource, u,current_path)]
        
        # Extend this permanent label from node u to each successor v.
        for v in graph.successors(u):
            # Compute transition time only if both nodes are real services.
            transition_time = 0
            if u not in [-2, -1] and v not in [-2, -1]:
                transition_time = max(0, service_dict[v].start_time - service_dict[u].end_time)
            
            # Additional resource consumption is the service duration at v.
            additional_duration = service_dict[v].serv_dur if v not in [-2, -1] else 0
            # new_resource = current_resource + transition_time + additional_duration
            new_resource = current_resource + additional_duration
            if new_resource > max_resource:
                continue  # Skip if this extension violates resource constraint.
            
            # Update cost: subtract dual value for node u if applicable.
            additional_cost = -dual_values[f"service_{u}"] if u != -2 else 0
            new_cost = cost + additional_cost
            
            # Update path: convert to tuple so the label becomes hashable.
            new_path = current_path + (v,)
            new_label = (new_cost,new_resource,v, new_path)
            
            # Dominance check at node v.
            dominated = False
            non_dominated = []
            for existing in tentative_labels.get(v, []):
                # If an existing label is as good or better in cost and resource, new_label is dominated.
                if existing[0] <= new_cost and existing[1] <= new_resource:
                    dominated = True
                    break
                # If new_label does not dominate an existing label, keep the existing one.
                if not (new_cost <= existing[0] and new_resource <= existing[1]):
                    non_dominated.append(existing)
            if dominated:
                continue
            
            # Also check against permanent labels.
            for existing in permanent_labels.get(v, []):
                if existing[0] <= new_cost and existing[1] <= new_resource:
                    dominated = True
                    break
            if dominated:
                continue
            
            # Update tentative labels at v.
            tentative_labels[v] = non_dominated + [new_label]
            
            # Update final_labels[v] as the union (convert to set to remove duplicates).
            final_labels.setdefault(v, [])
            # Convert each label into a tuple that is fully hashable.
            final_labels[v] = list({new_label} | set(final_labels[v]))
            
            # Push new_label onto the heap.
            heapq.heappush(heap, new_label)
    
    # At termination, combine permanent and tentative labels for the sink (t = -1)
    sink = -1
    if sink in final_labels and final_labels[sink]:
        best_label = min(final_labels[sink], key=lambda lab: lab[0])
        best_path = list(best_label[3])  # convert tuple back to list if needed
        best_cost = best_label[0]
        return best_path, best_cost, final_labels
    else:
        return None, None, final_labels
# ///////////////////////////////////////priority queue approach with pred////////////////////////////////////////////
import heapq

def reconstruct_path(label):
    """
    Reconstructs the full path from a label using its predecessor pointers.
    The label is assumed to be a tuple: (cost, resource, node, pred),
    where pred is either another label or None (for the source).
    Returns the path as a list of nodes.
    """
    path = []
    while label is not None:
        cost, resource, node, pred = label
        path.append(node)
        label = pred
    return list(reversed(path))

def new_duty_with_RCSP_priority_pred(graph, dual_values, service_dict, max_resource):
    """
    Finds a new duty (path from source -2 to sink -1) using a Resource-Constrained
    Shortest Path (RCSP) algorithm with a priority queue and predecessor pointers.
    
    Instead of storing the full path in each label, each label is stored as a tuple:
      (cost, resource, node, pred)
    where 'pred' is a pointer to the predecessor label. This is more memory efficient,
    and the full path can be reconstructed later via backtracking.
    
    The label tuple is ordered as (cost, resource, node, pred) so that if two labels have 
    the same cost, the one with the lower resource consumption is processed first.
    
    Parameters:
      graph         - A NetworkX DiGraph.
      dual_values   - Dictionary of dual values (e.g., {"service_1": value, ...}).
      service_dict  - Dictionary mapping service numbers to Service objects.
      max_resource  - Maximum allowed resource (e.g., maximum duty duration in minutes).
    
    Returns:
      best_path     - The best path (list of nodes) from source (-2) to sink (-1).
      best_cost     - The associated cost (reduced cost) of that path.
      labels        - Dictionary of labels at each node (for debugging).
    """
    # Here, we denote source as -2 and sink as -1.
    # Label tuple: (cost, resource, node, pred)
    # For the source, pred is None.
    labels = { -2: [(0, 0, -2, None)] }
    # The heap stores labels. It is ordered lexicographically, so (cost, resource, ...) works as desired.
    heap = [(0, 0, -2, None)]
    while heap:
        cost, resource, u, pred = heapq.heappop(heap)
        # Extend the label from node u to every successor v.
        for v in graph.successors(u):
            # # Compute transition time only if both u and v are service nodes (not source or sink)
            # transition_time = 0
            # if u not in [-2, -1] and v not in [-2, -1]:
            #     transition_time = max(0, service_dict[v].start_time - service_dict[u].end_time)
            
            # Additional resource consumption is the service duration at v (if v is a service)
            additional_duration = service_dict[v].serv_dur if v not in [-2, -1] else 0
            # new_resource = resource + transition_time + additional_duration
            new_resource = resource + additional_duration
            if new_resource > max_resource:
                continue  # Skip if resource limit is exceeded
            
            # Update cost: subtract the dual value for u if u is not the source.
            # if u != -2:
            # service_idx_u = u
            # # dual_u = dual_values[service_idx_u]
            dual_u = dual_values[f"service_{u}"] if u != -2 else 0
            additional_cost = -(dual_u)
            new_cost = cost + additional_cost
            
            # Create a new label. Instead of storing the full path, store a pointer (predecessor) to the current label.
            # We set current_label to the label we just popped.
            current_label = (cost, resource, u, pred)
            new_label = (new_cost, new_resource, v, current_label)
            
            # Dominance check at node v.
            dominated = False
            non_dominated = []
            for existing in labels.get(v, []):
                # Compare by cost and resource.
                if existing[0] <= new_cost and existing[1] <= new_resource:
                    dominated = True
                    break
                if not (new_cost <= existing[0] and new_resource <= existing[1]):
                    non_dominated.append(existing)
            if dominated:
                continue
            
            # Update labels at node v.
            labels.setdefault(v, [])
            labels[v] = non_dominated + [new_label]
            
            # Push the new label onto the heap.
            heapq.heappush(heap, new_label)
    
    # Termination: Check if any labels reached the sink (-1)
    if -1 in labels and labels[-1]:
        best_label = min(labels[-1], key=lambda x: x[0])
        best_path = reconstruct_path(best_label)
        best_cost = best_label[0]
        return best_path, best_cost, labels
    else:
        return None, None, labels

# ///////////////////////////////////////priority queue approach////////////////////////////////////////////
import heapq
# heap of heap

def new_duty_with_RCSP_priority(graph, dual_values, service_dict, max_resource):
    """
    Finds a new duty (path from source -2 to sink -1) using a Resource-Constrained
    Shortest Path (RCSP) algorithm with a priority queue. Uses a simplified label tuple:
    (cost, current_node, resource, path).

    Parameters:
      graph         - NetworkX DiGraph.
      dual_values   - Dictionary of dual values (e.g., {"service_1": value, ...}).
      service_dict  - Dictionary mapping service numbers to Service objects.
      max_resource  - Maximum allowed resource (e.g., maximum duty duration in minutes).

    Returns:
      best_path     - The best path (list of nodes) from source (-2) to sink (-1).
      best_cost     - The associated cost (reduced cost) of that path.
      labels        - Dictionary of labels at each node (for debugging).
    """
    # Initialize labels dictionary with the source node (-2)
    labels = { -2: [(0, -2, 0, [-2])] }  # (cost, node, resource, path)
    
    # Initialize the priority queue with the source label.
    heap = [(0, -2, 0, [-2])]  # (cost, node, resource, path)

    while heap:
        cost, u, current_resource, current_path = heapq.heappop(heap)

        # For each successor of u, try to extend the label.
        for v in graph.successors(u):
            # Calculate transition time between service u and v (if both are actual services).
            transition_time = 0
            if u not in [-2, -1] and v not in [-2, -1]:
                transition_time = max(0, service_dict[v].start_time - service_dict[u].end_time)
            
            # Additional resource consumption is the service duration at v (if applicable).
            additional_duration = service_dict[v].serv_dur if v not in [-2, -1] else 0
            
            # new_resource = current_resource + transition_time + additional_duration
            new_resource = current_resource + additional_duration
            if new_resource > max_resource:
                continue  # Skip if this extension exceeds the allowed resource
            
            # Update cost. Here, we subtract the dual value of the current node (if not source).
            additional_cost = -dual_values.get(f"service_{u}", 0) if u != -2 else 0
            new_cost = cost + additional_cost
            
            new_path = current_path + [v]
            new_label = (new_cost, v, new_resource, new_path)
            
            # Dominance check at node v: remove labels that dominate or are dominated by new_label.
            dominated = False
            non_dominated = []
            for existing in labels.get(v, []):
                # existing: (ex_cost, node, ex_resource, path)
                # If existing label is as good or better in cost and resource, discard new_label.
                if existing[0] <= new_cost and existing[2] <= new_resource:
                    dominated = True
                    break
                # Otherwise, if new_label dominates existing, skip keeping the existing one.
                if not (new_cost <= existing[0] and new_resource <= existing[2]):
                    non_dominated.append(existing)
            if dominated:
                continue
            
            # Update labels at node v.
            labels.setdefault(v, [])
            labels[v] = non_dominated + [new_label]
            
            # Push the new label onto the priority queue.
            heapq.heappush(heap, new_label)
    
    # After processing all labels, check for labels that have reached the sink (-1).
    if -1 in labels:
        best_label = min(labels[-1], key=lambda x: x[0])
        best_path = best_label[3]
        best_cost = best_label[0]
        return best_path, best_cost, labels
    else:
        return None, None, labels

def count_overlaps(selected_duties, services):
    '''
    Checks the number of overlaps of services in selected_duties, and prints them

    Arguments: selected_duties - duties that are selected after column generation
               services - all services

    Returns: Boolean - False, if number of services != all services covered in selected_duties; else True
    '''
    services_covered = {}

    for service in services:
        services_covered[service.serv_num] = 0

    for duty in selected_duties:
        for service in duty:
            services_covered[service] += 1

    num_overlaps = 0
    num_services = 0
    for service in services_covered:
        if services_covered[service] > 1:
            num_overlaps += 1
        if services_covered[service] != 0:
            num_services += 1

    print(f"Number of duties selected: {len(selected_duties)}")
    print(f"Total number of services: {len(services)}")
    print(f"Number of services that overlap in duties: {num_overlaps}")
    print(f"Number of services covered in duties: {num_services}")

    if len(services) != num_services:
        return False
    else:
        return True
    
def solve_MIP(services, duties, threshold=0, cutoff= 100, mipgap = 0.01, timelimit = 600):
    '''
    Solves the RMLP 

    Arguments: services - list of Service objects,
            duties - list of duties
    
    Returns: selected_duties - list of selected duties,
            dual_values - list of dual values for each service,
            selected_duties_vars - list of selected duty variables
            objective_value = objective value of the iteration
    '''
    objective = 0
    model = gp.Model("CrewScheduling_IP")
    model.setParam('OutputFlag', 0)
    
    duty_vars = []
    for i in range(len(duties)):
        duty_vars.append(model.addVar(vtype=GRB.BINARY, name=f"x{i}"))

    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    service_constraints = []
    for service_idx, service in enumerate(services):
        constr = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty) == 1,
            name=f"Service_{service.serv_num}")
        service_constraints.append(constr)

    # model.setParam('MIPGap', mipgap)
    model.setParam('TimeLimit', timelimit)
    model.setParam('MIPFocus', 1)
    model.setParam('Cutoff', cutoff)
    model.optimize()

    # Step 5: Check the solution and retrieve dual values and selected duties
    if model.status == GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
        selected_duties = [i for i, var in enumerate(duty_vars) if var.x > 0.5]
        return model.ObjVal, selected_duties, model
    else:
        return None, None, model
    # ////////////////////////////////////////////////////////////////////////////////

    # //////////////////////////////////////////////////////////////////////////////
# //tent and perm
import heapq


# ////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////// 3rd approach //////////////////////////////////////////
def new_duty_with_label_belmonford(graph, dual_values, service_dict, max_duration):
    """
    Finds a new duty (a path from -2 to -1) using a modified Bellman-Ford algorithm that
    minimizes cost (based on dual values) while ensuring that the cumulative duty duration
    (sum of service durations) is at most max_duration minutes.

    Arguments:
        graph       - Directed graph of services (nodes), with source (-2) and sink (-1)
        dual_values - Dictionary of dual values (e.g., {"service_123": value, ...})
        service_dict- Dictionary mapping service numbers to Service objects (each having serv_dur)
        max_duration- Maximum allowed cumulative duty duration (default is 360 minutes)

    Returns:
        path        - List of service nodes representing the new duty (from -2 to -1)
        best_cost   - Total (reduced) cost of the path
        graph_copy  - Copy of the graph with adjusted edge weights
    """
    import math
    graph_copy = graph.copy()
    
    # Adjust edge weights based on dual values (only for service nodes, not for the source -2)
    for u, v in graph_copy.edges():
        if u != -2:
            dual_u = dual_values.get(f"service_{u}", 0)
            graph_copy[u][v]['weight'] = -(dual_u)
    
    # Initialize labels for each node.
    # Each label is a tuple: (cost, cumulative_duration, predecessor)
    # Start node (-2) has cost 0 and duration 0.
    labels = {node: [] for node in graph_copy.nodes()}
    labels[-2] = [(0, 0, None)]
    
    # Number of nodes in the graph; at most |V|-1 relaxations are needed.
    num_nodes = len(graph_copy.nodes())
    
    # Relaxation loop: iterate up to (num_nodes - 1) times.
    for _ in range(num_nodes - 1):
        updated = False
        # Loop over every edge (u, v) in the graph.
        for u, v in graph_copy.edges():
            # For each label at u, try to relax the edge (u, v)
            for cost_u, dur_u, _ in labels[u]:
                # Compute new cost: add adjusted weight on edge (u,v)
                new_cost = cost_u + graph_copy[u][v]['weight']
                # Compute additional duration for node v: if v is a service (not source/sink), add its duration.
                additional_dur = service_dict[v].serv_dur if v not in [-2, -1] else 0
                new_duration = dur_u + additional_dur
                # Skip if the new duration exceeds the maximum allowed.
                if new_duration > max_duration:
                    continue
                new_label = (new_cost, new_duration, u)
                
                # Check dominance: add the new label if it is not dominated by an existing label at v.
                # A label (c1, d1) dominates (c2, d2) if c1 <= c2 and d1 <= d2.
                dominated = False
                non_dominated = []
                for existing in labels[v]:
                    # If an existing label is as good or better in both cost and duration, skip new label.
                    if existing[0] <= new_cost and existing[1] <= new_duration:
                        dominated = True
                        break
                    # Otherwise, keep labels that are not dominated by new_label.
                    if not (new_cost <= existing[0] and new_duration <= existing[1]):
                        non_dominated.append(existing)
                if dominated:
                    continue
                # Add the new label and remove any labels that are dominated by it.
                non_dominated.append(new_label)
                if len(non_dominated) != len(labels[v]):
                    labels[v] = non_dominated
                    updated = True
                else:
                    # Even if lengths are the same, update if new label not already present.
                    if new_label not in labels[v]:
                        labels[v] = non_dominated
                        updated = True
        # If no label was updated during this full pass, break early.
        if not updated:
            break

    # If no label reaches the sink (-1), then no feasible path exists.
    if not labels[-1]:
        return None, math.inf, graph_copy

    # Choose the label at the sink (-1) with the lowest cost.
    best_label = min(labels[-1], key=lambda x: x[0])
    best_cost = best_label[0]

    # Reconstruct the path from sink (-1) back to source (-2) using predecessor pointers.
    path = []
    current = -1
    curr_label = best_label
    while current is not None:
        path.append(current)
        # Get the predecessor from the label; if current is the source, stop.
        current = curr_label[2]
        if current is not None:
            # Find a label at 'current' that could have led to the label we used.
            # (Here, we take the first matching label; in practice, labels may be unique.)
            for lbl in labels[current]:
                if lbl[0] <= best_cost:  # simple check; we could refine this if needed.
                    curr_label = lbl
                    break
    # Reverse the path so that it goes from source (-2) to sink (-1)
    path.reverse()
    
    return path, best_cost, graph_copy
    
# //////// tried this  but it doesnt modify the number of duties
# def solve_MIP(services, duties, threshold=0, cutoff=100, mipgap=0.01, timelimit=600):
#     """
#     Solves the final IP with an objective that rewards duties covering more services.
    
#     Instead of minimizing the total number of duties, we assign a weight to each duty.
#     For example, weight = 1 / (number of services in the duty). This makes broader duties
#     (covering more services) less expensive, encouraging the IP to select them.
    
#     Returns:
#         model.ObjVal, selected_duties, model
#     """
#     model = gp.Model("CrewScheduling_IP")
#     model.setParam('OutputFlag', 0)
    
#     # Calculate a weight for each duty that rewards merging
#     duty_weights = []
#     for duty in duties:
#         # Ensure duty is non-empty; if empty, assign a large weight
#         weight = 1.0 / len(duty) if len(duty) > 0 else 100.0
#         duty_weights.append(weight)
    
#     duty_vars = []
#     for i in range(len(duties)):
#         duty_vars.append(model.addVar(vtype=GRB.BINARY, name=f"x{i}"))
    
#     # Modified objective: minimize the weighted sum of selected duties.
#     model.setObjective(gp.quicksum(duty_weights[i] * duty_vars[i] for i in range(len(duties))), GRB.MINIMIZE)
    
#     # Each service must be covered exactly once.
#     for service in services:
#         model.addConstr(
#             gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty) == 1,
#             name=f"Service_{service.serv_num}"
#         )
    
#     # Set solver parameters (these remain unchanged)
#     # model.setParam('MIPGap', mipgap)
#     model.setParam('TimeLimit', timelimit)
#     model.setParam('MIPFocus', 1)
#     model.setParam('Cutoff', cutoff)
#     model.optimize()
    
#     if model.status == GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
#         selected_duties = [i for i, var in enumerate(duty_vars) if var.x > 0.5]
#         return model.ObjVal, selected_duties, model
#     else:
#         return None, None, model

# //////////////////////////////////////////////
# def mergeable(duty_i, duty_j, service_dict):
#     """
#     Returns True if duty_i and duty_j can be merged, i.e. if the last "real" service
#     in duty_i can be feasibly followed by the first "real" service in duty_j.
#     The source (-2) and sink (-1) nodes are ignored in the check.
#     Debug print statements are included to trace the merging decision.
#     """
#     # Filter out the source (-2) and sink (-1) nodes.
#     duty_i_filtered = [s for s in duty_i if s not in (-2, -1)]
#     duty_j_filtered = [s for s in duty_j if s not in (-2, -1)]
    
#     print("Merging Debug:")
#     print(f"Original duty_i: {duty_i}, filtered: {duty_i_filtered}")
#     print(f"Original duty_j: {duty_j}, filtered: {duty_j_filtered}")
    
#     # Check that both duties have at least one "real" service.
#     if not duty_i_filtered or not duty_j_filtered:
#         print("One of the duties has no real services. Not mergeable.")
#         return False
    
#     # Identify the last service in duty_i and the first service in duty_j.
#     last_service_num = duty_i_filtered[-1]
#     first_service_num = duty_j_filtered[0]
    
#     # Retrieve the corresponding Service objects.
#     last_service = service_dict[last_service_num]
#     first_service = service_dict[first_service_num]
    
#     print(f"Comparing merge feasibility:")
#     print(f"  Last service in duty_i (filtered): {last_service_num} (Train: {last_service.train_num})")
#     print(f"  First service in duty_j (filtered): {first_service_num} (Train: {first_service.train_num})")
    
#     # Prepare the current duty (list of Service objects) for the feasibility check.
#     duty_i_services = [service_dict[s] for s in duty_i_filtered]
    
#     # Use your can_append logic to check if the first service of duty_j can follow the last service of duty_i.
#     result = can_append(duty_i_services, first_service)
    
#     print(f"Merge feasibility result: {result}")
#     return result



# def solve_MIP(services, duties, service_dict, threshold=0, cutoff=100, mipgap=0.01, timelimit=600):
#     """
#     Final IP model that selects a set of duties (columns) such that each service is covered exactly once.
#     We add merging constraints to discourage the selection of two duties that can be merged into one.
#     """
#     model = gp.Model("CrewScheduling_IP")
#     model.setParam('OutputFlag', 0)
    
#     # Create binary variables for each duty.
#     duty_vars = []
#     for i in range(len(duties)):
#         duty_vars.append(model.addVar(vtype=GRB.BINARY, name=f"x{i}"))
    
#     # Standard objective: minimize the number of selected duties.
#     model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)
    
#     # Coverage constraints: each service must be covered exactly once.
#     for service in services:
#         model.addConstr(
#             gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty) == 1,
#             name=f"Service_{service.serv_num}"
#         )
    
#     # ---- Additional merging constraints ----
#     # For every pair of duties that are mergeable (i.e. could be combined into one longer duty),
#     # force the model not to select both. This encourages the use of longer columns.
#     for i in range(len(duties)):
#         for j in range(i + 1, len(duties)):
#             if mergeable(duties[i], duties[j], service_dict):
#                 model.addConstr(duty_vars[i] + duty_vars[j] <= 1, name=f"merge_{i}_{j}")
    
#     # Set solver parameters.
#     model.setParam('MIPGap', mipgap)
#     model.setParam('TimeLimit', timelimit)
#     model.setParam('MIPFocus', 1)
#     model.setParam('Cutoff', cutoff)
#     model.optimize()
    
#     if model.status == GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
#         selected_duties = [i for i, var in enumerate(duty_vars) if var.x > 0.5]
#         return model.ObjVal, selected_duties, model
#     else:
#         return None, None, model










class DynamicBundleStabilisation:
    def __init__(self, services, alpha = 0.5, max_bundle_size = 10):
        self.services = services
        self.service_indices = {service.serv_num: i for i, service in enumerate(services)}
        self.alpha = alpha
        self.max_bundle_size = max_bundle_size

        # initialising bundle
        self.bundle = []        # list of [dual values, objective values] pairs
        self.stability_center = None
        self.best_objective = float('inf')

        # initialise duals to 0
        self.current_duals = {f"Service_{service.serv_num}": 0 for service in services}

        # proximity parameter for trust region
        self.mu = 1.0

    def get_stabilised_duals(self, duals, objective_value):
        """
        Update the bundle and caluclate stabilised dual values
        Input:
            duals: dictionary of dual values
            objective_value: objective value of the current iteration
        Output:
            stabilised_duals: dictionary of stabilised dual values
        """

        # first call
        if self.stability_center is None:
            self.stability_center = dict(duals)
            self.best_objective = objective_value
            return duals
        

def isolve_MIP(services, duties, incumbent_duties=None, threshold=0, cutoff=100, mipgap=0.01, timelimit=600):
    '''
    Solves the RMLP using a MIP formulation with an optional warm start.
    
    Arguments:
        services - list of Service objects,
        duties - list of duties (each duty is a list of service numbers),
        incumbent_duties - optional list of indices representing a warm start solution,
        threshold - threshold for variable selection (unused in current formulation),
        cutoff - objective cutoff value,
        mipgap - acceptable optimality gap,
        timelimit - time limit for the solver in seconds.
    
    Returns:
        objective_value: the objective value of the solution,
        selected_duties: list of indices for the selected duties,
        model: the Gurobi model object.
    '''
    model = gp.Model("CrewScheduling_IP")
    model.setParam('OutputFlag', 0)
    
    # Create binary decision variables for each duty
    duty_vars = []
    for i in range(len(duties)):
        duty_vars.append(model.addVar(vtype=GRB.BINARY, name=f"x{i}"))
    
     # If an incumbent (warm-start solution) is provided, set the initial values.
    if incumbent_duties is not None:
        for i, var in enumerate(duty_vars):
            # Debug: Report warm-start value assignment.
            if i in incumbent_duties:
                var.start = 1
                print(f"Variable x{i} warm-started to 1.")
            else:
                var.start = 0
                print(f"Variable x{i} warm-started to 0.")
    else:
        print("No incumbent solution provided; no warm-start values set.")
    # Objective: minimize the total number of duties selected
    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)
    
    # For every service, ensure exactly one duty that covers it is selected
    for service in services:
        model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty) == 1,
            name=f"Service_{service.serv_num}"
        )
    
    # Set solver parameters
    model.setParam('MIPGap', mipgap)
    model.setParam('TimeLimit', timelimit)
    model.setParam('MIPFocus', 1)
    model.setParam('Cutoff', cutoff)
    
    model.optimize()
    
    # Check if the solution is optimal or time limit reached
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        selected_duties = [i for i, var in enumerate(duty_vars) if var.x > 0.5]
        return model.ObjVal, selected_duties, model
    else:
        return None, None, model

# ///////////////////////////////////////////////////new heuristic ip////////////////////////////////////////////

from collections import defaultdict

def compute_service_frequency(duties):
    """ computes how many times each service appears across all duties """
    service_count = defaultdict(int)
    for duty in duties:
        for service in duty:
            service_count[service] += 1
    return service_count

def find_critical_duties(duties, service_frequency):
    """ finds duties that contain unique services that no other duty covers """
    critical_duties = set()
    for duty_index, duty in enumerate(duties):
        for service in duty:
            if service_frequency[service] == 1:  # This service appears in only one duty
                critical_duties.add(duty_index)
                break
    return critical_duties

def solve_RMLP_with_hr(services, duties, fixed_duties=set()):
    model = gp.Model("CrewScheduling")
    model.setParam('OutputFlag', 0)
    
    duty_vars = {}
    for i in range(len(duties)):
        if i not in fixed_duties:
            duty_vars[i] = (model.addVar(vtype=GRB.CONTINUOUS, ub=1, lb=0, name=f"x{i}"))
            # duty_vars.append(model.addVar(vtype=GRB.CONTINUOUS, name=f"x{i}", lb=0.3, ub=0.5))
        # else:
        # duty_vars.append(model.addVar(vtype=GRB.CONTINUOUS, ub=1, lb=0, name=f"x{i}"))

    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    service_constraints = []
    for service_idx, service in enumerate(services):
        constr = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty and duty_idx not in fixed_duties) >= 0.95,
            name=f"Service_{service.serv_num}_lower")
        constr2 = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty and duty_idx not in fixed_duties) <= 1.05,
            name=f"Service_{service.serv_num}_upper")

        service_constraints.append(constr)
        service_constraints.append(constr2)

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print('Infeasible problem!')
        return None, None, None, None, model
    elif model.status == GRB.OPTIMAL:
        objective = model.getObjective()
        dual_values = {f"Service_{service.serv_num}": constr.Pi for service, constr in zip(services, service_constraints)}
        primals = {v.varName: v.x for v in model.getVars()}
        selected_duties = [v for v in model.getVars()]
        return selected_duties, dual_values, primals, objective.getValue(), model
    else:
        print("No optimal solution found.")
        return None, None, None, None, model
def branching_heuristic(init_duties2, services, selected_duties=set(), depth=0):
    """ recursive sort of branch-and-bound heuristic for solving the integer program """
    print(f"[Depth {depth}] Solving RMLP with fixed {len(selected_duties)} duties...")
    
    selected_duties_list, dual_values, primals, objective_value, model = solve_RMLP_with_hr(services, init_duties2, selected_duties)
    if selected_duties_list is None:
        print("Infeasible RMLP, backtracking...")
        return selected_duties
    
    service_frequency = compute_service_frequency(init_duties2)
    critical_duties = find_critical_duties(init_duties2, service_frequency)
    selected_duties.update(critical_duties)
    
    high_value_duties = {int(v.varName[1:]) for v in selected_duties_list if v.x >= 0.8}
    selected_duties.update(high_value_duties)
    
    covered_services = set()
    for i in selected_duties:
        covered_services.update(init_duties2[i])
    
    if len(covered_services) == len(services):
        print("All services covered")
        return selected_duties
    
    remaining_duties = set(range(len(init_duties2))) - selected_duties
    best_duty = None
    max_unique_services = 0
    for duty in remaining_duties:
        uncovered_services = set(init_duties2[duty]) - covered_services
        if len(uncovered_services) > max_unique_services:
            max_unique_services = len(uncovered_services)
            best_duty = duty
    
    if best_duty is None:
        print("No more duties available to fix.")
        return selected_duties
    
    selected_duties.add(best_duty)
    print(selected_duties)
    return branching_heuristic(init_duties2, services, selected_duties, depth+1)


# //////////////////////////////////////////////////new ip solver////////////////////////////////////////////
def solve_final_ip(services, duties, mip_gap=0.01, time_limit=600):
    """
    Solves the final IP for the crew scheduling problem using a set partitioning model.
    Each duty (column) is a binary decision variable and each service must be covered exactly once.
    
    Arguments:
        services: List of Service objects (each with a unique serv_num attribute).
        duties:   List of duties (each duty is a list of service numbers representing a column).
        mip_gap:  The MIP gap tolerance (default 0.01 for 1% optimality gap).
        time_limit: Time limit for the solver in seconds.
    
    Returns:
        obj_val:  The optimal objective value (number of duties selected).
        selected: List of indices corresponding to selected duties.
        model:    The Gurobi model (for further inspection if needed).
    """
    import gurobipy as gp
    from gurobipy import GRB

    # Create the model.
    model = gp.Model("FinalIP")
    # Uncomment the following line for more detailed output:
    model.setParam('OutputFlag', 1)
    
    # Create binary decision variables for each duty (column).
    duty_vars = {}
    for i, duty in enumerate(duties):
        duty_vars[i] = model.addVar(vtype=GRB.BINARY, name=f"duty_{i}")
    model.update()

    # Add coverage constraints for each service.
    uncovered_services = []
    for service in services:
        # Collect all duty variables corresponding to duties that cover this service.
        relevant_duties = [duty_vars[i] for i, duty in enumerate(duties) if service.serv_num in duty]
        if not relevant_duties:
            print(f"Warning: Service {service.serv_num} is not covered by any duty!")
            uncovered_services.append(service.serv_num)
        else:
            model.addConstr(gp.quicksum(relevant_duties) == 1, name=f"cover_service_{service.serv_num}")
    if uncovered_services:
        print("Warning: The following services are not covered by any duty:", uncovered_services)
        # You can decide to exit or raise an exception if this is critical.
    
    # Set the objective: minimize the total number of duties selected.
    model.setObjective(gp.quicksum(duty_vars[i] for i in duty_vars), GRB.MINIMIZE)

    # Set solver parameters.
    model.setParam('MIPGap', mip_gap)
    model.setParam('TimeLimit', time_limit)

    # Optimize the model.
    model.optimize()

    # Check if a valid solution is found.
    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        print("Final IP solution not found. Model status:", model.status)
        return None, None, model

    # Extract selected duties (those with value 1).
    selected = [i for i, var in duty_vars.items() if var.X > 0.5]
    obj_val = model.ObjVal

    # # Post-solution check: Verify each service's coverage constraint is satisfied.
    # print("Performing post-solution coverage checks:")
    # for service in services:
    #     constr = model.getConstrByName(f"cover_service_{service.serv_num}")
    #     if constr:
    #         # Compute the left-hand side value of the coverage constraint.
    #         lhs_val = sum(duty_vars[i].X for i, duty in enumerate(duties) if service.serv_num in duty)
    #         if abs(lhs_val - constr.RHS) > 1e-5:
    #             print(f"Check failed for service {service.serv_num}: LHS = {lhs_val}, expected = {constr.RHS}")
    #         else:
    #             print(f"Service {service.serv_num} covered correctly: {lhs_val} = {constr.RHS}")
    return obj_val, selected, model
# /////////////////////////////////////////////////////////////////////////////////////////////////
# ///network redutcion
# import networkx as nx

# import networkx as nx

# def network_reduction(graph, max_resource):
#     """
#     Reduces the network using lower bounds computed from the source and sink.

#     Assumptions:
#       - The graph is a NetworkX DiGraph.
#       - Source node is -2 and sink node is -1.
#       - Each arc (x, y) in the graph has a 'weight' attribute representing the primary cost (e.g., time).
#       - max_resource is the maximum allowed resource (e.g., maximum duty duration).

#     The function performs:
#       1. Compute g(z): shortest path distances from source (-2) to every node.
#       2. Compute g_complement(z): shortest path distances from every node to sink (-1) on the reversed graph.
#       3. For every arc (x, y), if g(x) + weight(x,y) + g_complement(y) > max_resource, the arc is removed.

#     Debugging statements print sample lower bounds and indicate which arcs or nodes are pruned.

#     Returns:
#       A reduced NetworkX DiGraph.
#     """
#     source = -2
#     sink = -1

#     # Compute g: shortest distances from source to all nodes.
#     try:
#         g = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
#         print("Computed g(z) from source.")
#     except Exception as e:
#         print("Error computing g from source:", e)
#         g = {}
    
#     # Compute g_complement: shortest distances from all nodes to sink on the reversed graph.
#     rev_graph = graph.reverse(copy=True)
#     try:
#         g_comp = nx.single_source_dijkstra_path_length(rev_graph, sink, weight="weight")
#         print("Computed g_complement(z) from sink.")
#     except Exception as e:
#         print("Error computing g_complement from sink:", e)
#         g_comp = {}

#     # Debug: Print a few sample g(z) and g_complement(z) values.
#     print("Sample g(z) values (first 5 nodes):", {node: g[node] for node in list(g)[:5]})
#     print("Sample g_complement(z) values (first 5 nodes):", {node: g_comp[node] for node in list(g_comp)[:5]})
    
#     # Iterate over edges; since we might remove edges, make a list copy of the edges.
#     for x, y in list(graph.edges()):
#         if x not in g or y not in g_comp:
#             print(f"Removing edge ({x}, {y}) because either g({x}) or g_complement({y}) is not computed.")
#             graph.remove_edge(x, y)
#             continue
        
#         weight = graph[x][y].get("weight", 0)
#         lower_bound = g[x] + weight + g_comp[y]
#         print(f"Edge ({x}, {y}) with weight {weight}: g({x}) = {g[x]}, g_complement({y}) = {g_comp[y]}, total lower_bound = {lower_bound}")
        
#         if lower_bound > max_resource:
#             print(f"Pruning edge ({x}, {y}) because lower_bound {lower_bound} exceeds max_resource {max_resource}.")
#             graph.remove_edge(x, y)
    
#     # Optionally, remove nodes that have become isolated (with no incoming or outgoing edges) except for source and sink.
#     for node in list(graph.nodes()):
#         if node not in [source, sink]:
#             if graph.in_degree(node) == 0 or graph.out_degree(node) == 0:
#                 print(f"Removing isolated node {node}.")
#                 graph.remove_node(node)
    
#     print("Network reduction complete. Remaining nodes:", graph.number_of_nodes(), "Remaining edges:", graph.number_of_edges())
#     return graph

# # Example usage:
# # Assume 'graph' is your full NetworkX DiGraph, max_resource is defined (e.g., 360 minutes).
# # reduced_graph = network_reduction(graph.copy(), max_resource=360)
# import networkx as nx

# import networkx as nx

import networkx as nx

def network_reduction(graph, service_dict,max_resource):
    """
    Reduces the network using lower bounds computed from the source (-2) and sink (-1),
    while enforcing specific weight assignments on edges:
      - For an edge (x,y) with x and y being service numbers, weight is set to service_dict[x].serv_dur.
      - For an edge (service, -1), weight is set to service_dict[service].serv_dur.
      - For an edge (-2, service), weight is set to 0.
    
    Then, it computes:
      g(z): the shortest path cost from source (-2) to each node.
      g_complement(z): the shortest path cost from each node to sink (-1) (computed on the reversed graph).
    
    For each edge (x,y), if:
          g(x) + weight(x,y) + g_complement(y) > max_resource,
    the edge is pruned from the network.
    
    Debug statements print sample g(z) values (last 5 nodes) and details of pruned arcs.
    
    Parameters:
      graph         - A NetworkX DiGraph representing the crew scheduling network.
      max_resource  - Maximum allowed resource (e.g., maximum duty duration in minutes).
      service_dict  - Dictionary mapping service numbers to Service objects.
    
    Returns:
      A reduced NetworkX DiGraph.
    """
    source = -2
    sink = -1
    
    # Enforce the desired weight assignment on all edges.
    for x, y in list(graph.edges()):
        if x == source:
            # For edge (-2, service), weight is 0.
            graph[x][y]['weight'] = 0
        elif y == sink:
            # For edge (service, -1), weight is service_dict[x].serv_dur.
            try:
                graph[x][y]['weight'] = service_dict[x].serv_dur
            except KeyError:
                print(f"Warning: service {x} not found in service_dict.")
                graph[x][y]['weight'] = 0
        else:
            # For edge (x,y) where x and y are service numbers.
            try:
                graph[x][y]['weight'] = service_dict[x].serv_dur
            except KeyError:
                print(f"Warning: service {x} not found in service_dict.")
                graph[x][y]['weight'] = 0

    # Compute g(z): shortest distances from source to all nodes.
    try:
        g = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        print("Computed g(z) from source.")
    except Exception as e:
        print("Error computing g(z) from source:", e)
        g = {}
    
    # Compute g_complement(z): shortest distances from every node to sink on the reversed graph.
    rev_graph = graph.reverse(copy=True)
    try:
        g_comp = nx.single_source_dijkstra_path_length(rev_graph, sink, weight="weight")
        print("Computed g_complement(z) from sink.")
    except Exception as e:
        print("Error computing g_complement(z) from sink:", e)
        g_comp = {}
    
    # Debug: Print sample g(z) values from the last 5 nodes.
    g_keys = list(g.keys())
    sample_nodes = g_keys[-5:] if len(g_keys) >= 5 else g_keys
    print("Sample g(z) values (last 5 nodes):", {node: g[node] for node in sample_nodes})
    
    # Iterate over edges (make a copy of the edge list because we'll remove edges).
    for x, y in list(graph.edges()):
        if x not in g or y not in g_comp:
            print(f"Removing edge ({x}, {y}) because g({x}) or g_complement({y}) is not computed.")
            graph.remove_edge(x, y)
            continue
        
        weight_xy = graph[x][y].get("weight", 0)
        lower_bound = g[x] + weight_xy + g_comp[y]
        print(f"Edge ({x}, {y}): weight = {weight_xy} | g({x}) = {g[x]}, g_complement({y}) = {g_comp[y]} | lower_bound = {lower_bound}")
        
        if lower_bound > max_resource:
            print(f"Pruning edge ({x}, {y}) because lower_bound {lower_bound} exceeds max_resource {max_resource}.")
            graph.remove_edge(x, y)
    
    # Optionally, remove nodes that become isolated (except source and sink).
    for node in list(graph.nodes()):
        if node not in [source, sink]:
            if graph.in_degree(node) == 0 or graph.out_degree(node) == 0:
                print(f"Removing isolated node {node}.")
                graph.remove_node(node)
    
    print("Network reduction complete. Remaining nodes:", graph.number_of_nodes(), 
          "Remaining edges:", graph.number_of_edges())
    return graph

# Example usage:
# reduced_graph = network_reduction(graph.copy(), max_resource=360, service_dict=service_dict)
