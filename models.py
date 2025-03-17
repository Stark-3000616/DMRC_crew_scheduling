import gurobipy as gp 
from gurobipy import GRB
import networkx as nx
from collections import defaultdict

from helper import Service, hhmm2mins, mins2hhmm, fetch_data, draw_graph_with_edges, node_legal, no_overlap, create_duty_graph, extract_nodes, generate_paths, roster_statistics, get_bad_paths, get_lazy_constraints, solve_RMLP, new_duty_with_bellman_ford

def simple_mpc(graph, service_dict, show_logs = True, show_duties = False, show_roster_stats = False):

    model = gp.Model("MPC")

    incoming_var = defaultdict(list)
    outgoing_var = defaultdict(list) 
    edge_vars = {} #xij - binary

    incoming_adj_list = nx.to_dict_of_lists(graph.reverse())

    #Decision Variables
    for (i,j) in graph.edges():
        edge_vars[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        incoming_var[j].append(edge_vars[i,j])
        outgoing_var[i].append(edge_vars[i,j])


    #Objective 
    model.setObjective(gp.quicksum(edge_vars[i,-1] for i in graph.nodes if i not in [-1, -2]), GRB.MINIMIZE)


    #Constraints - Flow conservation
    flow_constraints = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            constr = model.addConstr(gp.quicksum(incoming_var[i])== 1,name=f"Service_flow_{i}") #gp.quicksum(outgoing_var[i]
            flow_constraints.append(constr)



    cover_constraints = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1,name=f"cover_{i}")
            cover_constraints.append(constr)

    if not show_logs:
        print("\nNumber of decision variables: ", len(edge_vars))
        print("Number of flow constraints: ", len(flow_constraints))
        print("Number of cover constraints: ", len(cover_constraints))
        model.setParam('OutputFlag', 0)
    model.optimize()

    paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)
    if show_roster_stats:
        roster_statistics(paths, service_dict) 
    return paths, len(paths)


def mpc_duration_constr(graph, service_dict, show_logs = True, max_duty_duration=6*60, time_limit = 60, show_duties = False, show_roster_stats = False):

    model = gp.Model("MPC")
    model.setParam('OutputFlag', 0)
    if time_limit:
        model.setParam('TimeLimit', time_limit)

    incoming_var = defaultdict(list)
    outgoing_var = defaultdict(list)
    incoming_relation_var = defaultdict(list) 
    edge_vars = {} #xij - binary
    edge_cumu = {} #zij -continuous

    incoming_adj_list = nx.to_dict_of_lists(graph.reverse())
    # outgoing_adj_list = nx.to_dict_of_lists(graph)

    #Decision Variables
    for (i,j) in graph.edges():
        # print(i,j)
        edge_vars[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        incoming_var[j].append(edge_vars[i,j])
        outgoing_var[i].append(edge_vars[i,j])
        edge_cumu[i,j] = model.addVar(vtype=GRB.CONTINUOUS, lb = 0, name=f"z_{i}_{j}")
        incoming_relation_var[j].append(edge_cumu[i,j])

    path_duration = {}    
    for node in graph.nodes():
        path_duration[node] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{node}")

    # path_duration[-1] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{-1}")
    # path_duration[-2] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{-2}")

    #Objective 
    model.setObjective(gp.quicksum(edge_vars[i,-1] for i in graph.nodes if i not in [-1, -2]), GRB.MINIMIZE)
    # print(incoming_var)

    #Constraints - Flow conservation
    flow_constraints = []
    z_y_relation_constraints = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            constr = model.addConstr(gp.quicksum(incoming_var[i])== 1,name=f"Service_flow_{i}") #gp.quicksum(outgoing_var[i]
            flow_constraints.append(constr)

            #z_y relationship constraints -- new model
            constr2 = model.addConstr(path_duration[i] >= service_dict[i].serv_dur + gp.quicksum(edge_cumu[j,i] for j in incoming_adj_list[i]),name=f"relation_{i}")
            z_y_relation_constraints.append(constr2)

    #z_y relationship constraints; start, end -- new model
    constr2_start = model.addConstr(path_duration[-2] >= 0,name=f"relation_{-2}")
    z_y_relation_constraints.append(constr2_start)

    constr2_end = model.addConstr(path_duration[-1] >= 0 + gp.quicksum(edge_cumu[j,-1] for j in incoming_adj_list[-1]),name=f"relation_{-1}")
    z_y_relation_constraints.append(constr2_end)

    cover_constraints = []
    upper_bound_path_duration = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            #Constraints - Node cover exactly once
            constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1,name=f"cover_{i}")
            cover_constraints.append(constr)


            #Upper bound on path duration - new model
            constr2 = model.addConstr(path_duration[i] <= max_duty_duration)
            upper_bound_path_duration.append(constr2)

    upper_bound_path_duration.append(path_duration[-1] <= max_duty_duration)

    #constraint on z -- new model 
    linearisation = defaultdict(list)
    for (i,j) in graph.edges():
        constr1 = model.addConstr(edge_cumu[i,j] <= max_duty_duration *edge_vars[i,j])
        constr2 = model.addConstr(edge_cumu[i,j] <= path_duration[i])
        # constr3 = model.addConstr(edge_cumu[i,j] >= 0) #implicit
        constr4 = model.addConstr(edge_cumu[i,j] >= path_duration[i] - (max_duty_duration* (1- edge_vars[i,j])))
        linearisation[i,j].append(constr1)
        linearisation[i,j].append(constr2)
        # linearisation[i,j].append(constr3)
        linearisation[i,j].append(constr4)

    
    if show_logs:
        print("Number of decision variables: ", len(edge_vars))
        print("Number of flow constraints: ", len(flow_constraints))
        print("Number of cover constraints: ", len(cover_constraints))
        print("Number of linearisation constraints: ", len(linearisation)*3)
        print("Number of relationship constraints: ", len (z_y_relation_constraints))
        model.setParam('OutputFlag', 0)
    model.optimize()

    paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)
    if show_roster_stats:
        roster_statistics(paths, service_dict)

    return paths, len(paths)
# ///////////////////////////////////////////////////////////////////////
# considering 180 minutes as the maximum continuous work duration and 6 hours as the maximum duty duration
def impc_duration_constr(graph, service_dict, show_logs=True, max_duty_duration=445, time_limit=60, 
                        show_duties=False, show_roster_stats=False):
    model = gp.Model("MPC")
    if time_limit:
        model.setParam('TimeLimit', time_limit)

    # Data structures for decision variables
    incoming_var = defaultdict(list)
    outgoing_var = defaultdict(list)
    incoming_relation_var = defaultdict(list)
    edge_vars = {}   # Binary variables for edge selection
    edge_cumu = {}   # Continuous variables for cumulative (waiting) time along an edge

    incoming_adj_list = nx.to_dict_of_lists(graph.reverse())

    # Precompute edge types using a helper function (or inline logic)
    # "direct": satisfies direct connection conditions
    # "break": satisfies the break conditions (reset work time)
    # "other": fallback case (treated like direct)
    edge_type = {}
    for (i, j) in graph.edges():
        if i in service_dict and j in service_dict:
            service_i = service_dict[i]
            service_j = service_dict[j]
            start_end_stn_tf = service_i.end_stn == service_j.start_stn
            start_end_time_tf = 5 <= (service_j.start_time - service_i.end_time) <= 15
            start_end_stn_tf_after_break = service_i.end_stn[:4] == service_j.start_stn[:4]
            start_end_time_within = 50 <= (service_j.start_time - service_i.end_time) <= 150
            if service_i.stepback_train_num == "No StepBack":
                start_end_rake_tf = service_i.train_num == service_j.train_num
            else:
                start_end_rake_tf = service_i.stepback_train_num == service_j.train_num

            if start_end_rake_tf and start_end_stn_tf and start_end_time_tf:
                edge_type[(i, j)] = "direct"
            elif start_end_time_within and start_end_stn_tf_after_break:
                edge_type[(i, j)] = "break"
            else:
                edge_type[(i, j)] = "other"
        else:
            edge_type[(i, j)] = "other"

    # Decision Variables: for every edge, add binary and continuous (edge_cumu) variables
    for (i, j) in graph.edges():
        edge_vars[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        incoming_var[j].append(edge_vars[i, j])
        outgoing_var[i].append(edge_vars[i, j])
        edge_cumu[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{i}_{j}")
        incoming_relation_var[j].append(edge_cumu[i, j])

    # Variables to track overall shift time and active working time
    path_duration = {}
    work_time = {}
    for node in graph.nodes():
        path_duration[node] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{node}")
        work_time[node] = model.addVar(vtype=GRB.CONTINUOUS, name=f"w_{node}")

    # Objective: Minimize the sum of edge variables from nodes to sink (-1)
    model.setObjective(gp.quicksum(edge_vars[i, -1] for i in graph.nodes() if i not in [-1, -2]),
                       GRB.MINIMIZE)

    # Flow Conservation and Path Duration Relations
    flow_constraints = []
    z_y_relation_constraints = []
    for i in graph.nodes():
        if i in [-1, -2]:
            continue
        constr = model.addConstr(gp.quicksum(incoming_var[i]) == 1, name=f"Service_flow_{i}")
        flow_constraints.append(constr)
        constr2 = model.addConstr(
            path_duration[i] >= service_dict[i].serv_dur + gp.quicksum(edge_cumu[j, i] for j in incoming_adj_list[i]),
            name=f"relation_{i}"
        )
        z_y_relation_constraints.append(constr2)

    # Source and sink path_duration settings
    constr2_start = model.addConstr(path_duration[-2] >= 0, name="relation_-2")
    z_y_relation_constraints.append(constr2_start)
    constr2_end = model.addConstr(
        path_duration[-1] >= gp.quicksum(edge_cumu[j, -1] for j in incoming_adj_list[-1]),
        name="relation_-1"
    )
    z_y_relation_constraints.append(constr2_end)

    # Work Time Update: track the cumulative active service time
    model.addConstr(work_time[-2] == 0, name="work_time_start")
    # Use max_duty_duration as our Big-M value.
    for (i, j) in graph.edges():
        # Only apply work time update if both nodes represent actual services.
        if i not in service_dict or j not in service_dict:
            continue
        if edge_type[(i, j)] == "direct":
            model.addConstr(
                work_time[j] >= work_time[i] + service_dict[j].serv_dur - max_duty_duration * (1 - edge_vars[i, j]),
                name=f"work_update_direct_{i}_{j}"
            )
        elif edge_type[(i, j)] == "break":
            model.addConstr(
                work_time[j] >= service_dict[j].serv_dur - max_duty_duration * (1 - edge_vars[i, j]),
                name=f"work_update_break_{i}_{j}"
            )
        else:
            model.addConstr(
                work_time[j] >= work_time[i] + service_dict[j].serv_dur - max_duty_duration * (1 - edge_vars[i, j]),
                name=f"work_update_{i}_{j}"
            )

    # Cover Constraints: ensure each service (except source and sink) is covered exactly once
    cover_constraints = []
    for i in graph.nodes():
        if i in [-1, -2]:
            continue
        constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1, name=f"cover_{i}")
        cover_constraints.append(constr)

    # Linearisation constraints on edge_cumu (z_ij)
    linearisation = defaultdict(list)
    for (i, j) in graph.edges():
        constr1 = model.addConstr(edge_cumu[i, j] <= max_duty_duration * edge_vars[i, j],
                                  name=f"lin1_{i}_{j}")
        constr2 = model.addConstr(edge_cumu[i, j] <= path_duration[i],
                                  name=f"lin2_{i}_{j}")
        constr4 = model.addConstr(edge_cumu[i, j] >= path_duration[i] - (max_duty_duration * (1 - edge_vars[i, j])),
                                  name=f"lin4_{i}_{j}")
        linearisation[i, j].append(constr1)
        linearisation[i, j].append(constr2)
        linearisation[i, j].append(constr4)

    # Upper Bound Constraints for the New Rules:
    model.addConstr(path_duration[-1] <= 445, name="shift_time_limit")
    model.addConstr(work_time[-1] <= 180, name="work_time_limit")

    if not show_logs:
        print("Number of decision variables:", len(edge_vars))
        print("Number of flow constraints:", len(flow_constraints))
        print("Number of cover constraints:", len(cover_constraints))
        print("Number of relation constraints:", len(z_y_relation_constraints))
        model.setParam('OutputFlag', 0)

    model.optimize()

    # Check if the optimization produced a feasible solution
    if model.status != GRB.OPTIMAL:
        print("No optimal solution found. Status:", model.status)
        return [], 0

    # Retrieve solution paths only if a solution exists.
    paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)
    if show_roster_stats:
        roster_statistics(paths, service_dict)

    return paths, len(paths)


def lazy(graph, service_dict, show_logs = True, max_duty_duration=6*60, lazy_iterations =100, show_lazy_updates_every = 10, show_duties = False, show_roster_stats = False):
    model = gp.Model("Lazy")

    incoming_var = defaultdict(list)
    outgoing_var = defaultdict(list) 
    edge_vars = {} #xij - binary

    incoming_adj_list = nx.to_dict_of_lists(graph.reverse())

    #Decision Variables
    for (i,j) in graph.edges():
        edge_vars[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        incoming_var[j].append(edge_vars[i,j])
        outgoing_var[i].append(edge_vars[i,j])


    #Objective 
    model.setObjective(gp.quicksum(edge_vars[i,-1] for i in graph.nodes if i not in [-1, -2]), GRB.MINIMIZE)


    #Constraints - Flow conservation
    flow_constraints = []
    z_y_relation_constraints = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            constr = model.addConstr(gp.quicksum(incoming_var[i])== 1,name=f"Service_flow_{i}") #gp.quicksum(outgoing_var[i]
            flow_constraints.append(constr)



    cover_constraints = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1,name=f"cover_{i}")
            cover_constraints.append(constr)

    if not show_logs:
        print("Number of decision variables: ", len(edge_vars))
        print("Number of flow constraints: ", len(flow_constraints))
        print("Number of cover constraints: ", len(cover_constraints))
        model.setParam('OutputFlag', 0)
    model.optimize()

    print("\n\nInitial Solve Completed!")
    paths, paths_decision_vars = generate_paths(outgoing_var, show_paths = False)
    roster_statistics(paths, service_dict)


    k=1
    lazy_constrs = [] 
    for i in range(lazy_iterations):

        bad_paths, bad_paths_decision_vars = get_bad_paths(paths, paths_decision_vars, service_dict)
        lazy_constraints = get_lazy_constraints(bad_paths, bad_paths_decision_vars, service_dict)

        ###Resolving the model
        if model.Status == GRB.OPTIMAL:

            for lazy_vars in lazy_constraints:
                # print(lazy_vars)
                constr = model.addConstr(gp.quicksum(lazy_vars)<= len(lazy_vars)-1,name=f"lazy_{k}")
                lazy_constrs.append(constr)
                k+=1
            model.reset()
            model.optimize()
            solve_time = model.Runtime

        paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)
        if i in [j for j in range(0,lazy_iterations, show_lazy_updates_every)]:
            print("\nLazy Constraints addition iteration number: ", i)
            print(f"Model solved in {solve_time} seconds.")
            print("Total number of constraints: ", len(model.getConstrs()))
            roster_statistics(paths, service_dict)

    if show_duties:
        paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)

    if show_roster_stats:
        roster_statistics(paths, service_dict)

    return paths, len(paths)


def column_generation(method, graph, services, init_duties, num_iter = 10, threshold = 0):        # Method 1: Bellman Ford, Method 2: Topological sort
    if method == 1:
        objectives = []
        for _ in range(num_iter):
            # print(f"Iteration {_}")
            selected_dooties, dual_values, selected_duties_vars, obj = solve_RMLP(services, init_duties, threshold)
            objectives.append(obj)
            path, length, graph_copy = new_duty_with_bellman_ford(graph, dual_values)
            init_duties.append(path[1:-1])
        indexes = [int(duty_num[1:]) for duty_num in selected_duties_vars]
        selected_duties = [init_duties[i] for i in indexes]
        return objectives, selected_duties, indexes
    elif method == 2:
        pass
    else:
        print("Invalid method. Please choose either 1 or 2.")
        return

    