import gurobipy as gp 
from gurobipy import GRB
import networkx as nx
from collections import defaultdict

from helper import Service, hhmm2mins, mins2hhmm, fetch_data, draw_graph_with_edges, node_legal, no_overlap, create_duty_graph, extract_nodes, generate_paths, roster_statistics, get_bad_paths, get_lazy_constraints, solve_RMLP,pricing_rcsp_dag_full, new_duty_with_bellman_ford,updated_new_duty_with_bellman_ford

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

    # #constraint on z -- new model 
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
# def updated_mpc_duration_constr(graph, service_dict, show_logs=True, max_duty_duration=8*60, time_limit=60, show_duties=False, show_roster_stats=False):

#     model = gp.Model("MPC")
#     model.setParam('OutputFlag', 0)
#     if time_limit:
#         model.setParam('TimeLimit', time_limit)

#     incoming_var = defaultdict(list)
#     outgoing_var = defaultdict(list)
#     incoming_relation_var = defaultdict(list) 
#     edge_vars = {}  # xij - binary
#     edge_cumu = {}  # zij - continuous (pure accumulated service time)

#     incoming_adj_list = nx.to_dict_of_lists(graph.reverse())
#     # Decision Variables: create x and z for each edge.
#     for (i, j) in graph.edges():
#         edge_vars[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
#         incoming_var[j].append(edge_vars[i, j])
#         outgoing_var[i].append(edge_vars[i, j])
#         edge_cumu[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{i}_{j}")
#         incoming_relation_var[j].append(edge_cumu[i, j])

#     # Create cumulative duration (y) variables for each node.
#     path_duration = {}    
#     for node in graph.nodes():
#         path_duration[node] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{node}")

#     # Objective: minimize sum of edges entering the sink (-1)
#     model.setObjective(gp.quicksum(edge_vars[i, -1] for i in graph.nodes() if i not in [-1, -2]), GRB.MINIMIZE)

#     flow_constraints = []
#     z_y_relation_constraints = []
#     # For each service node (excluding source and sink)
#     for i in graph.nodes():
#         if i in [-1, -2]:
#             continue
#         else:
#             # Flow conservation: exactly one incoming edge.
#             constr = model.addConstr(gp.quicksum(incoming_var[i]) == 1, name=f"Service_flow_{i}")
#             flow_constraints.append(constr)
#             # Node relation constraint with explicit gap:
#             # For each incoming edge from j to i, we add the pure accumulated time plus gap.
#             # Now, if j is the source (-2) or i is the sink (-1), we set gap = 0.
#             constr2 = model.addConstr(
#                 path_duration[i] >= service_dict[i].serv_dur + 
#                 gp.quicksum(
#                     edge_cumu[j, i] + (
#                         0 if j == -2 else (service_dict[i].start_time - service_dict[j].end_time)
#                     ) * edge_vars[j, i]
#                     for j in incoming_adj_list[i]
#                 ),
#                 name=f"relation_{i}"
#             )
#             z_y_relation_constraints.append(constr2)

#     # Source and sink relation constraints.
#     constr2_start = model.addConstr(path_duration[-2] >= 0, name="relation_-2")
#     z_y_relation_constraints.append(constr2_start)
#     constr2_end = model.addConstr(path_duration[-1] >= gp.quicksum(edge_cumu[j, -1] for j in incoming_adj_list[-1]),
#                                   name="relation_-1")
#     z_y_relation_constraints.append(constr2_end)

#     cover_constraints = []
#     upper_bound_path_duration = []
#     # Cover constraints: each service node has exactly one outgoing edge.
#     for i in graph.nodes():
#         if i in [-1, -2]:
#             continue
#         else:
#             constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1, name=f"cover_{i}")
#             cover_constraints.append(constr)
#             constr2 = model.addConstr(path_duration[i] <= max_duty_duration, name=f"upper_{i}")
#             upper_bound_path_duration.append(constr2)
#     upper_bound_path_duration.append(model.addConstr(path_duration[-1] <= max_duty_duration, name="upper_-1"))

#     # Linearization constraints for z (pure accumulated service time).
#     # These constraints ensure that if edge (i,j) is used then z(i,j) equals path_duration[i].
#     linearisation = defaultdict(list)
#     for (i, j) in graph.edges():
#         constr1 = model.addConstr(edge_cumu[i, j] <= max_duty_duration * edge_vars[i, j], name=f"lin_ub_{i}_{j}")
#         constr2 = model.addConstr(edge_cumu[i, j] <= path_duration[i], name=f"lin_y_{i}_{j}")
#         constr3 = model.addConstr(edge_cumu[i, j] >= path_duration[i] - max_duty_duration * (1 - edge_vars[i, j]), name=f"lin_lb_{i}_{j}")
#         linearisation[i, j].append(constr1)
#         linearisation[i, j].append(constr2)
#         linearisation[i, j].append(constr3)

#     if show_logs:
#         print("Number of decision variables: ", len(edge_vars))
#         print("Number of flow constraints: ", len(flow_constraints))
#         print("Number of cover constraints: ", len(cover_constraints))
#         print("Number of linearisation constraints: ", len(linearisation) * 3)
#         print("Number of relationship constraints: ", len(z_y_relation_constraints))
#         model.setParam('OutputFlag', 0)
        
#     model.optimize()

#     paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)
#     if show_roster_stats:
#         roster_statistics(paths, service_dict)

#     return paths, len(paths)

# ///////////drive durtion +path duration take 2///////////////////////////////////////////////////////

# def updated_mpc_duration_constr(graph, service_dict, show_logs=True, max_drive_duration=6*60, max_duty_duration=8*60, time_limit=60, show_duties=False, show_roster_stats=False):
#     model = gp.Model("MPC")
#     if time_limit:
#         model.setParam('TimeLimit', time_limit)
#     """
#     Updated model with two types of cumulative duration:
#       - path_duration: total drive duration (sum of service durations only)
#       - duty_duration: overall duty duration (drive time plus gap/wait time between services)
      
#     Parameters:
#       - max_duty_duration: maximum allowed drive time (service durations only)
#       - max_total_duty_duration: maximum allowed overall duty time (drive time + waiting)
#     """
#     model = gp.Model("MPC")
#     if time_limit:
#         model.setParam('TimeLimit', time_limit)

#     # Dictionaries for drive-time tracking (as before)
#     incoming_var = defaultdict(list)
#     outgoing_var = defaultdict(list)
#     incoming_relation_var = defaultdict(list)
#     edge_vars = {}  # binary decision variables for edges
#     edge_cumu = {}  # continuous variable for drive-time propagation (without gap)

#     # Build incoming adjacency list (for drive time)
#     incoming_adj_list = nx.to_dict_of_lists(graph.reverse())

#     # Create decision variables for each edge (drive time part)
#     for (i, j) in graph.edges():
#         edge_vars[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
#         incoming_var[j].append(edge_vars[i, j])
#         outgoing_var[i].append(edge_vars[i, j])
#         edge_cumu[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{i}_{j}")
#         incoming_relation_var[j].append(edge_cumu[i, j])

#     # Create drive-time cumulative variables for each node.
#     path_duration = {}
#     for node in graph.nodes():
#         path_duration[node] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{node}")

#     # OBJECTIVE: minimize the number of duties (edges into sink -1)
#     model.setObjective(gp.quicksum(edge_vars[i, -1] for i in graph.nodes() if i not in [-1, -2]), GRB.MINIMIZE)

#     # CONSTRAINTS for drive time (service durations only)
#     flow_constraints = []
#     z_y_relation_constraints = []
#     for i in graph.nodes():
#         if i in [-1, -2]:
#             continue
#         # Each service node must have exactly one incoming edge.
#         constr = model.addConstr(gp.quicksum(incoming_var[i]) == 1, name=f"Service_flow_{i}")
#         flow_constraints.append(constr)
#         # The drive time at node i is at least its service duration plus the cumulative drive time carried by incoming edges.
#         constr2 = model.addConstr(path_duration[i] >= service_dict[i].serv_dur +
#                                   gp.quicksum(edge_cumu[j, i] for j in incoming_adj_list[i]),
#                                   name=f"relation_{i}")
#         z_y_relation_constraints.append(constr2)

#     # For the source (-2) and sink (-1)
#     constr2_start = model.addConstr(path_duration[-2] >= 0, name="relation_-2")
#     z_y_relation_constraints.append(constr2_start)
#     constr2_end = model.addConstr(path_duration[-1] >= gp.quicksum(edge_cumu[j, -1] for j in incoming_adj_list[-1]),
#                                   name="relation_-1")
#     z_y_relation_constraints.append(constr2_end)

#     # COVER CONSTRAINTS for drive time: each service node has exactly one outgoing edge.
#     cover_constraints = []
#     upper_bound_path_duration = []
#     for i in graph.nodes():
#         if i in [-1, -2]:
#             continue
#         constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1, name=f"cover_{i}")
#         cover_constraints.append(constr)
#         # Upper bound on drive time at node i.
#         constr2 = model.addConstr(path_duration[i] <= max_drive_duration)
#         upper_bound_path_duration.append(constr2)
#     upper_bound_path_duration.append(model.addConstr(path_duration[-1] <= max_drive_duration))

#     # LINEARISATION CONSTRAINTS for drive time propagation (edge_cumu variables)
#     linearisation = defaultdict(list)
#     for (i, j) in graph.edges():
#         constr1 = model.addConstr(edge_cumu[i, j] <= max_drive_duration * edge_vars[i, j])
#         constr2 = model.addConstr(edge_cumu[i, j] <= path_duration[i])
#         constr3 = model.addConstr(edge_cumu[i, j] >= path_duration[i] - (max_drive_duration * (1 - edge_vars[i, j])))
#         linearisation[i, j].append(constr1)
#         linearisation[i, j].append(constr2)
#         linearisation[i, j].append(constr3)

#     # ----- NEW SECTION: Overall Duty Duration (drive + gap/wait time) -----
#     # Create new cumulative variables for overall duty duration at each node.
#     duty_duration = {}
#     for node in graph.nodes():
#         duty_duration[node] = model.addVar(vtype=GRB.CONTINUOUS, name=f"duty_{node}")

#     # Create new edge variables to propagate duty duration (includes gap time).
#     edge_duty = {}
#     for (i, j) in graph.edges():
#         edge_duty[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"ed_{i}_{j}")

#     # Duty relation constraints: for each service node, overall duty time is
#     # its service duration plus the cumulative duty carried in from predecessors.
#     for i in graph.nodes():
#         if i in [-1, -2]:
#             continue
#         model.addConstr(duty_duration[i] >= service_dict[i].serv_dur +
#                         gp.quicksum(edge_duty[j, i] for j in incoming_adj_list[i]),
#                         name=f"duty_relation_{i}")

#     # Set source and sink duty durations.
#     model.addConstr(duty_duration[-2] >= 0, name="duty_relation_-2")
#     model.addConstr(duty_duration[-1] >= gp.quicksum(edge_duty[j, -1] for j in incoming_adj_list[-1]),
#                     name="duty_relation_-1")

#     # Linearisation constraints for duty duration propagation:
#     # For each edge, duty at node i plus the gap to j is carried by edge_duty.
#     # Define gap for edge (i,j): if both are actual services, gap = start_time(j) - end_time(i); else gap = 0.
#     duty_linearisation = defaultdict(list)
#     for (i, j) in graph.edges():
#         if i in service_dict and j in service_dict:
#             gap = service_dict[j].start_time - service_dict[i].end_time
#         else:
#             gap = 0
#         # Use a big-M value for duty duration propagation.
#         M = max_duty_duration  
#         duty_constr1 = model.addConstr(edge_duty[i, j] <= M * edge_vars[i, j])
#         duty_constr2 = model.addConstr(edge_duty[i, j] <= duty_duration[i] + gap)
#         duty_constr3 = model.addConstr(edge_duty[i, j] >= (duty_duration[i] + gap) - M * (1 - edge_vars[i, j]))
#         duty_linearisation[i, j].append(duty_constr1)
#         duty_linearisation[i, j].append(duty_constr2)
#         duty_linearisation[i, j].append(duty_constr3)

#     # Upper bound on overall duty duration at each service node and sink.
#     for i in graph.nodes():
#         if i not in [-1, -2]:
#             model.addConstr(duty_duration[i] <= max_duty_duration, name=f"duty_ub_{i}")
#     model.addConstr(duty_duration[-1] <= max_duty_duration, name="duty_ub_-1")

#     # ----- End New Duty Duration Section -----

#     if not show_logs:
#         print("Number of decision variables: ", len(edge_vars))
#         print("Number of flow constraints: ", len(flow_constraints))
#         print("Number of cover constraints: ", len(cover_constraints))
#         print("Number of drive linearisation constraints: ", len(linearisation) * 3)
#         print("Number of drive relation constraints: ", len(z_y_relation_constraints))
#         print("Number of duty linearisation constraints: ", len(duty_linearisation) * 3)
#         model.setParam('OutputFlag', 0)
#     model.optimize()

#     paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)
#     if show_roster_stats:
#         roster_statistics(paths, service_dict)

#     return paths, len(paths)




# ///////////////////////////////////////////////////////////////////////
# //////////////////////////////drive durtion +path duration take 1/////////////////////////////////////////
def updated_mpc_duration_constr(graph, service_dict, show_logs=True, max_duty_duration=8*60, max_drive_duration=6*60, time_limit=60, show_duties=False, show_roster_stats=False):
    # Create the model and set a time limit if provided.
    model = gp.Model("MPC")
    if time_limit:
        model.setParam('TimeLimit', time_limit)

    # Initialize dictionaries for decision variables.
    incoming_var = defaultdict(list)
    outgoing_var = defaultdict(list)
    incoming_relation_var = defaultdict(list)
    edge_vars = {}  # Binary variables for using an edge (i,j)
    edge_cumu = {}  # Continuous variables used in linearization for drive duration

    # Build incoming adjacency list from the reversed graph.
    incoming_adj_list = nx.to_dict_of_lists(graph.reverse())

    # Decision variables: for each edge (i, j)
    for (i, j) in graph.edges():
        edge_vars[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        incoming_var[j].append(edge_vars[i, j])
        outgoing_var[i].append(edge_vars[i, j])
        edge_cumu[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{i}_{j}")
        incoming_relation_var[j].append(edge_cumu[i, j])

    # Introduce two sets of node variables:
    #   drive_duration: sum of service durations only (the "driving" time)
    #   path_duration: drive_duration plus waiting (gap) times (overall elapsed time).
    drive_duration = {}
    path_duration = {}
    for node in graph.nodes():
        drive_duration[node] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{node}")
        path_duration[node] = model.addVar(vtype=GRB.CONTINUOUS, name=f"t_{node}")

    # Objective: minimize the number of duties (edges reaching the sink, node -1)
    model.setObjective(gp.quicksum(edge_vars[i, -1] for i in graph.nodes() if i not in [-1, -2]), GRB.MINIMIZE)

    # Flow conservation and drive duration relationship.
    flow_constraints = []
    drive_relation_constraints = []
    for i in graph.nodes():
        if i in [-1, -2]:
            continue
        else:
            # Each service node gets exactly one incoming edge.
            constr = model.addConstr(gp.quicksum(incoming_var[i]) == 1, name=f"Service_flow_{i}")
            flow_constraints.append(constr)
            # The drive duration at node i must be at least its own service duration plus
            # the accumulated contribution (via edge_cumu) from incoming edges.
            constr2 = model.addConstr(
                drive_duration[i] >= service_dict[i].serv_dur + gp.quicksum(edge_cumu[j, i] for j in incoming_adj_list[i]),
                name=f"relation_drive_{i}"
            )
            drive_relation_constraints.append(constr2)

    # For the source (-2) and sink (-1) nodes.
    constr_source_drive = model.addConstr(drive_duration[-2] >= 0, name="relation_drive_-2")
    drive_relation_constraints.append(constr_source_drive)
    constr_sink_drive = model.addConstr(
        drive_duration[-1] >= gp.quicksum(edge_cumu[j, -1] for j in incoming_adj_list[-1]),
        name="relation_drive_-1"
    )
    drive_relation_constraints.append(constr_sink_drive)

    # Cover constraints: each service node must have exactly one outgoing edge.
    cover_constraints = []
    drive_upper_bound_constraints = []
    for i in graph.nodes():
        if i in [-1, -2]:
            continue
        else:
            constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1, name=f"cover_{i}")
            cover_constraints.append(constr)
            # Use max_drive_duration as the upper bound for drive duration.
            constr2 = model.addConstr(drive_duration[i] <= max_drive_duration, name=f"drive_bound_{i}")
            drive_upper_bound_constraints.append(constr2)
    drive_upper_bound_constraints.append(model.addConstr(drive_duration[-1] <= max_drive_duration, name="drive_bound_-1"))

    # Linearization constraints for edge cumulative drive durations.
    linearisation = defaultdict(list)
    for (i, j) in graph.edges():
        constr1 = model.addConstr(edge_cumu[i, j] <= max_duty_duration * edge_vars[i, j], name=f"lin1_{i}_{j}")
        constr2 = model.addConstr(edge_cumu[i, j] <= drive_duration[i], name=f"lin2_{i}_{j}")
        constr4 = model.addConstr(edge_cumu[i, j] >= drive_duration[i] - max_duty_duration * (1 - edge_vars[i, j]), name=f"lin4_{i}_{j}")
        linearisation[i, j].append(constr1)
        linearisation[i, j].append(constr2)
        linearisation[i, j].append(constr4)

    # New constraints for path_duration that incorporate waiting (gap) times.
    path_relation_constraints = []
    # Set the path_duration at the source (-2) to 0.
    source_path_constr = model.addConstr(path_duration[-2] == 0, name="path_source")
    path_relation_constraints.append(source_path_constr)
    # For each edge, add a constraint that propagates path_duration.
    # gap_ij is defined as (service_dict[j].start_time - service_dict[i].end_time) for service edges.
    # For edges from the source (-2) or to the sink (-1), we set gap_ij = 0.
    for (i, j) in graph.edges():
        if i == -2 or j == -1:
            gap_ij = 0
        else:
            gap_ij = service_dict[j].start_time - service_dict[i].end_time
        # For service duration of node j, if j is not in service_dict (e.g. sink), then use 0.
        service_dur_j = service_dict[j].serv_dur if j in service_dict else 0
        # Big-M constant for linearization.
        M = max_duty_duration
        constr_path = model.addConstr(
            path_duration[j] >= path_duration[i] + gap_ij + service_dur_j - M * (1 - edge_vars[i, j]),
            name=f"relation_path_{i}_{j}"
        )
        path_relation_constraints.append(constr_path)
    # Enforce an upper bound on path_duration at the sink.
    sink_path_bound = model.addConstr(path_duration[-1] <= max_duty_duration, name="path_bound_sink")
    path_relation_constraints.append(sink_path_bound)
    
    # New: Add an upper bound for path_duration at each intermediate node.
    path_upper_bound_constraints = []
    for i in graph.nodes():
        if i not in [-1, -2]:
            constr_bound = model.addConstr(path_duration[i] <= max_duty_duration, name=f"path_bound_{i}")
            path_upper_bound_constraints.append(constr_bound)

    if not show_logs:
        print("Number of decision variables: ", len(edge_vars))
        print("Number of flow constraints: ", len(flow_constraints))
        print("Number of cover constraints: ", len(cover_constraints))
        print("Number of linearisation constraints: ", len(linearisation) * 3)
        print("Number of drive relation constraints: ", len(drive_relation_constraints))
        print("Number of path relation constraints: ", len(path_relation_constraints))
        print("Number of path upper-bound constraints: ", len(path_upper_bound_constraints))
        model.setParam('OutputFlag', 0)

    model.optimize()

    paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)
    if show_roster_stats:
        roster_statistics(paths, service_dict)

    return paths, len(paths)

# ////////////////////////////////////////////////////////
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

def run_column_generation(services, graph, service_dict, init_duties, 
                                   threshold=-1e-6, max_duration=6*60, max_iter=100000):
    """
    Runs the column generation phase, logging iteration number, LP objective,
    new column reduced cost, generated path, and a message for special termination conditions.

    Parameters:
      services     : List of Service objects.
      graph        : A NetworkX directed graph representing the service network.
      service_dict : Dictionary mapping service numbers to Service objects.
      init_duties  : Initial list of duties (each duty is a list of service numbers).
      threshold    : Convergence threshold for the reduced cost (e.g., -1e-6).
      max_duration : Maximum allowed duty duration (in minutes).
      max_iter     : Maximum number of iterations.
      
    Returns:
      duties_pool  : The updated list of duties (columns) after generation.
      logs         : A list of tuples:
                     (iteration number, LP objective, new duty cost, path, message)
    """
    duties_pool = init_duties.copy()
    logs = []
    print("Number of duties before column generation:", len(duties_pool))
    
    for iter_num in range(max_iter):
        selected_duties, duals, selected_duty_vars, obj = solve_RMLP(services, duties_pool)
        current_max_duration = max_duration

        # path, cost = pricing_rcsp_dag_full(graph, duals, service_dict, current_max_duration)
        path, cost, graphcopy = updated_new_duty_with_bellman_ford(graph, duals, service_dict, current_max_duration)
        # path, cost = pricing_rcsp_topolabel(graph, duals, service_dict, current_max_duration)
        
        # Initialize message as empty; we'll update it if one of the conditions occurs.
        message = ""
        print(f"Iteration {iter_num}: LP Obj = {obj}, New duty cost = {cost}, Path = {path}")
        
        # Check termination conditions and record message
        if path is None:
            message = "No path found."
            logs.append((iter_num, obj, cost, path, message))
            print(message)
            break
        elif path[1:-1] in duties_pool:
            message = "Path already present, skipping."
            logs.append((iter_num, obj, cost, path, message))
            print(message)
            continue
        elif cost >= threshold:
            message = "Column generation converged: No new column with significant negative reduced cost."
            logs.append((iter_num, obj, cost, path, message))
            print(message)
            break
        else:
            logs.append((iter_num, obj, cost, path, message))
        
        # Append the new duty (excluding the artificial source and sink) if valid.
        duties_pool.append(path[1:-1])
    
    return duties_pool, logs
