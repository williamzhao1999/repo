import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import colorama
import json
import pandas as pd
from colorama import Fore, Back, Style
import time
import os
from tqdm import trange
import os 

dir_path = "./data/mini_graph"

if os.path.isdir(dir_path) == False:
    os.makedirs(dir_path) 

np.random.seed(0)

class CTM:
    G = None
    free_flow_speed = -1
    maximum_flow = -1
    maximum_density = -1
    backward_propagation_speed = -1
    length_cell = -1
    paths = None
    rates = None
    num_paths = -1
    cell_indexes = None
    time_step = 0
    A_and_B_matrices_correct = None
    H_matrix_correct = None


    def __init__(self, G, paths, rates, ffs, mf, md, bps, lc, edges_labels, debug = False):
        self.free_flow_speed = ffs
        self.maximum_flow = mf
        self.maximum_density = md
        self.backward_propagation_speed = bps
        self.G = G
        self.paths = paths
        self.rates = rates
        self.num_paths = len(paths)
        self.length_cell = lc
        self.cell_indexes = edges_labels
        self.graph_nodes = list(G.nodes)
        self.time_step = 10 / 3600
        self.debug = debug
        self.A_and_B_matrices_correct = True
        self.H_matrix_correct = True  

        print(Fore.BLUE)
        print('─' * 46 + " Cell Trasmission Model Path Based " + '─' * 46)
        print(Fore.WHITE)
        print("Simulator configuration:")
        print(f"Free flow speed: {self.free_flow_speed} (km/h), Maximum flow: {self.maximum_flow} (veh/h), Maximum density: {self.maximum_density} (veh/km), Backward propagation speed: {self.backward_propagation_speed} (km/h), Length cell: {self.length_cell} (km), Time Step: {self.time_step} (h), {self.time_step*3600}(s)" + Back.RESET)
        print(Back.RESET)
        print("Lambdas rates:" + Fore.CYAN)
        print(self.rates)
        print(Fore.BLUE)
        print(Fore.RESET + "OD Matrix:")

        indexes = self.graph_nodes
        indexes2 = list(self.graph_nodes)

        for index in indexes:
            if index[0] != "O":
                indexes2.remove(index)

        df = pd.DataFrame(0, index=indexes2, columns=self.graph_nodes)
        list_od = self.getFirstCellAndLastCellInPath(self.paths)
        
        for idx, od in enumerate(list_od):
            df.loc[od[0], od[1]] += self.rates[idx]
        print(df)
        print("")

        print('─' * 128)
        print(Style.RESET_ALL)

        #for start_node, end_node, attributes in G.edges.data():
        #    print(f"{start_node, end_node}, {self.getPathIndexByEdge((start_node, end_node))}")
    def getFirstCellAndLastCellInPath(self, paths):
        list_paths = []
        for p in paths:
            path = (p[0][0], p[-1][-1])
            list_paths.append(path)

        return list_paths
    # Get the path index by edge
    def getPathIndexByEdge(self, edge):

        path_indexes = []
        e = (edge[0], edge[1])
        for idx, path_edges in enumerate(self.paths):
            for edge2 in path_edges:
                if edge2 == e:
                    path_indexes.append(idx)
        return path_indexes

    def update(self, t):
        # Update Demand, Supply, Density of cell i
        ################################################################################################################
        for start_node, end_node, attributes in self.G.edges.data():
            i = self.cell_indexes[(start_node, end_node)]
            path_indexes = self.getPathIndexByEdge((start_node, end_node))

            for path_index in path_indexes:    
                # ts is 10 seconds (time steps)
                self.ro[t+1][i][path_index] = self.ro[t][i][path_index] + (self.time_step / self.length_cell) * (self.inflow[t][i][path_index] - self.outflow[t][i][path_index])

            self.ro_signed[t+1][i] = min(self.maximum_density, self.ro[t+1][i].sum())
            self.inflow_signed[t][i] = self.inflow[t][i].sum()
            self.outflow_signed[t][i] = self.outflow[t][i].sum()
            
            # The demand of cell i for all paths
            self.demand[t][i] = min(self.free_flow_speed*self.ro_signed[t][i], self.maximum_flow)

            # supply of cell i for all paths
            self.supply[t][i] = min(self.maximum_flow, self.backward_propagation_speed*(self.maximum_density - self.ro_signed[t][i]))
            ################################################################################################################

        
        for i in range(len(self.cell_indexes)):
            self.H[t][i][i*self.num_paths:(i*self.num_paths)+self.num_paths] = 1 


    
            

        # H trovato, è la matrice che contiene 1 se la densità del percorso della cella 1 interessa alla cella 1
        if self.debug:
            #print( (self.H[t] @ self.ro[t].flatten()[:, np.newaxis]).flatten())
            #print(self.ro_signed[t]) 
            print(f"Timestamp: {t-1}")
            print(self.ro_signed[t-1])
            #print(self.A[t-1][0:5][0:5])
            print(f"Timestamp: {t}")
            print(self.ro_signed[t])

            
            #print(self.A[t][0:5][0:5])
            time.sleep(0.25)
            os.system('cls')
        
        

            
            
            #a = 10
    def getCellIndex(self, neighbor, start_node, end_node):
        if (neighbor, start_node) in self.cell_indexes:
            return self.cell_indexes[(neighbor, start_node)]
        else:
            return self.cell_indexes[(neighbor, end_node)]

    def getIntersectedPaths(self, ps1, ps2):
        return [x for x in ps1 if x in ps2]

    def update_matrices(self, t):
        if t == 0:
            return True
        '''
        # x_t+1 = A_t*x_t + B_t*u_t
        # x è un vettore di dimensione numero Links L per numero di percorso m, Lm x 1
        # A è la matrice di dimensione Lm x Lm
        # B è la matrice di dimensione Lm x m
        # u è il vettore di dimensione m x 1
        # per cui mi torna un vettore (Lm x Lm) * (Lm x 1) + (Lm x m) * (m x 1) = (Lm x 1)
        # per semplicità di notazione, definisco W = B_t*u_t, quindi x_t+1 = A_t*x_t + W
        # ora costruisco la matrice diagonale A, dove ogni elemento della diagonale i-esima 
        # è uguale alla densità del percorso passante per la cella i al tempo meno la densità entrante al tempo t in quel percorso
        # tutto diviso per la densità del percorso della cella i al tempo precedente
        '''
        '''
        for i in range(len(self.cell_indexes)):
            for j in range(self.num_paths):
                row_index = (i*self.num_paths)+j
                column_index = (i*self.num_paths) + j
                if self.ro[t - 1][i][j] > 0:
                    w = self.B[t - 1][row_index][j] * self.u[t - 1][j]
                    self.A[t - 1][row_index][column_index] = (self.ro[t][i][j] - w)/self.ro[t - 1][i][j]
        '''
        # X[T] = A[T - 1] * X[T - 1] + B[T - 1] * U[T - 1]
        # W = B[T - 1] * U[T - 1]
        # X[T] = A[T - 1] * X[T - 1] + W
        # X[T] - W = A[T - 1] * X[T - 1]
        # A[T - 1] = (X[T] - W) * X[T - 1]^-1
        # A[T - 1] = (Y) * X[T - 1]^-1
        # X[T - 1]^-1 = pseudo inverse of X[T] np.linalg.pinv
        
        #if t == 359:
        #    a = 10
        
        W = np.matmul(self.B[t - 1], self.u[t - 1]) # (456, 24) x (24, 1) = (456,1)

        
        X_T = self.ro[t].flatten()
        Y = X_T - W
        X_T1 = self.ro[t - 1].flatten()
        X_T1 = X_T1[:, np.newaxis]
        X_T1_INVERSE = np.linalg.pinv(X_T1)
        self.A[t - 1] = np.matmul(Y[:, np.newaxis], X_T1_INVERSE)
        W = self.B[t - 1] @ self.u[t - 1]
        ro_previous_flatten = self.ro[t - 1].flatten()[:, np.newaxis] #(456,1)
        ro_calculated_at_time_t = np.matmul(self.A[t - 1] , ro_previous_flatten) #(456,456) x (456,1) = (456,)
        ro_calculated_at_time_t = ro_calculated_at_time_t.flatten() # (456,1)
        
        #W = np.matmul(self.B[t], self.u[t]) # (456, 24) x (24, 1) = (456,1)
        ro_calculated_at_time_t = ro_calculated_at_time_t + W

        ro_flatten = self.ro[t].flatten()
        truth_table_x = np.isclose(ro_calculated_at_time_t, ro_flatten)

        y_derived_ro_signed = (self.H[t] @ ro_calculated_at_time_t).flatten()
        truth_table_y = np.isclose(self.ro_signed[t], y_derived_ro_signed)
        #print( (self.H[t] @ self.ro[t].flatten()[:, np.newaxis]).flatten())
        #print(self.ro_signed[t])

        
        if self.debug:
            print(ro_calculated_at_time_t) # (456, 1)
            print(ro_flatten)  
            
            print(truth_table_x)
            print(self.B[t])
        
        if all(truth_table_x):
            self.A_and_B_matrices_correct = self.A_and_B_matrices_correct & True 
            
        else:
            self.A_and_B_matrices_correct = False
        
        if all(truth_table_y):
            self.H_matrix_correct = self.H_matrix_correct & True 
        else:
            self.H_matrix_correct = False
                
    def generateData(self, noObservations, initialState):
        state = np.zeros((noObservations + 1, len(self.cell_indexes) * self.num_paths))
        observation = np.zeros((noObservations, len(self.cell_indexes)))
        state[0] = initialState
        
        for t in range(1, noObservations):
            state[t] = np.matmul(self.A[t - 1], state[t - 1].T) + np.matmul(self.B[t - 1],self.u[t - 1])
            observation[t] = np.matmul(self.H[t], state[t])

        return(state, observation)

    def simulate(self, hours):

        T = round(hours / self.time_step)

        print(f"Simulation started for time period: {hours} hours, Time step: {self.time_step}(hours) or {self.time_step*3600}(s), Total time steps: {T} steps")
        print("")

        G = self.G
        # traffic enter per path
        self.u = np.zeros(shape=(T+1,self.num_paths)) 

        # density for each path p passing through cell i
        self.ro = np.zeros(shape=(T+1,len(self.cell_indexes),self.num_paths)) 

        # demand of cell i for all paths passing through cell i
        self.demand = np.zeros(shape=(T+1,len(self.cell_indexes))) 

        # demand of cell j that requests to enter cell i at time t
        self.demand_matrix = np.zeros(shape=(T+1,len(self.cell_indexes), len(self.cell_indexes))) 

        # supply of cell i for all paths passing through cell i
        self.supply = np.zeros(shape=(T+1,len(self.cell_indexes))) 

        # inflow per path passing through cell i
        self.inflow = np.zeros(shape=(T+1,len(self.cell_indexes),self.num_paths)) 

        # outflow per path passing through cell i
        self.outflow = np.zeros(shape=(T+1,len(self.cell_indexes),self.num_paths)) 

        # density of all paths passing through cell i
        self.ro_signed = np.zeros(shape=(T+1,len(self.cell_indexes))) 

        # inflow for all paths passing through cell i
        self.inflow_signed = np.zeros(shape=(T+1,len(self.cell_indexes))) 

        # outflow for all paths passing through cell i
        self.outflow_signed = np.zeros(shape=(T+1,len(self.cell_indexes))) 

        # Matrices for dinamic system
        # x_t+1 = A * x_t + B * u_t + episolon_t
        # y_t = H * x_t + delta_t
        self.A = np.zeros(shape=(T, len(self.cell_indexes) * self.num_paths, len(self.cell_indexes) * self.num_paths))
        self.B = np.zeros(shape=(T, len(self.cell_indexes)* self.num_paths, self.num_paths))
        self.H = np.zeros(shape=(T, len(self.cell_indexes),  len(self.cell_indexes) *  self.num_paths))

        for t in trange(T):
            for start_node, end_node, attributes in G.edges.data():

                path_indexes = self.getPathIndexByEdge((start_node, end_node))
                # I use start node of the link as ref to cell index
                i = self.cell_indexes[(start_node, end_node)]
               
                '''
                if attributes['type_upstream_connection'] == 'ordinary':
                    neighbor = next(iter(G.predecessors(start_node)))
                    neighbor_path_indexes = self.getPathIndexByEdge((neighbor, start_node))
                    path_indexes_intersected = self.getIntersectedPaths(path_indexes, neighbor_path_indexes)

                    j = self.cell_indexes[(neighbor, start_node)]

                    sum_neighbor_density = self.ro[t][j][neighbor_path_indexes].sum()
                    self.demand[t][j] = min(self.free_flow_speed*self.ro_signed[t][j], self.maximum_flow)
                    self.supply[t][i] = min(self.maximum_flow, self.backward_propagation_speed*(self.maximum_density - self.ro_signed[t][i]))
                    
                    if sum_neighbor_density > 0:
                        for path_index in path_indexes_intersected:
                            self.inflow[t][i][path_index] = (self.ro[t][j][path_index] / sum_neighbor_density) *  min(self.demand[t][j],self.supply[t][i])
                '''
                if attributes['type_upstream_connection'] == 'entering':

                    self.supply[t][i] = min(self.maximum_flow, self.backward_propagation_speed*(self.maximum_density - self.ro_signed[t][i]))
                    for path_index in path_indexes:
                        self.u[t][path_index] = np.random.poisson(self.rates[path_index], 1)[0]
                        row_index = (i*self.num_paths)+path_index
                        '''
                        la matrice B è di dimensione Lm x m
                        u è il vettore di flusso entrante m x 1
                        la matrice B è quasi diagonale, dove ogni riga ha il valore ts/li per trasformare il flusso entrante in una densità
                        questo valore corrisponde alla densità del percorso a cui entra il flusso passante per la cella
                        ad esempio cella 3 ha un percorso con index 11 ((12) Dashed Blue - 1 perché parte da 0 l'index di un array)
                        quindi ro_t(3,11), vado ad assegnare il valore nella matrice B nella riga ((3-1)+11) e nella colonna 11 
                        '''
                        self.B[t][row_index][path_index] = (self.time_step/self.length_cell)
                    
                    sum_u = self.u[t][path_indexes].sum()
                    for path_index in path_indexes:
                        self.inflow[t][i][path_index] = self.u[t][path_index]/sum_u * min(sum_u, self.supply[t][i])
                    

                elif attributes['type_upstream_connection'] == 'merging' or attributes['type_upstream_connection'] == 'ordinary':

                    self.supply[t][i] = min(self.maximum_flow, self.backward_propagation_speed*(self.maximum_density - self.ro_signed[t][i]))
                    neighbors = list(G.predecessors(start_node))
                    
                    for neighbor in neighbors:
                        j = self.cell_indexes[(neighbor, start_node)]
                        neighbor_path_indexes = self.getPathIndexByEdge((neighbor, start_node))
                        path_indexes_intersected = self.getIntersectedPaths(path_indexes, neighbor_path_indexes)

                        self.demand[t][j] = min(self.free_flow_speed*self.ro_signed[t][j], self.maximum_flow)
                        if self.ro_signed[t][j] > 0:
                            self.demand_matrix[t][j][i] = (self.ro[t][j][path_indexes_intersected].sum()/self.ro[t][j].sum()) * self.demand[t][j]
                    
                        # verify demand j
                        #successors = list(G.successors(start_node))
                        #verify_demand_j = 0
                        #for successor in successors:

                        #    i_prime = self.cell_indexes[(start_node, successor)]
                            
                        #    verify_demand_j += self.demand_matrix[t][j][i_prime]
                        
                        #if abs(verify_demand_j - self.demand[t][j]) >= 10e-9:
                        #    print(self.ro_signed[t][j])
                        #    print(f"Equation not correct, verify demand j: {verify_demand_j}, {self.demand[t][j]}")

                    sum_demand_j_i = 0
                    for neighbor in neighbors:
                        j = self.cell_indexes[(neighbor, start_node)]
                        sum_demand_j_i += self.demand_matrix[t][j][i]

                    min_value = 1
                    if(sum_demand_j_i > 0):
                        min_value = min(1, self.supply[t][i]/sum_demand_j_i)
                    
                    for neighbor in neighbors:
                        
                        j = self.cell_indexes[(neighbor, start_node)]
                        self.outflow_signed[t][j] = 0
                        
                        neighbor_path_indexes = self.getPathIndexByEdge((neighbor, start_node))
                        path_indexes_intersected = self.getIntersectedPaths(path_indexes, neighbor_path_indexes)

                        sum_ro_j_b = self.ro[t][j][neighbor_path_indexes].sum()
                        #sum_ro_j_b = self.ro[t][j][path_indexes_intersected].sum()
                        
                        if sum_ro_j_b > 0:
                            for p in path_indexes_intersected:
                                # Equantion 13
                                #self.outflow[t][j][p] = (self.ro[t][j][p] / sum_ro_j_b) * self.demand_matrix[t][j][i] * min_value
                                #self.outflow[t][j][p] = (self.ro[t][j][p] / sum_ro_j_b) * self.demand[t][j] * min_value
                                self.inflow[t][i][p] = (self.ro[t][j][p] / sum_ro_j_b) * self.demand[t][j] * min_value
                                self.outflow_signed[t][j] += self.outflow[t][j][p]
                        
                    '''
                    self.inflow_signed[t][i] = 0
                    for neighbor in neighbors:
                        j = self.cell_indexes[(neighbor, start_node)]
                        self.inflow_signed[t][i] += self.outflow_signed[t][j]
                    
                    # Check equation if it is correct
                    if abs(self.inflow[t][i].sum()- self.inflow_signed[t][i]) >= 10e-9:
                        print(f"219: {abs(self.inflow[t][i].sum()- self.inflow_signed[t][i])}, {self.inflow[t][i].sum()}, {self.inflow_signed[t][i]}")
                    '''
                '''      
                if attributes['type_downstream_connection'] == 'ordinary':
                    successor = next(iter(G.successors(end_node)))
                    successor_path_indexes = self.getPathIndexByEdge((end_node, successor))
                    path_indexes_intersected = self.getIntersectedPaths(path_indexes, successor_path_indexes)

                    k = self.cell_indexes[(end_node, successor)]

                    self.supply[t][k] = min(self.maximum_flow, self.backward_propagation_speed*(self.maximum_density - self.ro_signed[t][k]))
                    self.demand[t][i] = min(self.free_flow_speed*self.ro_signed[t][i], self.maximum_flow)

                    if self.ro[t][i][path_indexes].sum() > 0:
                        for p in path_indexes_intersected:
                            self.outflow[t][i][p] = (self.ro[t][i][p] / self.ro[t][i][path_indexes].sum()) *  min(self.demand[t][i],self.supply[t][k])
                    
                        self.outflow_signed[t][i] = min(self.demand[t][i],self.supply[t][k])

                        # Check equation (10) is correct with equation (8)
                        if abs(self.outflow[t][i].sum() - self.outflow_signed[t][i]) >= 10e-9:
                            print(f"238: {abs(self.outflow[t][i].sum() - self.outflow_signed[t][i])}, {self.outflow[t][i].sum()}, {self.outflow_signed[t][i]}")
                '''            

                if attributes['type_downstream_connection'] == 'exiting':
                    for path_index in path_indexes:
                        self.outflow[t][i][path_index] = self.free_flow_speed * self.ro[t][i][path_index]

                    
                # Diverging connection
                elif attributes['type_downstream_connection'] == 'diverging' or attributes['type_downstream_connection'] == 'ordinary':         

                    self.demand[t][i] = min(self.free_flow_speed*self.ro_signed[t][i], self.maximum_flow)         
                    successors = list(G.successors(end_node))

                    for successor in successors:
                        j_plus = self.cell_indexes[(end_node, successor)]

                        neighbor_path_indexes = self.getPathIndexByEdge((end_node, successor))
                        path_indexes_intersected = self.getIntersectedPaths(path_indexes, neighbor_path_indexes)

                        if self.ro_signed[t][i] > 0:
                            self.demand_matrix[t][i][j_plus] = (self.ro[t][i][path_indexes_intersected].sum()/self.ro[t][i].sum()) * self.demand[t][i]

                    #demand[t][i] = 0
                    #for successor in successors:
                    #    j_plus = self.cell_indexes[(end_node, successor)]
                    #    demand[t][i] += demand_matrix[t][i][j_plus]

                    sum_ro_i_b = self.ro[t][i][path_indexes].sum()
                    for successor in successors:
                        j_plus = self.cell_indexes[(end_node, successor)]
                        neighbor_path_indexes = self.getPathIndexByEdge((end_node, successor))
                        path_indexes_intersected = self.getIntersectedPaths(path_indexes, neighbor_path_indexes)
                        for b1 in path_indexes_intersected:
                            #sum_ro_i_b = self.ro[t][i][path_indexes_intersected].sum()
                            min_val = 1
                            if sum_ro_i_b > 0:
                                
                                for successor2 in successors:
                                    j_plus2 = self.cell_indexes[(end_node, successor2)]
                                    if self.demand_matrix[t][i][j_plus2] > 0:
                                        self.supply[t][j_plus2] = min(self.maximum_flow, self.backward_propagation_speed*(self.maximum_density - self.ro_signed[t][j_plus2]))
                                        min_val = min(min_val, (self.supply[t][j_plus2]/self.demand_matrix[t][i][j_plus2]))

                                #self.outflow[t][i][b1] = (self.ro[t][i][b1] / sum_ro_i_b) * self.demand_matrix[t][i][j_plus] * min_val
                                self.outflow[t][i][b1] = (self.ro[t][i][b1] / sum_ro_i_b) * self.demand[t][i] * min_val
                                #self.inflow[t][j_plus][b1] = self.outflow[t][i][b1]

                        #self.inflow_signed[t][j_plus] = self.demand_matrix[t][i][j_plus] * min_val
                    
                    # Check equation (14)
                    '''
                    self.outflow_signed[t][i] = 0
                    for successor in successors:
                        j_plus = self.cell_indexes[(end_node, successor)]
                        self.outflow_signed[t][i] += self.inflow_signed[t][j_plus]

                    if abs(self.outflow[t][i].sum() - self.outflow_signed[t][i]) >= 10e-9:
                        print(f"{abs(self.outflow[t][i].sum() - self.outflow_signed[t][i])}, {self.outflow[t][i].sum()}, {self.outflow_signed[t][i]}")
                    '''         
                self.update(t)
            
            self.update_matrices(t)


          
        print("Output Cell Densities at the last time step: ")
        print(Fore.GREEN )
        print(self.ro_signed[T])
        print("")
        print(self.ro[T])
        print(Style.RESET_ALL)

        if self.A_and_B_matrices_correct == True:
            print("A and B successfully derived")
        if self.H_matrix_correct == True:
            print("H successfully derived\n")

        
        ro_signed_json = self.ro_signed[0:-1].tolist()

        #for i, _ in enumerate(ro_signed_json):
        #    ro_signed_json[i].pop(0)
        print("Start saving all files:")

        (state, observation) = self.generateData(T, self.ro[0].flatten())

                
        with open(dir_path+'/obs.json', 'w') as f:
            json.dump(observation.tolist(), f)
        
        with open(dir_path+'/densities.json', 'w') as f:
            json.dump(ro_signed_json, f)
        print("History of cell densities are saved to densities.json file")

        with open(dir_path+'/densities_paths.json', 'w') as f:
            json.dump(self.ro.tolist(), f)
        print("History of cell densities are saved to densities.json file")
           
        with open(dir_path+'/lambdas.json', 'w') as f:
            json.dump(self.rates, f)
        print("Lambdas saved to lambdas.json file") 



        print("Starting saving matrices...")
        A_list = self.A.tolist()
        with open(dir_path+'/A.json', 'w') as f:
            json.dump(A_list, f)
        print("Matrix A saved to A.json file") 

        with open(dir_path+'/A_last.json', 'w') as f:
            json.dump(A_list[-1], f)
        print("Matrix A last saved to A_last.json file") 

            
        with open(dir_path+'/B.json', 'w') as f:
            json.dump(self.B.tolist(), f)
        print("Matrix B saved to B.json file")
        
           
        with open(dir_path+'/H.json', 'w') as f:
            json.dump(self.H.tolist(), f)
        print("Matrix H saved to H.json file") 

        with open(dir_path+'/u.json', 'w') as f:
            json.dump(self.u.tolist(), f)
        print("Matrix u saved to u.json file") 
        
        print("All files are successfully saved!")


pos = {"D4": (35, 40), "O1": (15, 30), "D5": (55, 40), "D1": (110, 20), "D2": (85, 10), "D3": (110,30), "D2": (35,30),
       "D3": (35, 20), "D6": (55,20), "D7": (55, 30), "D8": (85,30), "D9":(85,20), "D10": (55,10)}
# Create the graph as same the paper
G = nx.DiGraph()

# Add nodes to directed graph
G.add_nodes_from([
                ("O1", {"label": "1", "index":0}), 
                ("D2", {"label": "2", "index":1}), 
                ("D3", {"label": "3", "index":2}), 
                ("D4", {"label": "4", "index":3}),
                ("D5", {"label": "5", "index":4}),
                ])

# Add edges to the graph
G.add_edges_from([
                  ("O1", "D2"),
                  ("O1", "D3"),
                  ("D2", "D4"),
                  ("D4", "D5"),
                ])


#edges labels
edges_labels = {
                  ("O1", "D2"): 0,
                  ("O1", "D3"): 1, 
                  ("D2", "D4"): 2,
                  ("D4", "D5"): 3,
                }

# attributes for edges
attrs = {
            ("O1", "D2"): {"label":0,"type_upstream_connection":"entering", "type_downstream_connection": "diverging"},
            ("O1", "D3"): {"label":1,"type_upstream_connection":"entering", "type_downstream_connection": "exiting"},
            ("D2", "D4"): {"label":2,"type_upstream_connection":"ordinary", "type_downstream_connection": "exiting"},
            ("D4", "D5"): {"label":3,"type_upstream_connection":"ordinary", "type_downstream_connection": "diverging"},
        
}



# paths in the graph like in the paper 
paths = [
    # (1) Blue
    [("O1","D2"), ("D2", "D4"), ("D4","D5")],
    # (2) Red
    [("O1", "D2"), ("D2", "D4")],
    # (3) Green
    [("O1", "D3")]
]

# Set the attributes to edges and draw the graph on plot
nx.set_edge_attributes(G, attrs)
options = {"edgecolors": "tab:gray", "alpha": 0.9}
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='tab:red', font_color="whitesmoke",  **options)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_labels,  font_weight='bold', font_size=16, rotate=False)
plt.show()

lambdas_rates =  [50, 10, 15]
#lambdas_rates = [10, 15, 20, 25, 30, 5, 35, 40, 45, 50, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# simulate the road traffic

# 100 km/h free flow speed, 6000 veh/h maximum flow, 300 veh/km maximum density, 25km/h backward propagation speed, 0.5 km length cell
ctm = CTM( G, paths, lambdas_rates, 100, 6000, 300, 25, 0.5, edges_labels, False)
# output: densities of cells
ctm.simulate(1)