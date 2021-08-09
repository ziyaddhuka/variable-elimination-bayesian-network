import networkx as nx
import itertools
import numpy as np
import pandas as pd
from copy import deepcopy
import math
import sys


class GraphicalModel:
    def read_file(self,model_file_path,evidence_file_path, flag):

        lines = []
        # read all the lines and remove empty lines from it
        for line in open(model_file_path,'r').read().split('\n'):
            if line.rstrip():
                lines.append(line)

        line_no = 0

        # inititalize the graph object
        self.graph = nx.Graph()

        # capturing the network type
        self.network_type = lines[line_no]
        line_no+=1

        # capturing the number of variables
        self.no_of_variables = lines[line_no]
        line_no+=1

        # capturing the cardinalities of the variables
        cardinality_line = lines[line_no].rsplit()
        # storing cardinalities a as list of int
        self.variable_cardinalities = list(map(int,cardinality_line))
        line_no+=1

        # capturing number of cliques
        self.no_of_cliques = int(lines[line_no])
        line_no+=1

        self.factors = []
        self.cpt = []

        # looping through all cliques and adding it them as nodes and vertices of the graph
        for i in range(self.no_of_cliques):
            cliques_input = lines[line_no+i]
            cliques = list(map(int,lines[line_no+i].rsplit()))[1:]
            # adding nodes to the graph
            self.graph.add_nodes_from(cliques)
            # check length of cliques if > 1 then add that edge
            if(len(cliques)>1):
                # if there are more than 2 nodes in the cliques then generate all combinations of pairs and add edge to the graph
                self.graph.add_edges_from(list(set(itertools.combinations(cliques, 2))))
            # append cliques to the factors list
            self.factors.append(np.array(cliques))

        line_no = line_no+i+1

        # looping and saving all the factor table
        for k in range(self.no_of_cliques):
            var = lines[line_no+k]
            self.cpt.append(np.array(list(map(float,lines[line_no+k+1].split(' ')))))
            line_no = line_no+1



        # looping through evidence file and storing the evidence variables
        if flag==0:
            lines = open(evidence_file_path,'r').read().split('\n')
            line_content = lines[0].rsplit(' ')
            total_evid_vars = int(line_content[0])
            evidence = {}

            if total_evid_vars > 0:
                for i in range(1,total_evid_vars*2,2):
                    evidence[int(line_content[i])] = int(line_content[i+1])
            self.evidence = evidence

        else:
            self.evidence = {}

        # combining the factors and factor tables together in a single array. Will refer to it as "COMBINED ARRAY"
        self.factors_and_cpt = np.array((self.factors,self.cpt),dtype=object).T
        return


"""
instantiate_evidence(evidence,arr)
params: evidence variables (a dictionary containing the variable and the value) and the "COMBINED ARRAY"
uses evidence variables and reduces factors accordingly
stores the reduced factors once the evidence is instantiated
returns modified array after instantiating the array and reduced evidence factors
"""
def instantiate_evidence(evidence,arr):
    # copying the combined array contents into a new array to prevent it from modifying original array
    narr = deepcopy(arr)

    # initializing the empty arrays
    reduced_evidence_factors = []
    index_to_delete = []


    for key,value in evidence.items():
        for i in range(0,narr.shape[0]):
            if key in narr[:,0][i]:
                #find the position of key in the list and instantiate evidence accordingly
                dimension = len(narr[:,][i][0]) #finding the number of variables in the clique
                idx = np.where(narr[:,][i][0]==key)[0][0] # getting the index of evidence variable in the clique
                factor_index_to_keep = get_binary(dimension,idx,value) # getting the index of the factor to keep
                narr[:,1][i] = np.array(narr[:,1][i])[factor_index_to_keep] # getting the value of that index i.e evidence variable
                narr[:,][i][0] = np.delete(narr[:,][i][0],np.where(narr[:,][i][0]==key)) # deleting the variable
                if narr[:,][i][0].size==0: # checking if the reduced clique is 0
                    # if yes then store the single factor
                    reduced_evidence_factors.append(np.array(narr[:,][i][1][0]))
                    index_to_delete.append(i)
    # return the modified array and reduced factor
    return narr,reduced_evidence_factors



"""
function to get min degree order of the variables
input: graph and evidence_nodes
output: min degree variables order
"""
def get_min_degree_order(graph,evidence_nodes):
    evidence_nodes = list(evidence_nodes.keys())
    order = []
    # copy the graph and retain the original graph
    tp_g = graph.copy()

    # Get evidence nodes and remove them from the graph before computing the min degree order
    for node in evidence_nodes:
        tp_g.remove_node(node)

    # loop through all the nodes
    for i in range(0,len(tp_g.nodes)):

        # get degree of that node
        temp = dict(tp_g.degree())
        # upgrade it into a dictionary of key value pairs having node as key and degree of that node as value
        # and sort the dictionary by value
        temp = dict(sorted(temp.items(), key=lambda item: item[1]))
        # take the node with least degree
        s = list(temp.keys())[0]
        # get all the edges of that node
        edges = [i[1] for i in list(tp_g.edges(s))]
        # if there are more than 1 edges then we need to connect the edges once we delete the node
        if len(edges)>1:
            # connect all possible edges of all the nodes which are connected to the node to be deleted
            tp_g.add_edges_from(list(set(itertools.combinations(edges,2))))
        # delete the selected node
        tp_g.remove_node(s)
        # append the node to order
        order.append(s)
    return order


"""
function takes input dimension, position of variable, and evidence
suppose there are 2 variables- (X1,X2) then
dimension i.e n = 2
and lets say that the instantiated variable is X2 then var_pos = 2
and assume that the X2 takes the value X2=1 then the evid = 1
"""
def get_binary(n,var_pos,evid):
    # initialize empty array
    l = []
    # loop from 0 to 2^n, used bitwise shift here
    for i in range(1<<n):
        # bin(2)='0b10' and bin(4)='0b100' but we need only the string after b
        s=bin(i)[2:]
        # expanding the binary to fit it to the dimension for e.g. 10 will be 010 in dimension = 3
        s='0'*(n-len(s))+s
        # check if the the corresponding position of evidence matches with the bit value
        # if yes then we only need those
        if s[var_pos]==str(evid):
            l.append(i)
    return np.array(l)



"""
function to get unique variables in a factor
during bucket elimination there might be bucket which contains may factors and some of them get repeated
This function gives unique factors present in the array. For us it gives us unique variables in the bucket
"""
def get_unique_vars_factors(arr):
    output = set()
    for i in range(arr.shape[0]):
        for item in arr[i][0]:
            output.add(item)
    return list(output)



def variable_elimination(factors_and_fact_table,reduced_evidence_factors, order):
    # making copy of the array
    cp_factors_and_fact_table = deepcopy(factors_and_fact_table)
    # initializing the bucket as per the min-degree order
    bucket = {k: [] for k in order}

    # looping through all the bucket values in order
    for key,value in bucket.items():
        # initializing empty arrays
        factor_index = []
        cpt_value = []
        idx = []

        # looping through all the factors
        for i in range(cp_factors_and_fact_table.shape[0]):
            # checking if bucket variable is inside the factor array
            if key in cp_factors_and_fact_table[i][0]:
                # if yes than append the factor table of that corresponding factor
                cpt_value.append(cp_factors_and_fact_table[i][1])
                # store the factor as well
                factor_index.append(cp_factors_and_fact_table[i][0])
                # store the index of the factor so as to delete it once processe
                idx.append(i)

        # add those factors to the bucket
        bucket_array = np.array((factor_index,cpt_value,idx),dtype=object).T
        # get unique variables inside the bucket array
        new_clique = get_unique_vars_factors(bucket_array)

        # number of variables is the dimension
        dimension = len(new_clique)
        # format to generate binary numbers
        s = '{0:0'+str(dimension)+'b}' # used to interpret address
        idxes = []
        fact = []

        # loop to all possible numbers from 0 to 2^n
        for i in range(2**dimension):
            # genrate binary string of the number
            binar = s.format(i)
            # convert all the characters to list
            bin_d = list(binar)

            # creating an empty dictionary
            binary_dist = {}
            j=0

            # loop through all the node in the clique and assigning the corresponding bit to that node
            for node in new_clique:
                # assigning the corresponding bit to that node
                binary_dist[node] = bin_d[j]
                j = j+1

            """
            here we can reuse the instantiate evidence function.While doing such we will get all variables
            corresponding to that bit assignment. For e.g. if we have number 1 and two variables X1 and X2
            then the bit generated in the second loop the binary number of 1 will be 01 hence X1=0 and X2=1
            and then get corresponding values of that assignment
            """

            narr,_ = instantiate_evidence(binary_dist,bucket_array)

            # append the indexes which are looped through
            idxes.append(list(binary_dist.values()))
            # multiply all the corresponding factors
            fact.append(np.prod(narr[:,1]))

        # converting to numpy arrays
        idxes = np.array(idxes)
        fact = np.array(fact)

        # converting the arrays in dataframe of columns having nodes and factor
        temp_df = pd.DataFrame(np.hstack((idxes,fact)),columns= new_clique + ['factor'])
        # drop the key/bucket node
        temp_df = temp_df.drop(columns = key)
        # convert factor to float
        temp_df['factor'] = temp_df['factor'].astype(np.float64)

        # remove the bucket node from the array. thus giving us a new factor which is summed out and free of that node
        new_clique.remove(key)
        # check if the length of new clique is > 0
        if len(new_clique)>0:
            # if yes then we need to sum out
            # using groupby function of pandas and grouping by the new_cliques as the keys
            summed_out_factor = np.array(temp_df.groupby(new_clique,as_index=False).sum('factor')['factor'])

            # converting the lists to numpy array
            new_clique = np.array(new_clique)
            summed_out_factor = np.array(summed_out_factor)

            # we store new clique factor and the corresponding summed out value
            new_row_of_clique_and_factor = np.array((new_clique,summed_out_factor),dtype=object)

            # adding it to the last row of the factor and tables array
            cp_factors_and_fact_table = np.vstack((cp_factors_and_fact_table,new_row_of_clique_and_factor))

        else:
            # if length is = 0 this means that we have no factors left to sum out hence we have only one value at the end
            # this value will be stored and multiplied with the final factor remaining after the bucket elimination algorithm
            # we sum all the values and store
            reduced_evidence_factors.append(np.array(temp_df['factor'].sum()))

        # deleting the factors processed in one pass of bucket elimination algorithm
        cp_factors_and_fact_table = np.delete(cp_factors_and_fact_table,(bucket_array[:,-1].tolist()),axis=0)

    # returning the reduced factors which includes single factors and the factor at the very end of the bucket elimination algorithm
    return reduced_evidence_factors



if __name__=='__main__':

    flag = 0
    input_length = len(sys.argv)
    # checking the length of the inputs
    if input_length>2:
        # input is > 2 means the evidence file is passed
        evidence_file = sys.argv[2]
    else:
        # else only one file is passed then keep evidence file as None
        evidence_file=None
        flag = 1

    # network file is first argument
    network_file = sys.argv[1]

    # initializing the GraphicalModel object
    k = GraphicalModel()
    # reading the network and evidence files
    k.read_file(network_file,evidence_file,flag)
    # copying the factors
    factors_and_fact_table = deepcopy(k.factors_and_cpt)
    # calling the instantiate evidence function which returns (factors and factor table array) and (reduced_evidence_factors)
    factors_and_fact_table, reduced_evidence_factors = instantiate_evidence(k.evidence,factors_and_fact_table)
    # getting variable order using min degree ordering
    order = get_min_degree_order(k.graph,k.evidence)

    # calling the variable_elimination function and passing the factors and factor table array along with reduced_evidence_factors and variable order
    z = variable_elimination(factors_and_fact_table,reduced_evidence_factors, order)
    # taking log10 value of final Output
    zf = math.log10(np.prod(z))
    print(zf)

    # writing output to the file
    f = open("output.txt", "w")
    f.write("Output = {}".format(zf))
    f.close()
