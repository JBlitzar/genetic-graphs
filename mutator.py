import inspect
import modules
from moduledag import ModuleDag
from typing import Tuple, Union
import random
import uuid
import networkx as nx
from copy import deepcopy


def get_classes():
    all_classes = inspect.getmembers(modules, inspect.isclass)
    
    classes = [cls for name, cls in all_classes if cls.__module__ == "modules"]

    
    return classes


shape_dict = {}
classes = get_classes()
for cls in classes:

    if cls.shape_transform in shape_dict:
        
        shape_dict[cls.shape_transform].append(cls)
    else:
        shape_dict[cls.shape_transform] = [cls]

def initialize(cls):
    return cls(name=cls.__name__ +"_"+ str(uuid.uuid4()))


def rand_of_transform(transform: Union[Tuple[int,int,int],str]):
    choices = shape_dict[transform]
    return initialize(random.choice(choices))




    


def mutate(dag: ModuleDag,mtype:Union[str,None]=None):
    dag = deepcopy(dag)
    mutation_possibilities = ["sameTransformation","skip","downup","smallbig","inout","cull","replaceSame","noop","unskip","undownup","unsmallbig","uninout"]

    mutationType = random.choice(mutation_possibilities)
    if mtype is not None:
        mutationType = mtype
    print(mutationType)

    
    def mutateFromType(mutationType):
        match mutationType:

            case "sameTransformation":
                u,v = dag.randomEdge()
                print("sametrans", u, v)
                dag.insertBetween(rand_of_transform((1,1,1)),u,v)
            case "skip":

                connector = rand_of_transform("comb")


                candidate = None

                
                while True:
                    candidate = dag.get_random_node()
                    if dag.has_one_predecessor(candidate.name) and hasattr(candidate, "shape_transform") and candidate.realShape != None:
                        break


                candidate_2 = None

                i = 0
                while True:
                    candidate_2 = dag.get_random_node()
                    if i > 1000 or dag.has_one_predecessor(candidate_2.name) and hasattr(candidate_2, "shape_transform") and candidate_2.get_output_shape() == candidate.realShape and candidate_2 != candidate and len(list(dag.graph.predecessors(candidate_2.name))) >= 1:
                        break
                    i += 1



                print("========")
                print(candidate.realShape)
                print(candidate_2.realShape)
                print("========")



                
                try:
                    if i > 1000 or candidate.realShape != candidate_2.realShape:
                        raise ValueError
                    pred_ref = list(dag.graph.predecessors(candidate_2.name))[0]
                    print(candidate)
                    print("->")
                    print(pred_ref)
                    print(candidate_2)
                    
                    dag.add_module(connector)
                    dag.remove_connection(pred_ref,candidate_2.name)
                    dag.add_connection(pred_ref,connector.name)
                    dag.add_connection(connector.name, candidate_2.name)
                    dag.add_connection(candidate.name, connector.name)
                    if(connector.input_connections < 2) or len(list(dag.graph.predecessors(connector.name))) < 2: # hack
                        raise ValueError
                except ValueError: # cycle
                    # hacky try-excepts because idk it raises valueError for all sorts of reasons.
                    try:
                        dag.remove_connection(candidate.name, connector.name)
                    except nx.exception.NetworkXError:
                        pass
                    try:
                        dag.remove_connection(connector.name, candidate_2.name)
                    except nx.exception.NetworkXError:
                        pass
                    try:
                        dag.remove_connection(pred_ref,connector.name)
                    except (nx.exception.NetworkXError,UnboundLocalError):
                        pass
                    
                    
                    

                    try:
                        dag.add_connection(pred_ref,candidate_2.name)
                    except UnboundLocalError:
                        pass
                    try:
                        dag.remove_module(connector.name)
                    except nx.exception.NetworkXError:
                        pass

                    mutateFromType("skip") #hack


                    #TODO fix
                    




            case "smallbig":
                u,v = dag.randomEdge()
                dag.insertChainBetween([rand_of_transform((1,0.5,0.5)),rand_of_transform((1,2,2))],u,v)
            case "inout":
                u,v = dag.randomEdge()
                print(u,v)
                dag.insertChainBetween([rand_of_transform((2,1,1)),rand_of_transform((0.5,1,1))],u,v)
            case "downup":
                u,v = dag.randomEdge()
                dag.insertChainBetween([rand_of_transform((2,0.5,0.5)),rand_of_transform((0.5,2,2))],u,v)
            case "cull":
                candidate = None

                i = 0
                while i < 1000:
                    i += 1
                    candidate = dag.get_random_node()
                    if dag.has_one_predecessor(candidate.name) and hasattr(candidate, "shape_transform") and (all(x == 1 for x in candidate.shape_transform)) and candidate.num_outputs == 1:
                        break

                if candidate != None:
                    dag.remove_node_and_link(candidate.name)

            case "replaceSame":
                while True:
                    n = dag.get_random_node()
                    if hasattr(n, "shape_transform"):
                        break
                dag.replace_node(n.name, rand_of_transform(n.shape_transform))
            case "noop":
                pass

            case "unskip":
                connector = None
                candidate = None
                candidate_2 = None


                for node in dag.graph.nodes:
                    if node.shapeTransform == "comb":
                        connector = node
                        predecessors = list(dag.graph.predecessors(node.name))
                        if len(predecessors) == 2:
                            candidate = predecessors[1]
                            candidate_2 = list(dag.graph.successors(node.name))[0]
                            break

                if connector is None or candidate is None or candidate_2 is None:
                    raise ValueError("No valid skip operation to undo.")

                pred_ref = list(dag.graph.predecessors(connector.name))[0]

                dag.remove_connection(candidate.name, connector.name)
                dag.remove_connection(connector.name, candidate_2.name)
                dag.remove_connection(pred_ref, connector.name)
                dag.add_connection(pred_ref, candidate_2.name)
                dag.remove_module(connector.name)

            case "unsmallbig":
                flag = False
                for u, v in dag.graph.edges:

                    if dag.graph.out_degree(u) == 1 and dag.graph.in_degree(v) == 1:

                        middle_node = list(dag.graph.successors(u))[0]
                        if (hasattr(middle_node, 'shape_transform') and
                            dag.graph.out_degree(middle_node) == 1 and
                            dag.graph.in_degree(middle_node) == 1):
                            
                            next_node = list(dag.graph.successors(middle_node))[0]
                            if (hasattr(next_node, 'shape_transform') and
                                dag.graph.out_degree(next_node) == 1 and
                                dag.graph.in_degree(next_node) == 1):
                                

                                if middle_node.shape_transform == (1, 0.5, 0.5) and next_node.shape_transform == (1, 2, 2):

                                    dag.remove_connection(u, middle_node.name)
                                    dag.remove_connection(middle_node.name, next_node.name)
                                    dag.remove_connection(next_node.name, v)

                                    dag.add_connection(u, v)

                                    dag.remove_module(middle_node.name)
                                    dag.remove_module(next_node.name)

                                    print("unsmallbig performed successfully")
                                    flag = True
                                    break
                if not flag:
                    raise ValueError("No valid smallbig operation to undo.")
            case "uninout":
                flag = False
                for u, v in dag.graph.edges:

                    if dag.graph.out_degree(u) == 1 and dag.graph.in_degree(v) == 1:

                        middle_node = list(dag.graph.successors(u))[0]
                        if (hasattr(middle_node, 'shape_transform') and
                            dag.graph.out_degree(middle_node) == 1 and
                            dag.graph.in_degree(middle_node) == 1):
                            
                            next_node = list(dag.graph.successors(middle_node))[0]
                            if (hasattr(next_node, 'shape_transform') and
                                dag.graph.out_degree(next_node) == 1 and
                                dag.graph.in_degree(next_node) == 1):
                                

                                if middle_node.shape_transform == (2, 1, 1) and next_node.shape_transform == (0.5, 1, 1):

                                    dag.remove_connection(u, middle_node.name)
                                    dag.remove_connection(middle_node.name, next_node.name)
                                    dag.remove_connection(next_node.name, v)

                                    dag.add_connection(u, v)

                                    dag.remove_module(middle_node.name)
                                    dag.remove_module(next_node.name)

                                    print("unsmallbig performed successfully")
                                    flag = True
                                    break
                if not flag:
                    raise ValueError("No valid smallbig operation to undo.")
            case "undownup":
                flag = False
                for u, v in dag.graph.edges:

                    if dag.graph.out_degree(u) == 1 and dag.graph.in_degree(v) == 1:

                        middle_node = list(dag.graph.successors(u))[0]
                        if (hasattr(middle_node, 'shape_transform') and
                            dag.graph.out_degree(middle_node) == 1 and
                            dag.graph.in_degree(middle_node) == 1):
                            
                            next_node = list(dag.graph.successors(middle_node))[0]
                            if (hasattr(next_node, 'shape_transform') and
                                dag.graph.out_degree(next_node) == 1 and
                                dag.graph.in_degree(next_node) == 1):
                                

                                if middle_node.shape_transform == (2, 0.5, 0.5) and next_node.shape_transform == (0.5, 2, 2):

                                    dag.remove_connection(u, middle_node.name)
                                    dag.remove_connection(middle_node.name, next_node.name)
                                    dag.remove_connection(next_node.name, v)

                                    dag.add_connection(u, v)

                                    dag.remove_module(middle_node.name)
                                    dag.remove_module(next_node.name)

                                    print("unsmallbig performed successfully")
                                    flag = True
                                    break
                if not flag:
                    raise ValueError("No valid smallbig operation to undo.")
            
        dag.propogate_shapes()
        print("Just mutated", mutationType)


    ogdag = deepcopy(dag)
    i = 0
    while i < 1000:
        try:
            
            mutateFromType(mutationType)
            dag.validate_graph()
            break
        except:
            i += 1
            dag = deepcopy(ogdag)
    return dag



if __name__ == "__main__":
    print(shape_dict)