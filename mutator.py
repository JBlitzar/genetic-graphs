import inspect
import modules
from moduledag import ModuleDag
from typing import Tuple, Union
import random
import uuid
import networkx as nx


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




    


def mutate(dag: ModuleDag):
    mutation_possibilities = ["sameTransformation","skip","downup","smallbig","inout","cull","replaceSame","noop"]

    mutationType = "skip"#random.choice(mutation_possibilities)

    print(mutationType)

    

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
                if dag.has_one_predecessor(candidate.name) and hasattr(candidate, "shape_transform"):
                    break


            candidate_2 = None

            
            while True:
                candidate_2 = dag.get_random_node()
                if dag.has_one_predecessor(candidate_2.name) and hasattr(candidate_2, "shape_transform") and candidate_2.realShape == candidate.realShape:
                    break
            


            pred_ref = list(dag.graph.predecessors(candidate_2.name))[0]
            print(candidate)
            print("->")
            print(pred_ref)
            print(candidate_2)
            
            dag.add_module(connector)
            dag.remove_connection(pred_ref,candidate_2.name)
            try:
                dag.add_connection(pred_ref,connector.name)
                dag.add_connection(connector.name, candidate_2.name)
                dag.add_connection(candidate.name, connector.name)
            except ValueError: # cycle
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
                except nx.exception.NetworkXError:
                    pass
                
                
                

                
                dag.add_connection(pred_ref,candidate_2.name)
                pred_ref = list(dag.graph.predecessors(candidate.name))[0]
                dag.remove_connection(pred_ref,candidate_2.name)
                print(candidate)
                print("->")
                print(pred_ref)
                print(candidate_2)


                dag.add_connection(pred_ref,connector.name)
                dag.add_connection(connector.name, candidate.name)
                dag.add_connection(candidate_2.name, connector.name)




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


            while True:
                candidate = dag.get_random_node()
                if dag.has_one_predecessor(candidate.name) and hasattr(candidate, "shape_transform"):
                    break

            
            dag.remove_node_and_link(candidate.name)

        case "replaceSame":
            while True:
                n = dag.get_random_node()
                if hasattr(n, "shape_transform"):
                    break
            dag.replace_node(n.name, rand_of_transform(n.shape_transform))
        case "noop":
            pass





if __name__ == "__main__":
    print(shape_dict)