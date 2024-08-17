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




    


def mutate(dag: ModuleDag,mtype:Union[str,None]=None):
    mutation_possibilities = ["sameTransformation","skip","downup","smallbig","inout","cull","replaceSame","noop"]

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
        dag.propogate_shapes()
        print("Just mutated", mutationType)
    mutateFromType(mutationType)



if __name__ == "__main__":
    print(shape_dict)