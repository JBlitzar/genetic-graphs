import inspect
import modules
from moduledag import ModuleDag
from typing import Tuple
import random


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
def rand_of_transform(transform: Tuple[int,int,int]):
    choices = shape_dict[transform]
    return random.choice(choices)




    


def mutate(dag: ModuleDag):
    mutation_possibilities = ["sameTransformation","skip","downup","cull","replaceSame"]

    mutationType = random.choice(mutation_possibilities)


    

    match mutationType:
        case "sameTransformation":
            u,v = dag.randomEdge()
            dag.insertBetween(rand_of_transform((1,1,1)),u,v)
        case "skip":
            pass
        case "downup":
            pass
        case "cull":
            pass
        case "replaceSame":
            pass





if __name__ == "__main__":
    print(shape_dict)