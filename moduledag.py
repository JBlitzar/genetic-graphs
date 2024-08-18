import networkx as nx
import json
import torch.nn as nn
import torch
import random
import inspect
from display import display_graph

class FunctionalToClass(nn.Module):
    def __init__(self, function, *args, **kwargs) -> None:
        self.function = function
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return self.function(x)
    
class Module(nn.Module):
    def __init__(self, name, num_inputs, num_outputs):
        super().__init__()
        self.name = name
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_connections = 0
        self.output_connections = 0
        self.realShape = None

    def can_add_input(self):
        return self.input_connections < self.num_inputs
    
    def can_add_output(self):
        return self.output_connections < self.num_outputs
    
    def add_input_connection(self, other):
        if self.can_add_input():
            self.input_connections += 1
        else:
            raise ValueError(f"{self.name} has reached its maximum number of inputs.")

    def add_output_connection(self, other):
        if self.can_add_output():
            self.output_connections += 1
        else:
            raise ValueError(f"{self.name} has reached its maximum number of outputs.")
        
    def validate_inout(self):
        return self.input_connections == self.num_inputs and self.output_connections == self.num_outputs
    
    def propogate_realShape(self, others):
        for other in others:
            other.realShape = self.realShape
    

    def forward(self, inputs):
        # print(f"{self.name} recieving inputs {inputs}")
        
       
        if len(inputs) != self.num_inputs:
            raise ValueError(f"{self.name} expects {self.num_inputs} inputs, but got {len(inputs)}.")
        
        output = sum(inputs) # Filler function: Just sum the inputs for now
        
        return [output]# * self.num_outputs


class ImageModule(Module):
    def __init__(self, name, shapeTransform):
        super().__init__(name, 1, 1)

        self.realShape = None
        if len(shapeTransform) == len((1,1,1)):
            shapeTransform = [1] + list(shapeTransform)
        self.shapeTransform = shapeTransform

    def setShape(self,shape):
        self.realShape = shape


    def get_output_shape(self):
        return tuple(a * b for a, b in zip(self.realShape, self.shapeTransform))

    def init_block(self):
        # self.realShape is available
        self.block = nn.Identity()

    
    def can_add_output(self):
        return True

    
    def add_input_connection(self, other):
        if self.can_add_input():
            self.input_connections += 1
        else:
            raise ValueError(f"{self.name} has reached its maximum number of inputs.")

    def add_output_connection(self, other):
        if self.can_add_output():
            self.output_connections += 1
            self.num_outputs = self.output_connections
        else:
            raise ValueError(f"{self.name} has reached its maximum number of outputs.")
        other.realShape = self.get_output_shape()

        print(f"{self.name} adding an output connection! {self.get_output_shape()}")
        try:
            other.init_block()
        except AttributeError:
            # Module is a noop or output layer, ignore
            pass

    def propogate_realShape(self, others):
        for other in others:
            other.realShape = self.get_output_shape()
    

    def forward(self, inputs):
        x = torch.stack(inputs) if type(inputs) == type([]) else inputs

        x = x.squeeze(0)


        return [self.block(x)]


class CombinationModule(Module):
    def __init__(self, name):
        super().__init__(name, 2, 1)

        self.realShape = None
        self.shapeTransform = (1,1,1,1)


    def get_output_shape(self):
        return tuple(a * b for a, b in zip(self.realShape, self.shapeTransform))

    def init_block(self):
        # self.realShape is available
        self.block = FunctionalToClass(lambda tensors: torch.stack(tensors).sum(dim=0))

    
    def can_add_output(self):
        return True
    

    def validate_inout(self):
        return True # hack
    
    def add_input_connection(self, other):
        if self.can_add_input():
            self.input_connections += 1
        else:
            raise ValueError(f"{self.name} has reached its maximum number of inputs.")

    def add_output_connection(self, other):
        if self.can_add_output():
            self.output_connections += 1
            self.num_outputs += 1
        else:
            raise ValueError(f"{self.name} has reached its maximum number of outputs.")
        other.realShape = self.get_output_shape()
        try:
            other.init_block()
        except AttributeError:
            pass

    def propogate_realShape(self, others):
        for other in others:
            other.realShape = self.get_output_shape()
    

    def forward(self, inputs):
        x = tuple(inputs)


        return [self.block(x)]




class ModuleDag(Module):
    def __init__(self, name, input_size: tuple):
        super().__init__(name, 1,1)
        self.name = name
        self.num_inputs = 1
        self.num_outputs = 1
        self.input_connections = 0
        self.output_connections = 0
        self.input_size = input_size

        self.graph = nx.DiGraph()
        self.modules_moduledag = nn.ModuleDict()
    
    def add_module(self, module):
        def get_child_attributes(child_class):

            child_attrs = set(dir(child_class))


            parent_attrs = set()
            parent_attrs = set()
            for cls in inspect.getmro(type(child_class)):
                if cls is not type(child_class):
                    parent_attrs.update(dir(cls))
                    child_specific_attrs = child_attrs - parent_attrs
            
            child_specific_attrs = [a for a in child_specific_attrs if not a.startswith("_")]
            return child_specific_attrs
        
        print(get_child_attributes(module))
        self.graph.add_node(module.name, module_type=module.__class__.__name__)
        self.modules_moduledag[module.name] = module

    def remove_module(self, module):
        self.graph.remove_node(module)
        self.modules_moduledag.pop(module, None)
    
    def add_connection(self, from_module, to_module):
        if from_module not in self.modules_moduledag or to_module not in self.modules_moduledag:
            raise ValueError(f"Both modules must exist in the graph before connecting them. {from_module in self.modules_moduledag} {to_module in self.modules_moduledag}")
        
        from_mod = self.modules_moduledag[from_module]
        to_mod = self.modules_moduledag[to_module]
        
        if not from_mod.can_add_output():
            raise ValueError(f"{from_module} cannot add more output connections.")
        if not to_mod.can_add_input():
            raise ValueError(f"{to_module} cannot add more input connections.")
        
        self.graph.add_edge(from_module, to_module)
        

        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(from_module, to_module)
            raise ValueError("Adding this connection creates a cycle, which is not allowed.")
        

        from_mod.add_output_connection(to_mod)
        to_mod.add_input_connection(from_mod)
    

    def propogate_shapes(self):
        start = "InputModule"

        def prop(mod):
            children = self.graph.successors(mod.name)

            module_children = [self.modules_moduledag[a] for a in children]

            mod.propogate_realShape(module_children)


            for child in module_children:
                prop(child)

    def validate_graph(self):

        self.propogate_shapes()


        for module_name, module in self.modules_moduledag.items():

            if module_name == "InputModule":
                assert module.output_connections >= 1
                assert module.num_inputs >= 1 
            

            elif module_name == "OutputModule":
                assert module.num_outputs >= 1
                assert module.input_connections >= 1 
            
            # Validate all other modules normally
            else:
                if not module.validate_inout():
                    print(f"{module_name}")
                    #print(f"{module.}")
                    raise ValueError(f"{module_name} does not have the correct number of input/output connections. {module.input_connections} {module.output_connections}")
                
        with torch.no_grad():
            self.eval()
            
            self.forward(torch.randn(self.input_size))

        

    def serialize_to_json(self):
        data = nx.node_link_data(self.graph)
        json_data = json.dumps(data, indent=4)
        # print(json_data)
        return json_data
    
    # Step 7: Load from JSON
    def load_from_json(self, json_data):
        data = json.loads(json_data)
        self.graph = nx.node_link_graph(data)
        for node in self.graph.nodes(data=True):
            module_name = node[0]
            module_data = node[1]
            module = globals()[module_data["module_class"]](module_name)
            self.modules_moduledag[module_name] = module
        

        for edge in self.graph.edges():
            from_module = edge[0]
            to_module = edge[1]
            self.modules_moduledag[from_module].add_output_connection()
            self.modules_moduledag[to_module].add_input_connection()
        

        self.validate_graph()
    

    def forward(self, initial_input):

        inputs = {"InputModule": initial_input}
        outputs = {}


        for module_name in nx.topological_sort(self.graph):
            module = self.modules_moduledag[module_name]
            input_data = inputs.get(module_name, [])
            

            output_data = module.forward(input_data)
            outputs[module_name] = output_data
            

            print(f"Successors: {[a for a in self.graph.successors(module_name)]}")
            for successor in self.graph.successors(module_name):
                if successor in inputs:
                    inputs[successor].extend(output_data)
                else:
                    inputs[successor] = list(output_data)

                # print(f"{successor}'s inputs are set to {inputs[successor]}")
                
        

        final_module = "OutputModule"
        return outputs.get(final_module, [])
    

    def randomEdge(self):
        return random.choice(list(self.graph.edges))

    def insertBetween(self,new,u,v):

        self.add_module(new)
        self.remove_connection(u, v)


        self.add_connection(u, new.name)
        self.add_connection(new.name, v)

    def remove_connection(self,u,v):
        self.graph.remove_edge(u,v)
        self.modules_moduledag[u].output_connections -= 1
        self.modules_moduledag[v].input_connections -= 1

    def insertChainBetween(self, new_nodes, u, v):
        if not new_nodes:
            raise ValueError("The list of new nodes cannot be empty.")
        
        for node in new_nodes:
            if node not in self.graph:
                self.add_module(node)
        

        self.remove_connection(u, v)
        
        self.add_connection(u, new_nodes[0].name)
        
        for i in range(len(new_nodes) - 1):
            self.add_connection(new_nodes[i].name, new_nodes[i + 1].name)
        
        self.add_connection(new_nodes[-1].name, v)

    def has_one_predecessor(self,node):
        predecessors = list(self.graph.predecessors(node))

        return len(predecessors) == 1

    def get_random_node(self):
        nodes = list(self.graph.nodes)
        return self.modules_moduledag[random.choice(nodes)]
    
    def replace_node(self, old_node, new_node):
        new_node_ref = new_node.name
        if old_node not in self.graph:
            raise ValueError(f"Old node {old_node} not found in the graph.")

        if new_node_ref in self.graph:
            raise ValueError(f"New node {new_node} already exists in the graph.")


        predecessors = list(self.graph.predecessors(old_node))
        successors = list(self.graph.successors(old_node))


        
        self.remove_module(old_node)


        self.add_module(new_node)

        for pred in predecessors:
            self.modules_moduledag[pred].output_connections -= 1
            self.add_connection(pred, new_node_ref)
        
        for succ in successors:
            self.modules_moduledag[succ].input_connections -= 1
            self.add_connection(new_node_ref, succ)

    def remove_node_and_link(self, node):

        predecessors = list(self.graph.predecessors(node))
        successors = list(self.graph.successors(node))


        if len(predecessors) > 1:
            raise ValueError(f"Node {node} has more than one predecessor.")


        if predecessors:
            
            pred = predecessors[0]
            self.remove_connection(pred, node)
            for succ in successors:
                self.remove_connection(node, succ)
                self.add_connection(pred, succ)


        self.remove_module(node)
    def display(self):
        display_graph(self.graph)






if __name__ == "__main__":

    mg = ModuleDag("TestGraph", input_size=1)


    mg.add_module(Module("InputModule", num_inputs=1, num_outputs=2))
    mg.add_module(Module("ProcessModuleA", num_inputs=1, num_outputs=2))
    mg.add_module(Module("ProcessModuleB", num_inputs=1, num_outputs=1))
    mg.add_module(Module("ProcessModuleC", num_inputs=1, num_outputs=1))
    mg.add_module(Module("OutputModule", num_inputs=3, num_outputs=1))


    mg.add_connection("InputModule", "ProcessModuleA")
    mg.add_connection("InputModule", "ProcessModuleC")
    mg.add_connection("ProcessModuleA", "ProcessModuleB")
    mg.add_connection("ProcessModuleC", "OutputModule")
    mg.add_connection("ProcessModuleB", "OutputModule")
    mg.add_connection("ProcessModuleA", "OutputModule")


    mg.validate_graph()
    

    initial_input = [1]
    output = mg.forward(initial_input)
    print("Final Output:", output)

    mg.display()
