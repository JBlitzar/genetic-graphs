import networkx as nx
import json
import torch.nn as nn

class Module(nn.Module):
    def __init__(self, name, num_inputs, num_outputs):
        self.name = name
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_connections = 0
        self.output_connections = 0

    def can_add_input(self):
        return self.input_connections < self.num_inputs
    
    def can_add_output(self):
        return self.output_connections < self.num_outputs
    
    def add_input_connection(self):
        if self.can_add_input():
            self.input_connections += 1
        else:
            raise ValueError(f"{self.name} has reached its maximum number of inputs.")

    def add_output_connection(self):
        if self.can_add_output():
            self.output_connections += 1
        else:
            raise ValueError(f"{self.name} has reached its maximum number of outputs.")
    

    def forward(self, inputs):
        print(f"{self.name} recieving inputs {inputs}")
        
       
        if len(inputs) != self.num_inputs:
            raise ValueError(f"{self.name} expects {self.num_inputs} inputs, but got {len(inputs)}.")
        
        output = sum(inputs) # Filler function: Just sum the inputs for now
        
        return [output]# * self.num_outputs
    





class ModuleDag(Module):
    def __init__(self, name):
        self.name = name
        self.num_inputs = 1
        self.num_outputs = 1
        self.input_connections = 0
        self.output_connections = 0

        self.graph = nx.DiGraph()
        self.modules = {}
    
    def add_module(self, module):
        self.graph.add_node(module.name, num_inputs=module.num_inputs, num_outputs=module.num_outputs)
        self.modules[module.name] = module
    
    def add_connection(self, from_module, to_module):
        if from_module not in self.modules or to_module not in self.modules:
            raise ValueError("Both modules must exist in the graph before connecting them.")
        
        from_mod = self.modules[from_module]
        to_mod = self.modules[to_module]
        
        if not from_mod.can_add_output():
            raise ValueError(f"{from_module} cannot add more output connections.")
        if not to_mod.can_add_input():
            raise ValueError(f"{to_module} cannot add more input connections.")
        
        self.graph.add_edge(from_module, to_module)
        

        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(from_module, to_module)
            raise ValueError("Adding this connection creates a cycle, which is not allowed.")
        

        from_mod.add_output_connection()
        to_mod.add_input_connection()
    

    def validate_graph(self):
        for module_name, module in self.modules.items():

            if module_name == "InputModule":
                assert module.output_connections >= 1
                assert module.num_inputs >= 1 
            

            elif module_name == "OutputModule":
                assert module.num_outputs >= 1
                assert module.input_connections >= 1 
            
            # Validate all other modules normally
            else:
                if module.input_connections != module.num_inputs:
                    raise ValueError(f"{module_name} does not have the correct number of input connections.")
                if module.output_connections != module.num_outputs:
                    raise ValueError(f"{module_name} does not have the correct number of output connections.")
        

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
            module = Module(module_name, module_data['num_inputs'], module_data['num_outputs'])
            self.modules[module_name] = module
        

        for edge in self.graph.edges():
            from_module = edge[0]
            to_module = edge[1]
            self.modules[from_module].add_output_connection()
            self.modules[to_module].add_input_connection()
        

        self.validate_graph()
    

    def forward(self, initial_input):

        inputs = {"InputModule": initial_input}
        outputs = {}


        for module_name in nx.topological_sort(self.graph):
            module = self.modules[module_name]
            input_data = inputs.get(module_name, [])
            print(input_data, module_name)

            output_data = module.forward(input_data)
            outputs[module_name] = output_data
            

            print(f"Successors: {[a for a in self.graph.successors(module_name)]}")
            for successor in self.graph.successors(module_name):
                if successor in inputs:
                    inputs[successor].extend(output_data)
                else:
                    inputs[successor] = output_data
        

        final_module = "OutputModule"
        return outputs.get(final_module, [])
    



if __name__ == "__main__":

    mg = ModuleDag("TestGraph")


    mg.add_module(Module("InputModule", num_inputs=1, num_outputs=1))
    mg.add_module(Module("ProcessModuleA", num_inputs=1, num_outputs=2))
    mg.add_module(Module("ProcessModuleB", num_inputs=1, num_outputs=1))
    mg.add_module(Module("OutputModule", num_inputs=2, num_outputs=1))


    mg.add_connection("InputModule", "ProcessModuleA")
    mg.add_connection("ProcessModuleA", "ProcessModuleB")
    mg.add_connection("ProcessModuleB", "OutputModule")
    mg.add_connection("ProcessModuleA", "OutputModule")


    mg.validate_graph()
    

    initial_input = [1]
    output = mg.forward(initial_input)
    print("Final Output:", output)
