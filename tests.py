from moduledag import ModuleDag, Module, ImageModule
from modules import SimpleConv, ReLU
import torch
from mutator import mutate
import matplotlib.pyplot as plt

mg = ModuleDag("TestNet", input_size=(1,1,28,28))
inputmod = ImageModule("InputModule",shapeTransform=(1,1,1))
inputmod.setShape((1,1,28,28))
inputmod.init_block()
mg.add_module(inputmod)
mg.add_module(SimpleConv("conv1"))
mg.add_module(ReLU("relu1"))
mg.add_module(SimpleConv("conv2"))
mg.add_module(ReLU("relu2"))
mg.add_module(Module("OutputModule", num_inputs=1, num_outputs=1))


mg.add_connection("InputModule", "conv1")

mg.add_connection("conv1", "relu1")
mg.add_connection("relu1", "conv2")
mg.add_connection("conv2", "relu2")
mg.add_connection("relu2", "OutputModule")


mg.validate_graph()




initial_input = torch.randn((1,1,28,28))
output = mg.forward(initial_input)
print("Final Output:", output)

#mg.display()

plt.close()

mutate(mg,"skip")
for i in range(10):
    mg = mutate(mg)
    #mg.display()
    mg.validate_graph()




initial_input = torch.randn((1,1,28,28))
output = mg.forward(initial_input)
print("Final Output:", output)

mg.display()

comp = torch.compile(mg)
comp.to("mps")
print(comp(torch.randn((1,1,28,28))))