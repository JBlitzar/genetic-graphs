from moduledag import ModuleDag,ImageModule
from modules  import SimpleConv,ReLU,Module
from mutator import mutate
from trainer import train_model
import torch
import uuid
import numpy as np
from tqdm import trange
import os
import torch.nn as nn
import numpy as np
import os
os.system(f"caffeinate -is -w {os.getpid()} &")


class MNISTModuleDagWrapper(nn.Module):
    def __init__(self,dag,input_size,output_size):
        super().__init__()
        self.dag = dag
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,output_size)
        )
        self.input_size = input_size
        self.output_size = output_size

    def forward(self,x):
        x = torch.Tensor(x)
        #print(x.size())
        d = self.dag(x)
        #print(len(d))
        ds = torch.stack(d)
        ds = ds.squeeze(0)
        #print(ds.size())
        #print("^^^ d")
        z = self.block(torch.squeeze(ds))
        #print(z.size())
        return z
    
    def serialize_to_json(self):
        return self.dag.serialize_to_json()
    
    def mutate(self):
        return self.__class__(mutate(self.dag),input_size=self.input_size,output_size=self.output_size)


def get_seed():
    mg = ModuleDag("TestNet", input_size=(64,1,28,28))
    inputmod = ImageModule("InputModule",shapeTransform=(1,1,1))
    inputmod.setShape((64,1,28,28))
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
    return MNISTModuleDagWrapper(mg,input_size=28*28,output_size=10)


class GeneticAlgorithmTrainer:
    def __init__(self, population_size, generations, mutations_per_generation, get_seed):

        self.population_size = population_size
        self.generations = generations
        self.mutations_per_generation = mutations_per_generation
        self.get_seed = get_seed

        self.population = None

    def initialize_population(self):
        self.population = [self.get_seed() for _ in range(self.population_size)]
        print(self.population)

    def mutate(self, item, amt=1):
        with torch.no_grad():
            for _ in range(amt):
                item = item.mutate()
            return item


    def get_scores(self,generation):
        scores = []
        os.mkdir(f"runs/{generation}")
        for individual in self.population:
            uid = str(uuid.uuid4())
            
            os.mkdir(f"runs/{generation}/{uid}")
            with open(f"runs/{generation}/{uid}/model.json", "w+") as f:
                f.write(individual.serialize_to_json())
            
            individual_model = individual#torch.compile(individual)
            individual_model.to("mps")
            val_loss = train_model(individual_model, subdir=f"{generation}/{uid}")
            scores.append(val_loss)
        return scores

    def select_parents(self,generation):

        fitness_scores = np.array(self.get_scores(generation))
        fitness_scores = 1 - fitness_scores
        fitness_scores = fitness_scores - np.min(fitness_scores) + 1e-10
        probabilities = fitness_scores / np.sum(fitness_scores)
        indices = np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)

        return [self.population[i] for i in indices], fitness_scores


    def mutate_population(self):
        self.population = [self.mutate(individual, self.mutations_per_generation) for individual in self.population]

    def run(self):
        self.initialize_population()
        for generation in trange(self.generations):
            print(f"Generation {generation + 1}")
            self.population, scores = self.select_parents(generation)
            self.mutate_population()

            print(f"Best individual fitness: {max(scores)}")
        
    
if __name__ == "__main__":
    trainer = GeneticAlgorithmTrainer(10,10,1,get_seed)
    trainer.run()

