from moduledag import ModuleDag, ImageModule
from modules import *
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
from copy import deepcopy
from display import save_graph
from torchinfo import summary
from visualize_generation import reInit, onLoop

os.system(f"caffeinate -is -w {os.getpid()} &")


def multTuple(ta, b):
    return ta[0], ta[1], int(ta[2] * b), int(ta[3] * b)


class MNISTModuleDagWrapper(nn.Module):
    def __init__(self, dag, input_size, output_size):
        super().__init__()
        self.dag = dag
        self.block = nn.Sequential(nn.Flatten(), nn.Linear(input_size, output_size))
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        x = torch.Tensor(x)
        # print(x.size())
        d = self.dag(x)
        # print(len(d))
        ds = torch.stack(d)
        ds = ds.squeeze(0)
        # print(ds.size())
        # print("^^^ d")
        z = self.block(torch.squeeze(ds))
        # print(z.size())
        return z

    def serialize_to_json(self):
        return self.dag.serialize_to_json()

    def mutate(self):
        return self.__class__(
            mutate(self.dag), input_size=self.input_size, output_size=self.output_size
        )


class MNISTBlockModuleDagWrapper(nn.Module):
    def __init__(self, dag, input_size, output_size, num_blocks=2):
        super().__init__()

        self.num_blocks = num_blocks

        self.dag = dag

        self.dags = []
        final_outsize = multTuple(input_size, 1 / (2**num_blocks))
        final_outsize = final_outsize[-2] * final_outsize[-1]
        for i in range(num_blocks):
            ndag = deepcopy(dag)
            ndag.input_size = multTuple(input_size, 1 / (2**i))
            ndag.output_size = multTuple(input_size, 1 / (2 ** (i + 1)))
            ndag.modules_moduledag["InputModule"].setShape(
                multTuple(input_size, 1 / (2**i))
            )

            self.add_module(f"dag_{i}", ndag)
            self.dags.append(ndag)

        self.dagblock = nn.ModuleList(self.dags)
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_outsize, output_size),
        )
        self.input_size = input_size
        self.output_size = output_size
        self.num_blocks = num_blocks

    def forward(self, x):
        x = torch.Tensor(x)
        # print(x.size())

        d = x
        for a in self.dagblock:
            d = a(d)
        # print(len(d))
        ds = torch.stack(d)
        ds = ds.squeeze(0)
        # print(ds.size())
        # print("^^^ d")
        z = self.block(torch.squeeze(ds))
        # print(z.size())
        return z

    def serialize_to_json(self):
        return self.dag.serialize_to_json()

    def mutate(self):
        return self.__class__(
            mutate(self.dag),
            input_size=self.input_size,
            output_size=self.output_size,
            num_blocks=self.num_blocks,
        )


def get_block_seed():
    mg = ModuleDag("TestNet", input_size=(64, 1, 28, 28), output_size=(64, 1, 14, 14))
    inputmod = ImageModule("InputModule", shapeTransform=(1, 1, 1))
    inputmod.setShape((64, 1, 28, 28))
    inputmod.init_block()
    mg.add_module(inputmod)
    mg.add_module(SimpleConv("conv1"))
    mg.add_module(ReLU("relu1"))
    mg.add_module(ConvDown("convdown"))
    mg.add_module(ReLU("relu2"))
    mg.add_module(Module("OutputModule", num_inputs=1, num_outputs=1))

    mg.add_connection("InputModule", "conv1")

    mg.add_connection("conv1", "relu1")
    mg.add_connection("relu1", "convdown")
    mg.add_connection("convdown", "relu2")
    mg.add_connection("relu2", "OutputModule")

    mg.validate_graph()
    return MNISTBlockModuleDagWrapper(mg, input_size=(64, 1, 28, 28), output_size=10)


def get_seed():
    mg = ModuleDag("TestNet", input_size=(64, 1, 28, 28), output_size=(64, 1, 28, 28))
    inputmod = ImageModule("InputModule", shapeTransform=(1, 1, 1))
    inputmod.setShape((64, 1, 28, 28))
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
    return MNISTModuleDagWrapper(mg, input_size=28 * 28, output_size=10)


def get_down_seed():
    mg = ModuleDag("TestNet", input_size=(64, 1, 28, 28), output_size=(64, 4, 7, 7))
    inputmod = ImageModule("InputModule", shapeTransform=(1, 1, 1))
    inputmod.setShape((64, 1, 28, 28))
    inputmod.init_block()
    mg.add_module(inputmod)
    mg.add_module(SimpleConv("conv1"))
    mg.add_module(ConvDown("cd1"))
    mg.add_module(Conv211("c21"))
    mg.add_module(ReLU("relu1"))

    mg.add_module(SimpleConv("conv2"))
    mg.add_module(ConvDown("cd2"))
    mg.add_module(Conv211("c22"))
    mg.add_module(ReLU("relu2"))
    mg.add_module(Module("OutputModule", num_inputs=1, num_outputs=1))

    mg.add_connection("InputModule", "conv1")

    mg.add_connection("conv1", "cd1")
    mg.add_connection("cd1", "c21")
    mg.add_connection("c21", "relu1")
    mg.add_connection("relu1", "conv2")
    mg.add_connection("conv2", "cd2")
    mg.add_connection("cd2", "c22")
    mg.add_connection("c22", "relu2")

    mg.add_connection("relu2", "OutputModule")

    mg.validate_graph()
    return MNISTModuleDagWrapper(mg, input_size=4 * 7 * 7, output_size=10)


class GeneticAlgorithmTrainer:
    def __init__(
        self, population_size, generations, mutations_per_generation, get_seed
    ):

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

    def get_scores(self, generation):
        scores = []
        uids = []
        os.mkdir(f"runs/{generation}")
        reInit()
        for individual in self.population:
            uid = str(uuid.uuid4())
            uids.append(uid)

            os.mkdir(f"runs/{generation}/{uid}")
            with open(f"runs/{generation}/{uid}/model.json", "w+") as f:
                f.write(individual.serialize_to_json())

            model_summary = summary(individual, (64, 1, 28, 28)).__str__()
            with open(f"runs/{generation}/{uid}/summary.txt", "w+") as f:
                f.write(model_summary)

            save_graph(individual.dag.graph, f"runs/{generation}/{uid}/graph.png")

            individual_model = individual  # torch.compile(individual)
            individual_model.to("mps")
            val_loss, val_acc = train_model(
                individual_model, subdir=f"{generation}/{uid}"
            )
            scores.append(val_acc.item())
            onLoop(f"runs/{generation}/{uid}/graph.png", val_acc.item())
        return scores, uids

    def select_parents(self, generation):  # From chatgpt and you can tell
        a, uids = self.get_scores(generation)
        fitness_scores = np.array(a)
        real_scores = np.copy(fitness_scores)
        fitness_scores = 1 - fitness_scores
        fitness_scores = fitness_scores - np.min(fitness_scores) + 1e-10
        probabilities = fitness_scores / np.sum(fitness_scores)
        indices = np.random.choice(
            range(self.population_size), size=self.population_size, p=probabilities
        )

        return [self.population[i] for i in indices], fitness_scores, real_scores, uids

    def mutate_population(self):
        self.population = [
            self.mutate(individual, self.mutations_per_generation)
            for individual in self.population
        ]

    def run(self):
        self.initialize_population()
        self.mutate_population()
        for generation in trange(self.generations):
            print(f"Generation {generation + 1}")
            self.population, scores, rscores, uids = self.select_parents(generation)
            self.mutate_population()

            msg = f"Best individual fitness: {np.max(rscores)} ({np.max(scores)}) ({uids[np.argmax(rscores)]}) | Average fitness: {np.mean(rscores)} ({np.mean(scores)})"
            print(msg)
            with open(f"runs/{generation}/best.txt", "w+") as f:
                f.write(msg)


if __name__ == "__main__":
    trainer = GeneticAlgorithmTrainer(10, 10, 1, get_down_seed)
    trainer.run()
