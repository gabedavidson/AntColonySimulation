from dataclasses import dataclass
import math
import random


@dataclass
class Node:
    id: int


class Graph:
    def __init__(self):
        self.chart = {}

    def insert(self, i):
        if i not in self.chart.values():
            self.chart[i] = {}

    @staticmethod
    def distance(a: list, b: list):
        return round(math.dist(a, b), 2)

    def randomize(self, n, w, h, c_hint: float = .2, connected: bool = False):
        assert n > 2
        coordinates = {}
        for i in range(n):
            node = i
            self.insert(node)
            coordinates[i] = [random.random() * w, random.random() * h]

        # getting distances between every node
        for n1 in self.chart:
            for n2 in self.chart:
                if n2 >= n1:  # switch to n2 >= n1 for easier runtime
                    continue
                try:
                    self.chart[n1][n2] = self.chart[n2][n1]
                except KeyError:
                    self.chart[n1][n2] = self.distance(coordinates[n1], coordinates[n2])

        if connected:
            # assigning edges
            for n1 in range(n):
                for k in range(n):
                    if random.random() <= c_hint:  # as c_hint -> 1, number of edges decreases
                        try:
                            del self.chart[n1][k]
                        except KeyError:
                            continue
                        # del self.chart[k][n1]

            # asserting graph is interconnected (i.e. no subgraphs)
            nodes = []
            for n1 in range(n):
                for n2 in self.chart[n1]:
                    if n2 not in nodes:
                        nodes.append(n2)
            if len(nodes) != n:
                self.randomize(n, w, h, c_hint)
                return

    def print(self):
        for k in self.chart:
            print(k, "===>", self.chart[k])


@dataclass
class Ant:
    curr_node: int  # node the ant starts at
    nodes: list  # nodes the ant has to visit
    pheromones: dict  # the pheromones it deposits on each edge


class Simulator:
    @dataclass
    class Edge:
        start: int
        end: int

    def __init__(self, chart, pop=2.5, iterations=2.5, pheromone_accuracy: int = 3, pheromone_evap_acc_rate: float = 0.01, ignore_disconnected: bool = True, arbitrary_destination: bool = True, clamp_pheromones: bool = True):
        """
        Initializes the simulation variables.
        :param chart: The graph chart
        :param pop: The population of ants such that the total population equals pop * the number of nodes in the graph
        :param iterations: The number of iterations wanted, such that max iterations equals iterations * total ant population
        :param pheromone_accuracy: The number of sig-figs after the decimal of pheromone levels
        :param pheromone_evap_acc_rate: The acceleration rate of evaporation of pheromones
        :param ignore_disconnected: Path through all nodes, regardless of edge present in chart
        :param arbitrary_destination: Stop at the shortest path of all nodes, else stop at shortest path back to
        """

        # assertions
        assert pop >= 1 / len(chart)  # must be at least one Ant

        # pheromone min/max
        self.pheromone_min = 1
        self.pheromone_max = len(chart) * pop * len(chart)

        # main components
        self.chart = chart
        self.ants = [Ant(i % len(self.chart), [n for n in self.chart.keys()], {}) for i in range(round(pop * len(self.chart)))]
        self.pheromones = dict(zip(list(self.chart.keys()), [{list(self.chart[n1].keys())[n2]: self.pheromone_min for n2 in range(len(self.chart[n1]))} for n1 in range(len(self.chart))]))

        # runtime handlers
        self.iters = 0
        self.max_iters = iterations * len(self.ants)

        # edge encoding
        self.edge_encodings = {}
        for a in range(len(self.chart)):
            for b in range(a + 1, len(self.chart)):
                self.edge_encodings[self.encode(a, b)] = [a, b]  # each edge connects two nodes. each edge is given a unique int id

        # settings
        self.ignore_disconnected = ignore_disconnected
        self.arbitrary_destination = arbitrary_destination  # fixme
        self.pheromone_accuracy = pheromone_accuracy
        self.pheromone_evap_acc_rate = pheromone_evap_acc_rate  # fixme check for accuracy
        self.clamp_pheromones = clamp_pheromones

        self.initialize_ants()

    def initialize_ants(self):
        for n1 in range(len(self.chart)):
            self.ants[n1 % len(self.ants)].curr_node = n1

    def simulate(self):
        while self.iters < self.max_iters:
            self._simulate()
            self.iters += 1

    def _simulate(self):
        for ant in self.ants:
            if self.ignore_disconnected:
                ant.nodes = [n for n in self.chart.keys()]
            else:
                ant.nodes = [n for n in self.chart[ant.curr_node].keys()]
            for _ in range(len(self.chart) - 1):
                ant.nodes.remove(ant.curr_node)
                next_node = self.get_next_node(ant)
                ant.pheromones[self.encode(ant.curr_node if ant.curr_node < next_node else next_node, next_node if ant.curr_node < next_node else ant.curr_node)] = self.apply_pheromone(ant.curr_node, next_node)
                ant.curr_node = next_node
        self.update_global_pheromones()

    def apply_pheromone(self, c, n):
        """
        Returns the amount of pheromones that should be applied to an edge
        :param c: The current node
        :param n: The node to travel to
        :return: float
        """
        i1, i2 = c if c > n else n, n if c > n else c
        amount = 1 / (self.chart[i1][i2]) * self.pheromones[i1][i2]  # fixme likely need to slow this down
        evaporation = self.pheromones[i1][i2] * self.pheromone_evap_acc_rate  # fixme need to check impact and accuracy
        return amount - evaporation

    def update_global_pheromones(self):
        for ant in self.ants:
            for edge in ant.pheromones:
                start, end = self.edge_encodings[edge][0], self.edge_encodings[edge][1]
                i1, i2 = start if start > end else end, end if start > end else start
                pheromone_deposit = round(self.pheromones[i1][i2] + ant.pheromones[edge], self.pheromone_accuracy)
                self.pheromones[i1][i2] = (self.pheromone_max if pheromone_deposit > self.pheromone_max else pheromone_deposit if pheromone_deposit > self.pheromone_min else self.pheromone_min) if self.clamp_pheromones else pheromone_deposit

    def get_next_node(self, ant):
        weights = [self.pheromones[n1 if n1 > ant.curr_node else ant.curr_node][ant.curr_node if n1 > ant.curr_node else n1] for n1 in ant.nodes]
        return random.choices(population=ant.nodes, weights=weights, k=1)[0]

    def encode(self, a, b):
        def count(i):
            return len(self.chart) - (i + 1)

        def start(i):
            if i == 0:
                return 1
            return count(i - 1) + start(i - 1)

        assert b >= a  # >= to include node id encoding
        return start(a) + (b - a - 1)

    def results(self, start: int = 0):
        assert 0 <= start < len(self.chart)
        n = 0
        winning_ant = Ant(curr_node=start, nodes=[n for n in self.chart], pheromones={})
        while n < len(self.chart) - 1:
            try:
                winning_ant.nodes.remove(winning_ant.curr_node)
            except ValueError:
                pass
            print(f"At node {winning_ant.curr_node}")
            next_node = self.get_next_node(winning_ant)
            print(f"Travelling to node {next_node} with a distance of {self.chart[winning_ant.curr_node if winning_ant.curr_node > next_node else next_node][next_node if winning_ant.curr_node > next_node else winning_ant.curr_node]} and ph of {self.pheromones[winning_ant.curr_node if winning_ant.curr_node > next_node else next_node][next_node if winning_ant.curr_node > next_node else winning_ant.curr_node]}")
            winning_ant.curr_node = next_node
            n += 1
        print(f"Finished at node {winning_ant.curr_node}.")


graph = Graph()
graph.randomize(n=25, w=15, h=15, c_hint=0.5)
sim = Simulator(graph.chart)

sim.simulate()
sim.results()
