import random
import time
import sys
import os


class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight


# Napisać ogólny skrypt realizujący ewolucję algorytmu genetycznego. Parametrami dla skryptu powinny być m.in.: rozmiar populacji, liczba iteracji, funkcja przystosowania, funkcja selekcji, funkcje krzyżowania, funkcja mutacji.
class KnapsackGA:
    def __init__(self,
                 items,
                 capacity,
                 population_size=100,
                 generations=400,
                 crossover_rate=0.9,
                 mutation_rate=0.05,
                 selection="roulette",      # "roulette", "ranking", "tournament"
                 crossover="single",        # "single", "double"
                 tournament_size=3,         
                 use_repair=False):         # toggle repair for hard instances
        self.items = items
        self.capacity = capacity
        self.pop_size = population_size
        self.generations = generations
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.selection = selection
        self.crossover = crossover
        self.tournament_size = tournament_size
        self.use_repair = use_repair

        self.n = len(items)
        self.population = [self._random_chromosome() for _ in range(population_size)]
        self.best_history = []

    def _random_chromosome(self):
        p_one = min(0.05, 5.0 / self.n)  
        return [1 if random.random() < p_one else 0 for _ in range(self.n)]

    # Napisać funkcję przystosowania odpowiednią dla problemu plecakowego
    def fitness(self, chromosome):
        total_value = 0
        total_weight = 0
        for i in range(self.n):
            if chromosome[i]:
                total_weight += self.items[i].weight
                total_value += self.items[i].value
        return total_value if total_weight <= self.capacity else 0

    # Funkcja naprawcza do duzej ilosci danych
    def _repair(self, chromosome):
        if self.fitness(chromosome) > 0:
            return
        selected = [i for i in range(self.n) if chromosome[i]]
        total_weight = sum(self.items[i].weight for i in selected)
        while total_weight > self.capacity and selected:
            remove_idx = random.choice(selected)
            chromosome[remove_idx] = 0
            total_weight -= self.items[remove_idx].weight
            selected.remove(remove_idx)

    # Napisać funkcję realizującą selekcję ruletkową
    def _select_roulette(self):
        fitnesses = [self.fitness(chrom) for chrom in self.population]
        total = sum(fitnesses)
        if total == 0:
            return random.choice(self.population)
        pick = random.uniform(0, total)
        current = 0
        for chrom, fit in zip(self.population, fitnesses):
            current += fit
            if current >= pick:
                return chrom
        return self.population[-1]

    # Napisać funkcję realizującą selekcję rankingową
    def _select_ranking(self):
        sorted_pop = sorted(self.population, key=self.fitness, reverse=True)
        ranks = list(range(self.pop_size, 0, -1))
        total_rank = sum(ranks)
        pick = random.uniform(0, total_rank)
        current = 0
        for rank, chrom in zip(ranks, sorted_pop):
            current += rank
            if current >= pick:
                return chrom
        return sorted_pop[0]

    # Napisać funkcję realizującą selekcję turniejową
    def _select_tournament(self):
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=self.fitness)

    def _select_parent(self):
        if self.selection == "roulette":
            return self._select_roulette()
        elif self.selection == "ranking":
            return self._select_ranking()
        elif self.selection == "tournament":
            return self._select_tournament()
        else:
            raise ValueError("Nieznana metoda selekcji")

    # Napisać funkcję realizującą jednopunktowy operator krzyżowania
    def _crossover_single(self, p1, p2):
        if random.random() > self.cr:
            return p1[:]
        point = random.randint(1, self.n - 1)
        return p1[:point] + p2[point:]

    # Napisać funkcję realizującą krzyżowanie dwupunktowe
    def _crossover_double(self, p1, p2):
        if random.random() > self.cr:
            return p1[:]
        pt1 = random.randint(1, self.n - 2)
        pt2 = random.randint(pt1 + 1, self.n - 1)
        return p1[:pt1] + p2[pt1:pt2] + p1[pt2:]

    def _crossover(self, p1, p2):
        if self.crossover == "single":
            return self._crossover_single(p1, p2)
        else:
            return self._crossover_double(p1, p2)

    # Napisać funkcję mutacji
    def _mutate(self, chromosome):
        for i in range(self.n):
            if random.random() < self.mr:
                chromosome[i] = 1 - chromosome[i]

    # GŁÓWNA PĘTLA
    def run(self):
        if self.use_repair:
            for chrom in self.population:
                self._repair(chrom)

        for gen in range(self.generations):
            new_population = []
            for _ in range(self.pop_size):
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                if self.use_repair:
                    self._repair(child)
                new_population.append(child)

            best_old = max(self.population, key=self.fitness)
            self.population = new_population
            worst_new = min(self.population, key=self.fitness)
            if self.fitness(worst_new) < self.fitness(best_old):
                self.population[self.population.index(worst_new)] = best_old

            best_value = self.fitness(best_old)
            self.best_history.append(best_value)

            if gen % 50 == 0 or gen == self.generations - 1:
                print(f"Pokolenie {gen:4d} | Najlepsza wartość: {best_value}")

    def get_best(self):
        best = max(self.population, key=self.fitness)
        return best, self.fitness(best)


# Napisać funkcję importującą dane do problemu plecakowego z pliku tekstowego
def load_knapsack_file(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n, capacity = map(int, lines[0].split())
    items = []
    for line in lines[1:]:
        v, w = map(int, line.split())
        items.append(Item(v, w))

    if len(items) != n:
        print(f"BŁĄD: oczekiwano {n} przedmiotów, wczytano {len(items)}")
        sys.exit(1)

    return capacity, items, n


# Wczytywanie optimum
def load_optimum(opt_path):
    with open(opt_path, "r") as f:
        return int(f.read().strip())


def get_optimum_path(data_path):
    if "low-dimensional" in data_path:
        return data_path.replace("low-dimensional", "low-dimensional-optimum")
    elif "large_scale" in data_path:
        return data_path.replace("large_scale", "large_scale-optimum")
    else:
        raise ValueError("Nieznany typ zbioru danych")


# Uruchamianie eksperymentu 
def run_experiment(data_path, selection="roulette", crossover="single", cr=0.9, mr=0.05, 
                   pop_size=100, gens=400, tournament_size=3, use_repair=False):
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"START: {data_path}")
    print(f"Parametry: sel={selection}, cross={crossover}, cr={cr}, mr={mr}, pop={pop_size}, gens={gens}, repair={use_repair}")
    print(f"{'='*60}")
    
    opt_path = get_optimum_path(data_path)
    optimum = load_optimum(opt_path)

    capacity, items, n = load_knapsack_file(data_path)

    ga = KnapsackGA(
        items=items,
        capacity=capacity,
        population_size=pop_size,
        generations=gens,
        crossover_rate=cr,
        mutation_rate=mr,
        selection=selection,
        crossover=crossover,
        tournament_size=tournament_size,
        use_repair=use_repair
    )

    ga.run()

    solution, value = ga.get_best()
    used_weight = sum(items[i].weight for i in range(n) if solution[i])
    percent_opt = (value / optimum * 100) if optimum > 0 else 0

    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print(f"CZAS WYKONANIA: {elapsed:.1f} sekund ({elapsed/60:.1f} minut)")
    print(f"NAJLEPSZE: {value} / Optimum: {optimum} ({percent_opt:.2f}%)")
    print(f"Waga: {used_weight}/{capacity}")
    print("="*50)

    # Zapisz historię
    base_name = os.path.basename(data_path)
    hist_file = f"hist_{base_name}_{selection}_{crossover}_cr{cr}_mr{mr}.txt"
    with open(hist_file, "w") as f:
        for gen, val in enumerate(ga.best_history):
            f.write(f"{gen}\t{val}\n")
    print(f"Zapisano: {hist_file}\n")

if __name__ == "__main__":
    base = "daneAG/"

    # Małe zbiory
    run_experiment(base + "low-dimensional/f1_l-d_kp_10_269",  selection="ranking",    mr=0.01, cr=0.9, pop_size=100, gens=500)
    run_experiment(base + "low-dimensional/f1_l-d_kp_10_269",  selection="ranking",    mr=0.05, cr=0.9, pop_size=100, gens=500)
    run_experiment(base + "low-dimensional/f1_l-d_kp_10_269",  selection="ranking",    mr=0.10, cr=0.9, pop_size=100, gens=500)
    run_experiment(base + "low-dimensional/f1_l-d_kp_10_269",  selection="roulette",   mr=0.05, cr=0.9, pop_size=100, gens=500)
    run_experiment(base + "low-dimensional/f1_l-d_kp_10_269",  selection="tournament",tournament_size=5, mr=0.05, cr=0.9, pop_size=100, gens=500)

    run_experiment(base + "low-dimensional/f10_l-d_kp_20_879", selection="ranking",    mr=0.01, cr=0.9, pop_size=100, gens=600)
    run_experiment(base + "low-dimensional/f10_l-d_kp_20_879", selection="ranking",    mr=0.05, cr=0.9, pop_size=100, gens=600)
    run_experiment(base + "low-dimensional/f10_l-d_kp_20_879", selection="ranking",    mr=0.10, cr=0.9, pop_size=100, gens=600)

    # Duże zbiory 
    run_experiment(base + "large_scale/knapPI_1_100_1000_1", selection="ranking",    mr=0.05, cr=0.9, pop_size=300, gens=1000, use_repair=True)
    run_experiment(base + "large_scale/knapPI_1_100_1000_1", selection="roulette",   mr=0.05, cr=0.9, pop_size=300, gens=1000, use_repair=True)
    run_experiment(base + "large_scale/knapPI_1_100_1000_1", selection="tournament",tournament_size=5, mr=0.05, cr=0.9, pop_size=300, gens=1000, use_repair=True)

    run_experiment(base + "large_scale/knapPI_2_100_1000_1", selection="ranking",    mr=0.05, cr=0.9, pop_size=300, gens=1000, use_repair=True)
    run_experiment(base + "large_scale/knapPI_2_100_1000_1", selection="tournament",tournament_size=5, mr=0.05, cr=0.9, pop_size=300, gens=1000, use_repair=True)
