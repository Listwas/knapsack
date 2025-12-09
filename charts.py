import matplotlib.pyplot as plt
import os
import glob

class Charts:
    def __init__(self, data_dir='tests', output_dir='charts'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.titleweight'] = 'bold'

    # Pomocnicze funkcje
    def load_history_file(self, filepath):
        generations, values = [], []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    gen, val = line.strip().split('\t')
                    generations.append(int(gen))
                    values.append(int(val))
        except FileNotFoundError:
            print(f"UWAGA: Plik {filepath} nie istnieje, pomijam...")
        return generations, values

    def plot_lines(self, file_label_pairs, title, optimum=None, save_as=None, figsize=(14, 8)):
        plt.figure(figsize=figsize)
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        line_styles = ['-', '--', '-.', ':', '-']

        for i, (filepath, label) in enumerate(file_label_pairs):
            if os.path.exists(filepath):
                gens, vals = self.load_history_file(filepath)
                if gens:
                    plt.plot(gens, vals, linewidth=2, label=label,
                             color=colors[i % len(colors)],
                             linestyle=line_styles[i % len(line_styles)])

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Pokolenie', fontsize=12)
        plt.ylabel('Najlepsza wartość', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='lower right', fontsize=10)

        if optimum is not None:
            plt.axhline(y=optimum, color='black', linestyle='--', linewidth=2, alpha=0.7,
                        label=f'Optimum={optimum}')

        plt.tight_layout()
        if save_as:
            plt.savefig(os.path.join(self.output_dir, save_as),
                        dpi=300, bbox_inches='tight')
        plt.show()

    # Ocena 3.5: mr/cr
    def plot_mr_cr(self):
        print("Ocena 3.5: Wykresy mr/cr")

        datasets = {
            'f1_l-d_kp_10_269': 295,
            'f10_l-d_kp_20_879': 1025
        }

        for dataset, optimum in datasets.items():
            files = glob.glob(os.path.join(
                self.data_dir, f'hist_{dataset}_*_cr*_mr*.txt'))
            file_label_pairs = []
            for file in files:
                # Etykieta np. cr=0.9, mr=0.05
                basename = os.path.basename(
                    file).replace(f'hist_{dataset}_', '')
                label = basename.replace('.txt', '')
                file_label_pairs.append((file, label))
            self.plot_lines(file_label_pairs,
                            title=f'Różne mr/cr - {dataset}',
                            optimum=optimum,
                            save_as=f'wykres_mr_cr_{dataset}.png')

    # Ocena 4.5: selekcja ranking vs ruletka
    def plot_selection_ranking_vs_roulette(self):
        print("Ocena 4.5: Porównanie selekcji rankingowej i ruletkowej")

        datasets = {
            'f1_l-d_kp_10_269': 295,
            'knapPI_1_100_1000_1': 9147
        }

        for dataset, optimum in datasets.items():
            files = [
                (os.path.join(self.data_dir,
                 f'hist_{dataset}_ranking_single_cr0.9_mr0.05.txt'), 'Rankingowa'),
                (os.path.join(self.data_dir,
                 f'hist_{dataset}_roulette_single_cr0.9_mr0.05.txt'), 'Ruletkowa')
            ]
            self.plot_lines(files,
                            title=f'Selekcja rankingowa vs ruletkowa - {dataset}',
                            optimum=optimum,
                            save_as=f'wykres_selekcja_ranking_roulette_{dataset}.png')

    # Ocena 4.5: krzyżowanie jedno- vs dwupunktowe
    def plot_crossover_comparison(self):
        print("Ocena 4.5: Krzyżowanie jedno- vs dwupunktowe")

        datasets = {
            'f1_l-d_kp_10_269': 295,
            'f10_l-d_kp_20_879': 1025
        }

        for dataset, optimum in datasets.items():
            files = [
                (os.path.join(self.data_dir,
                 f'hist_{dataset}_ranking_single_cr0.9_mr0.05.txt'), 'Jednopunktowe'),
                (os.path.join(self.data_dir,
                 f'hist_{dataset}_ranking_double_cr0.9_mr0.05.txt'), 'Dwupunktowe')
            ]
            self.plot_lines(files,
                            title=f'Krzyżowanie jedno- vs dwupunktowe - {dataset}',
                            optimum=optimum,
                            save_as=f'wykres_crossover_{dataset}.png')

    # Ocena 5.0: wszystkie selekcje
    def plot_all_selections(self):
        print("Ocena 5.0: Porównanie wszystkich metod selekcji")

        datasets = {
            'f1_l-d_kp_10_269': 295,
            'knapPI_1_100_1000_1': 9147
        }

        for dataset, optimum in datasets.items():
            files = [
                (os.path.join(self.data_dir,
                 f'hist_{dataset}_ranking_single_cr0.9_mr0.05.txt'), 'Rankingowa'),
                (os.path.join(self.data_dir,
                 f'hist_{dataset}_roulette_single_cr0.9_mr0.05.txt'), 'Ruletkowa'),
                (os.path.join(self.data_dir,
                 f'hist_{dataset}_tournament_single_cr0.9_mr0.05.txt'), 'Turniejowa')
            ]
            self.plot_lines(files,
                            title=f'Wszystkie metody selekcji - {dataset}',
                            optimum=optimum,
                            save_as=f'wykres_all_selections_{dataset}.png')

    # Generowanie wszystkich wykresów
    def generate_all(self):
        self.plot_mr_cr()
        self.plot_selection_ranking_vs_roulette()
        self.plot_crossover_comparison()
        self.plot_all_selections()


if __name__ == "__main__":
    charts = Charts()
    charts.generate_all()
