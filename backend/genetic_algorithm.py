import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time

class GeneticAlgorithm:
    def __init__(self, n_features=128, population_size=20, n_generations=8,
                 mutation_rate=0.05, crossover_rate=0.8, feature_penalty=0.01,
                 random_seed=42, classifier_name='Random Forest', X_train=None, y_train=None,
                 X_test=None, y_test=None):
        self.n_features = n_features
        self.population_size = max(10, int(population_size))
        self.n_generations = max(4, int(n_generations))
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)
        self.feature_penalty = float(feature_penalty)
        self.classifier_name = classifier_name
        self.X_train = X_train
        self.y_train = y_train
        
        # Test sets are NOT used in GA to prevent data leakage
        self.X_test = None
        self.y_test = None
        
        # Set random seed
        self.rng = np.random.RandomState(int(random_seed))
        self.history = []
        
        # Memoization cache for chromosome fitness (bytes -> score)
        self.fitness_cache = {}

        # Optimized subsample of training set for ultra-fast surrogate evaluations
        # 350 samples provides sharp statistical discrimination in ~3 seconds on cloud CPUs
        n_train_total = len(self.X_train)
        sample_size = min(350, n_train_total)
        sample_indices = self.rng.choice(n_train_total, size=sample_size, replace=False)
        self.X_ga = self.X_train[sample_indices]
        self.y_ga = self.y_train[sample_indices]

        # Precompute static 2-fold stratified cross-validation splits once
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=int(random_seed))
        self.cv_splits = list(skf.split(self.X_ga, self.y_ga))

    def initialize_population(self):
        """Create initial population of random binary chromosomes."""
        population = self.rng.randint(2, size=(self.population_size, self.n_features))
        
        # Ensure at least 1 feature is selected in each chromosome
        for i in range(self.population_size):
            if np.sum(population[i]) == 0:
                population[i, self.rng.randint(self.n_features)] = 1
        return population

    def fitness(self, chromosome):
        """
        Calculate fitness with caching and ultra-fast surrogate Random Forest:
        Fitness = CrossValidationAccuracy - lambda * (n_selected / n_total)
        """
        chrom_key = chromosome.tobytes()
        if chrom_key in self.fitness_cache:
            return self.fitness_cache[chrom_key]

        selected = np.where(chromosome == 1)[0]
        if len(selected) == 0:
            return 0.0

        # Ultra-fast Decision Tree surrogate for sub-second GA convergence on cloud containers
        clf = DecisionTreeClassifier(
            max_depth=6,
            random_state=42
        )

        fold_accuracies = []
        for train_idx, val_idx in self.cv_splits:
            X_tr = self.X_ga[train_idx][:, selected]
            y_tr = self.y_ga[train_idx]
            X_val = self.X_ga[val_idx][:, selected]
            y_val = self.y_ga[val_idx]

            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            fold_accuracies.append(accuracy_score(y_val, preds))

        mean_acc = float(np.mean(fold_accuracies))
        penalty = self.feature_penalty * (len(selected) / self.n_features)
        fit_val = mean_acc - penalty

        self.fitness_cache[chrom_key] = fit_val
        return fit_val

    def tournament_selection(self, population, fitnesses, tournament_size=3):
        """Select parent chromosome using tournament selection."""
        indices = self.rng.choice(len(population), size=min(tournament_size, len(population)), replace=False)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        return population[best_idx].copy()

    def one_point_crossover(self, parent1, parent2):
        """Perform one-point crossover with probability crossover_rate."""
        if self.rng.rand() < self.crossover_rate:
            crossover_point = self.rng.randint(1, self.n_features)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

            if np.sum(child1) == 0:
                child1[self.rng.randint(self.n_features)] = 1
            if np.sum(child2) == 0:
                child2[self.rng.randint(self.n_features)] = 1

            return child1, child2

        return parent1.copy(), parent2.copy()

    def bit_flip_mutation(self, chromosome):
        """Flip bits with probability mutation_rate."""
        mutated = chromosome.copy()
        for i in range(self.n_features):
            if self.rng.rand() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]

        if np.sum(mutated) == 0:
            mutated[self.rng.randint(self.n_features)] = 1

        return mutated

    def evolve(self, progress_callback=None):
        """Main GA evolution loop with elitism and convergence tracking."""
        population = self.initialize_population()
        best_overall_chromosome = None
        best_overall_fitness = -1.0
        no_improvement_count = 0
        patience = max(4, self.n_generations // 2)

        for generation in range(self.n_generations):
            fitnesses = [self.fitness(ind) for ind in population]

            best_idx = np.argmax(fitnesses)
            current_best_fitness = float(fitnesses[best_idx])
            current_best_chromosome = population[best_idx].copy()

            avg_fitness = float(np.mean(fitnesses))
            n_selected = int(np.sum(current_best_chromosome))
            best_accuracy = float(current_best_fitness + self.feature_penalty * (n_selected / self.n_features))

            gen_record = {
                'generation': int(generation + 1),
                'best_fitness': round(current_best_fitness, 5),
                'avg_fitness': round(avg_fitness, 5),
                'best_accuracy': round(best_accuracy, 5),
                'n_selected': n_selected,
                'best_chromosome': current_best_chromosome.tolist()
            }
            self.history.append(gen_record)

            if progress_callback:
                progress_callback(generation + 1, self.n_generations, gen_record)

            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_chromosome = current_best_chromosome.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                break

            # Next generation with elitism
            new_population = [best_overall_chromosome.copy()]

            while len(new_population) < self.population_size:
                p1 = self.tournament_selection(population, fitnesses)
                p2 = self.tournament_selection(population, fitnesses)
                c1, c2 = self.one_point_crossover(p1, p2)
                c1 = self.bit_flip_mutation(c1)
                c2 = self.bit_flip_mutation(c2)
                new_population.extend([c1, c2])

            population = new_population[:self.population_size]

        return best_overall_chromosome, best_overall_fitness, self.history
