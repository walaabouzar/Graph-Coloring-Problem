import random
import numpy as np

# 1. Génération d'un graphe aléatoire et représentation sous forme de matrice d'adjacence
def generate_random_graph(n, p):
    """Génère un graphe aléatoire avec n nœuds et une probabilité p d'une arête entre chaque paire de nœuds."""
    adj_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1  # Graphe non orienté, donc symétrique
    return adj_matrix

# Affichage de la matrice d'adjacence
def print_adjacency_matrix(adj_matrix):
    """Affiche la matrice d'adjacence de manière lisible."""
    print("Matrice d'adjacence du graphe :")
    for row in adj_matrix:
        print(' '.join(map(str, row)))

# 2. Vérification de la validité de la solution
def is_valid_solution(solution, adj_matrix):
    """Vérifie si la solution (coloration) respecte les contraintes de non-collision des couleurs."""
    n = len(solution)
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1 and solution[i] == solution[j]:
                return False
    return True

# 3. Calcul du fitness de la solution (nombre de couleurs utilisées)
def fitness(solution):
    """Retourne le nombre de couleurs uniques utilisées dans la solution."""
    return len(set(solution))

# 4. Initialisation de la population (génération de solutions aléatoires)
def initialize_population(pop_size, n):
    """Génère une population initiale de solutions aléatoires."""
    population = []
    for _ in range(pop_size):
        solution = [random.randint(1, n) for _ in range(n)]  # Colorier aléatoirement avec n couleurs
        population.append(solution)
    return population

# 5. Sélection par tournoi
def tournament_selection(population, adj_matrix):
    """Sélectionne deux individus de manière aléatoire, en utilisant la méthode du tournoi."""
    tournament_size = 3
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda sol: fitness(sol) if is_valid_solution(sol, adj_matrix) else float('inf'))
    return selected[0], selected[1]

# 6. Croisement (crossover)
def crossover(parent1, parent2):
    """Croisement entre deux parents pour produire une descendance."""
    n = len(parent1)
    crossover_point = random.randint(0, n - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

# 7. Mutation
def mutate(solution, n):
    """Applique une mutation (changer la couleur d'un nœud)"""
    mutation_point = random.randint(0, len(solution) - 1)
    new_color = random.randint(1, n)
    solution[mutation_point] = new_color
    return solution                                                         

# 8. Algorithme génétique pour résoudre le problème de coloration de graphe
def genetic_algorithm(n, p, pop_size=100, generations=1000):
    """Applique un algorithme génétique pour résoudre le problème de coloration de graphe."""
    adj_matrix = generate_random_graph(n, p)
    print_adjacency_matrix(adj_matrix)  # Afficher la matrice d'adjacence du graphe
    population = initialize_population(pop_size, n)
    best_solution = None
    best_fitness = float('inf')
    
    for generation in range(generations):
        # Sélectionner deux parents
        parent1, parent2 = tournament_selection(population, adj_matrix)
        
        # Appliquer le croisement
        child = crossover(parent1, parent2)
        
        # Appliquer la mutation
        if random.random() < 0.1:  # Probabilité de mutation
            child = mutate(child, n)
        
        # Évaluer le fitness du nouvel individu
        if is_valid_solution(child, adj_matrix):
            child_fitness = fitness(child)
            # Si la solution est meilleure, la sauvegarder
            if child_fitness < best_fitness:
                best_fitness = child_fitness
                best_solution = child
        
        # Remplacer un individu de la population par le nouvel individu (élitisme)
        population.append(child)
        population.sort(key=lambda sol: fitness(sol) if is_valid_solution(sol, adj_matrix) else float('inf'))
        population = population[:pop_size]  # Maintenir la taille de la population
    
    return best_solution, best_fitness

# Exemple d'utilisation
n = 5  # Nombre de nœuds
p = 0.4  # Probabilité d'une arête
best_solution, best_fitness = genetic_algorithm(n, p)

print("\nMeilleure solution trouvée :")
print(best_solution)
print("Nombre de couleurs utilisées :", best_fitness)
