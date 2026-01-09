import numpy as np
import pandas as pd

# ======================================================
# Data Loading & Preprocessing
# ======================================================

def load_data(csv_path: str):
    """
    Load and preprocess cinema ticket dataset.
    """
    df = pd.read_csv(csv_path)
    df["Number_of_Person"] = pd.to_numeric(df["Number_of_Person"], errors="coerce")
    df = df.dropna(subset=["Number_of_Person"])
    return df


def build_category_index(df):
    """
    Build mapping from (Seat_Type, Movie_Genre) to vector index.
    """
    seat_types = df["Seat_Type"].unique()
    movie_genres = df["Movie_Genre"].unique()

    category_index_map = {}
    idx = 0
    for seat in seat_types:
        for genre in movie_genres:
            category_index_map[(seat, genre)] = idx
            idx += 1

    return seat_types, movie_genres, category_index_map


# ======================================================
# Fitness Functions
# ======================================================

def compute_fitness(candidate, df, category_index_map):
    """
    Single-objective fitness: maximize total revenue.
    """
    revenue = 0.0
    for _, row in df.iterrows():
        idx = category_index_map[(row["Seat_Type"], row["Movie_Genre"])]
        revenue += candidate[idx] * row["Number_of_Person"]
    return revenue


def compute_fitness_multiobjective(candidate, df, category_index_map, alpha):
    """
    Multi-objective fitness:
    Maximize revenue, minimize price variance.
    """
    revenue = compute_fitness(candidate, df, category_index_map)
    price_variance = np.var(candidate)
    return revenue - alpha * price_variance


# ======================================================
# Evolution Strategies Core
# ======================================================

def run_es(
    df,
    num_generations=50,
    population_size=10,
    offspring_per_parent=2,
    mutation_sigma=1.0,
    price_min=10.01,
    price_max=24.99,
    alpha=None,
):
    """
    Run Evolution Strategies optimization.

    If alpha is None → revenue-only optimization
    If alpha is provided → multi-objective optimization
    """

    seat_types, movie_genres, category_index_map = build_category_index(df)
    num_variables = len(category_index_map)

    # ----- Initialize population -----
    population = np.random.uniform(
        price_min, price_max, size=(population_size, num_variables)
    )
    population = np.round(population, 2)

    # ----- Initial fitness -----
    if alpha is None:
        fitness_scores = np.array([
            compute_fitness(c, df, category_index_map) for c in population
        ])
    else:
        fitness_scores = np.array([
            compute_fitness_multiobjective(c, df, category_index_map, alpha)
            for c in population
        ])

    best_fitness_history = []

    # ----- Evolution loop -----
    for _ in range(num_generations):
        offspring = []

        for parent in population:
            for _ in range(offspring_per_parent):
                child = parent + np.random.normal(
                    0, mutation_sigma, size=num_variables
                )
                child = np.clip(child, price_min, price_max)
                child = np.round(child, 2)
                offspring.append(child)

        offspring = np.array(offspring)

        # Evaluate offspring
        if alpha is None:
            offspring_fitness = np.array([
                compute_fitness(c, df, category_index_map) for c in offspring
            ])
        else:
            offspring_fitness = np.array([
                compute_fitness_multiobjective(c, df, category_index_map, alpha)
                for c in offspring
            ])

        # Selection
        combined_population = np.vstack((population, offspring))
        combined_fitness = np.concatenate((fitness_scores, offspring_fitness))

        best_indices = np.argsort(combined_fitness)[::-1][:population_size]
        population = combined_population[best_indices]
        fitness_scores = combined_fitness[best_indices]

        best_fitness_history.append(fitness_scores.max())

    # ----- Final results -----
    best_idx = np.argmax(fitness_scores)
    best_solution = population[best_idx]
    best_fitness = fitness_scores[best_idx]

    return {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "fitness_history": best_fitness_history,
        "seat_types": seat_types,
        "movie_genres": movie_genres,
        "category_index_map": category_index_map,
    }


# ======================================================
# Output Formatting (for Streamlit display)
# ======================================================

def extract_price_table(best_solution, seat_types, movie_genres, category_index_map):
    """
    Convert optimized price vector into a readable table.
    """
    rows = []
    for seat in seat_types:
        for genre in movie_genres:
            idx = category_index_map[(seat, genre)]
            rows.append([seat, genre, best_solution[idx]])

    return pd.DataFrame(
        rows,
        columns=["Seat_Type", "Movie_Genre", "Optimized_Price"]
    )
