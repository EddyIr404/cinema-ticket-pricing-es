import numpy as np
import pandas as pd

# ======================================================
# Data Loading & Demand Extraction
# ======================================================

def load_data(csv_path: str):
    """
    Load dataset and extract demand-relevant fields only.
    """
    df = pd.read_csv(csv_path)
    df["Number_of_Person"] = pd.to_numeric(df["Number_of_Person"], errors="coerce")
    df = df.dropna(subset=["Number_of_Person"])
    return df


def extract_total_demand(df):
    """
    Aggregate total demand from dataset.
    """
    return df["Number_of_Person"].sum()


# ======================================================
# Fitness Functions (Demand-Based)
# ======================================================

def compute_fitness(price, total_demand, beta):
    """
    Revenue with price-sensitive demand:
    demand decreases as price increases
    """
    effective_demand = total_demand * np.exp(-beta * price)
    return price * effective_demand


def compute_fitness_multiobjective(
    price, total_demand, alpha, reference_price, beta
):
    revenue = compute_fitness(price, total_demand, beta)
    stability_penalty = (price - reference_price) ** 2
    return revenue - alpha * stability_penalty


# ======================================================
# Evolution Strategies Core
# ======================================================

def run_es(
    df,
    num_generations=50,
    population_size=10,
    offspring_per_parent=2,
    mutation_sigma=1.0,
    price_min=10.99,
    price_max=29.99,
    alpha=None,
    beta=0.05,
):

    """
    Run demand-based Evolution Strategies optimization.
    """

    total_demand = extract_total_demand(df)

    # Use historical mean price as stability reference
    reference_price = df["Ticket_Price"].mean() if "Ticket_Price" in df else (
        price_min + price_max
    ) / 2

    # ----- Initialize population (1D price) -----
    population = np.random.uniform(
        price_min, price_max, size=(population_size, 1)
    )
    population = np.round(population, 2)

    # ----- Initial fitness -----
    if alpha is None:
        fitness_scores = np.array([
            compute_fitness(p[0], total_demand, beta) for p in population
        ])
    else:
        fitness_scores = np.array([
            compute_fitness_multiobjective(
                p[0], total_demand, alpha, reference_price, beta
            )
            for p in population
        ])

    best_fitness_history = []

    best_price_history = []


    # ----- Evolution loop -----
    for _ in range(num_generations):
        offspring = []

        for parent in population:
            for _ in range(offspring_per_parent):
                child_price = parent[0] + np.random.normal(0, mutation_sigma)
                child_price = np.clip(child_price, price_min, price_max)
                offspring.append([round(child_price, 2)])

        offspring = np.array(offspring)

        # Evaluate offspring
        if alpha is None:
            offspring_fitness = np.array([
                compute_fitness(p[0], total_demand, beta) for p in offspring
            ])
        else:
            offspring_fitness = np.array([
                compute_fitness_multiobjective(
                    p[0], total_demand, alpha, reference_price, beta
                )
                for p in offspring
            ])

        # Selection (μ + λ) with softened selection pressure
        combined_population = np.vstack((population, offspring))
        combined_fitness = np.concatenate((fitness_scores, offspring_fitness))

        # --- Add small noise to reduce premature convergence ---
        fitness_noise = np.random.normal(
        loc=0.0,
        scale=0.01 * np.std(combined_fitness),
        size=len(combined_fitness)
        )

        noisy_fitness = combined_fitness + fitness_noise

        best_indices = np.argsort(noisy_fitness)[::-1][:population_size]
        population = combined_population[best_indices]
        fitness_scores = combined_fitness[best_indices]


        best_fitness_history.append(fitness_scores.max())

        best_price_history.append(population[0][0])

    # ----- Final result -----
    best_idx = np.argmax(fitness_scores)

    return {
    "optimal_price": population[best_idx][0],
    "best_fitness": fitness_scores[best_idx],
    "fitness_history": best_fitness_history,  # optional now
    "price_history": best_price_history,       # main focus
    "total_demand": total_demand,
    "reference_price": reference_price,
}

