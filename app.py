import streamlit as st
import pandas as pd

from cinema_es_core import (
    load_data,
    run_es,
    extract_price_table
)

# ======================================================
# Page Setup
# ======================================================

st.set_page_config(
    page_title="Cinema Ticket Pricing Optimization (ES)",
    layout="wide"
)

st.title("🎬 Cinema Ticket Pricing Optimization")
st.markdown(
    "Optimize ticket prices using **Evolution Strategies (ES)**. "
    "Adjust parameters and run the optimizer to observe pricing behavior."
)

# ======================================================
# Sidebar – ES Parameters
# ======================================================

st.sidebar.header("⚙️ Evolution Strategies Parameters")

alpha = st.sidebar.slider(
    "α (Multi-objective trade-off)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
    help="Higher α penalizes price variance more strongly"
)

population_size = st.sidebar.slider(
    "Population Size (Exploration)",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
    help="Larger populations explore more but run slower"
)

num_generations = st.sidebar.slider(
    "Generations (Convergence)",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="More generations allow better convergence"
)

mutation_sigma = st.sidebar.slider(
    "Mutation σ (Exploration Strength)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Higher σ explores more aggressively"
)

use_multi_objective = st.sidebar.checkbox(
    "Enable Multi-objective Optimization",
    value=True
)

# ======================================================
# Dataset
# ======================================================

DATA_PATH = "cinema_hall_ticket_sales.csv"

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

st.success("Dataset loaded successfully")

# ======================================================
# Run Optimization
# ======================================================

if st.button("🚀 Run Optimization", type="primary"):

    with st.spinner("Running Evolution Strategies..."):

        results = run_es(
            df=df,
            num_generations=num_generations,
            population_size=population_size,
            mutation_sigma=mutation_sigma,
            alpha=alpha if use_multi_objective else None,
        )

    # ==================================================
    # Results
    # ==================================================

    st.subheader("📈 Optimization Results")

    st.metric(
        label="Best Fitness Value",
        value=f"{results['best_fitness']:.2f}"
    )

    # Fitness convergence plot
    st.line_chart(
        pd.DataFrame(
            results["fitness_history"],
            columns=["Best Fitness"]
        )
    )

    # Optimized price table
    price_df = extract_price_table(
        results["best_solution"],
        results["seat_types"],
        results["movie_genres"],
        results["category_index_map"]
    )

    st.subheader("💰 Optimized Ticket Prices")
    st.dataframe(price_df, use_container_width=True)

    st.caption(
        "Prices are optimized per Seat Type × Movie Genre combination."
    )
