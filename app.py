import streamlit as st
import pandas as pd

from cinema_es import (
    load_data,
    run_es
)

# ======================================================
# Page Setup
# ======================================================

st.set_page_config(
    page_title="Demand-Based Ticket Pricing (ES)",
    layout="wide"
)

st.title("🎟️ Demand-Based Cinema Ticket Pricing")
st.markdown(
    "Optimize ticket pricing using **Evolution Strategies (ES)** based on customer demand."
)

# ======================================================
# Sidebar – Optimization Mode
# ======================================================

st.sidebar.header("🧠 Optimization Objective")

objective_mode = st.sidebar.radio(
    "Select Optimization Mode",
    options=[
        "Single-objective (Maximize Revenue)",
        "Multi-objective (Revenue + Price Stability)"
    ]
)

is_multi_objective = (
    objective_mode == "Multi-objective (Revenue + Price Stability)"
)

# ======================================================
# Sidebar – ES Parameters
# ======================================================

st.sidebar.header("⚙️ Evolution Strategies Parameters")

population_size = st.sidebar.slider(
    "Population Size (Exploration)",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

num_generations = st.sidebar.slider(
    "Generations (Convergence)",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

mutation_sigma = st.sidebar.slider(
    "Mutation σ (Exploration Strength)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

beta = st.sidebar.slider(
    "β (Demand Price Sensitivity)",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    help="Higher β means demand drops faster as price increases"
)


# ------------------------------------------------------
# Multi-objective ONLY parameter
# ------------------------------------------------------

alpha = None
if is_multi_objective:
    alpha = st.sidebar.slider(
        "α (Price Stability Trade-off)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Higher α enforces stronger price stability"
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
            alpha=alpha
        )
        max_revenue = results["optimal_price"] * results["total_demand"]

    # ==================================================
    # Results
    # ==================================================

    st.subheader("📉 Ticket Price Evolution")

    st.metric(
        label="Optimal Price",
        value=f"{optimal_price:.2f}"
    )

    st.line_chart(
        pd.DataFrame(
            results["price_history"],
            columns=["Ticket Price"]
        )
    )

    st.metric(
        label="Maximum Revenue",
        value=f"{max_revenue:.2f}"
    )
    
    st.metric(
        label="Best Fitness (Objective Value)",
        value=f"{results['best_fitness']:.2f}"
    )

    # Contextual explanation
    if is_multi_objective:
        st.caption(
            f"Multi-objective optimization applied with α = {alpha}. "
            f"Reference price = {results['reference_price']:.2f}"
        )
    else:
        st.caption(
            "Single-objective optimization (revenue maximization only)."
        )
