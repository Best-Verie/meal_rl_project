import streamlit as st
import numpy as np
from stable_baselines3 import PPO
from environment.custom_env import KitchenMealPlanningEnv
import pandas as pd

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Meal Planning AI", layout="wide")

st.title(" Meal Planning AI")
st.markdown("Generate optimized meals using Reinforcement Learning")

# ---------------------------
# LOAD MODEL (cached)
# ---------------------------
@st.cache_resource
def load_model():
    return PPO.load("best_models/ppo/best_model.zip")

model = load_model()

# ---------------------------
# LOAD ENV (for metadata)
# ---------------------------
@st.cache_resource
def load_env():
    return KitchenMealPlanningEnv()

env_meta = load_env()

# ---------------------------
# SIDEBAR INPUTS (NO HARDCODING)
# ---------------------------
st.sidebar.header("⚙️ Settings")

scenario_names = [s["name"] for s in env_meta.scenarios]

scenario = st.sidebar.selectbox(
    "Select Scenario",
    scenario_names
)

max_steps = st.sidebar.slider("Max Steps", 3, 15, 6)
show_steps = st.sidebar.checkbox("Show step-by-step decisions", True)

# ---------------------------
# ACTION INTERPRETER
# ---------------------------
def interpret_action(action, info):
    names = info["ingredient_names"]
    n = len(names)

    if action < n:
        return f"ADD {names[action]}"
    elif action < 2 * n:
        return f"REMOVE {names[action - n]}"
    else:
        return "DONE"

# ---------------------------
# GENERATE MEAL FUNCTION
# ---------------------------
def generate_meal():
    env = KitchenMealPlanningEnv()

    obs, info = env.reset(options={"scenario_name": scenario})
    done, truncated = False, False

    rewards = []
    total_reward = 0
    step_data = []

    step = 0

    while not (done or truncated) and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(int(action))

        action_label = interpret_action(int(action), info)

        rewards.append(float(reward))
        total_reward += reward

        step_data.append({
            "Step": step + 1,
            "Action": action_label,
            "Reward": float(reward)
        })

        step += 1

    #  FINAL STATE (IMPORTANT)
    final_quantities = info["quantities"]
    ingredient_names = info["ingredient_names"]

    final_meal = [
        f"{ingredient_names[i]} x{q}"
        for i, q in enumerate(final_quantities)
        if q > 0
    ]

    nutrition = info.get("nutrition", {})

    return final_meal, rewards, total_reward, nutrition, step_data


# ---------------------------
# MAIN BUTTON
# ---------------------------
if st.button("🚀 Generate Meal Plan"):

    final_meal, rewards, total_reward, nutrition, step_data = generate_meal()

    if not final_meal:
        st.warning("No meal generated.")
    else:
        col1, col2 = st.columns(2)

        # ---------------------------
        # FINAL MEAL (CORRECT)
        # ---------------------------
        with col1:
            st.subheader("🥗 Final Meal Composition")
            for item in final_meal:
                st.markdown(f"• 🟢 **{item.replace('_', ' ').title()}**")

        # ---------------------------
        # NUTRITION
        # ---------------------------
        with col2:
            st.subheader("📊 Nutrition Summary")

            if nutrition:
                df = pd.DataFrame(nutrition.items(), columns=["Nutrient", "Value"])
                st.dataframe(df)

                # smarter feedback
                if nutrition["sodium"] < 400:
                    st.success("✅ Low Sodium")
                else:
                    st.error("⚠️ High Sodium")

                if nutrition["protein"] > 25:
                    st.success("💪 High Protein Meal")

            else:
                st.warning("No nutrition data available")

        # ---------------------------
        # REWARD
        # ---------------------------
        st.subheader("🏆 Total Reward")
        st.success(f"{round(total_reward, 2)}")

        # ---------------------------
        # STEP-BY-STEP
        # ---------------------------
        if show_steps:
            st.subheader("🧠 Decision Process")
            st.dataframe(pd.DataFrame(step_data))

        # ---------------------------
        # REWARD CHART
        # ---------------------------
        st.subheader("📈 Reward per Step")
        st.line_chart(rewards)