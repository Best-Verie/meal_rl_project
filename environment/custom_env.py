import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environment.rendering import render_meal_state



class KitchenMealPlanningEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=20):
        super().__init__()
        self.max_steps = max_steps

        self.ingredients = [
            {"name": "rice", "cal": 130, "protein": 2.7, "carbs": 28.0, "fat": 0.3, "fiber": 0.4, "sodium": 1},
            {"name": "beans", "cal": 127, "protein": 8.7, "carbs": 22.8, "fat": 0.5, "fiber": 6.4, "sodium": 2},
            {"name": "chicken", "cal": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "fiber": 0.0, "sodium": 74},
            {"name": "oil", "cal": 119, "protein": 0.0, "carbs": 0.0, "fat": 13.5, "fiber": 0.0, "sodium": 0},
            {"name": "cabbage", "cal": 25, "protein": 1.3, "carbs": 5.8, "fat": 0.1, "fiber": 2.5, "sodium": 18},
            {"name": "carrot", "cal": 41, "protein": 0.9, "carbs": 10.0, "fat": 0.2, "fiber": 2.8, "sodium": 69},
            {"name": "banana", "cal": 89, "protein": 1.1, "carbs": 23.0, "fat": 0.3, "fiber": 2.6, "sodium": 1},
            {"name": "soy_sauce", "cal": 9, "protein": 1.3, "carbs": 0.8, "fat": 0.0, "fiber": 0.1, "sodium": 879},
        ]

        self.num_ingredients = len(self.ingredients)
        self.max_quantity = 5

        self.goals = ["weight_loss", "weight_gain", "maintenance", "baby_safe"]
        self.conditions = ["none", "hypertension", "baby"]
        self.age_groups = ["adult", "infant"]

        self.scenarios = [
            {
                "name": "adult_weight_loss",
                "goal": "weight_loss",
                "condition": "none",
                "age_group": "adult",
                "target_calories": 400,
                "max_sodium": 500,
                "max_items": 6,
            },
            {
                "name": "adult_weight_gain",
                "goal": "weight_gain",
                "condition": "none",
                "age_group": "adult",
                "target_calories": 700,
                "max_sodium": 700,
                "max_items": 7,
            },
            {
                "name": "adult_hypertension",
                "goal": "maintenance",
                "condition": "hypertension",
                "age_group": "adult",
                "target_calories": 500,
                "max_sodium": 400,
                "max_items": 5,
            },
            {
                "name": "baby_meal",
                "goal": "baby_safe",
                "condition": "baby",
                "age_group": "infant",
                "target_calories": 250,
                "max_sodium": 200,
                "max_items": 4,
            },
        ]

        self.action_space = spaces.Discrete(self.num_ingredients * 2 + 1)

        obs_size = (
            self.num_ingredients
            + 6
            + len(self.goals)
            + len(self.conditions)
            + len(self.age_groups)
            + 2
            + 1
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.quantities = None
        self.current_step = 0
        self.last_score = 0.0
        self.current_scenario = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.quantities = np.zeros(self.num_ingredients, dtype=np.int32)
        self.current_step = 0

        if options is not None and "scenario_name" in options:
            scenario_name = options["scenario_name"]
            matched = [s for s in self.scenarios if s["name"] == scenario_name]
            self.current_scenario = matched[0] if matched else self.scenarios[np.random.randint(len(self.scenarios))]
        else:
            self.current_scenario = self.scenarios[np.random.randint(len(self.scenarios))]

        self.last_score = self._meal_score()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False

        old_quantities = self.quantities.copy()

        if action < self.num_ingredients:
            idx = action
            if self.quantities[idx] < self.max_quantity:
                self.quantities[idx] += 1
        elif action < self.num_ingredients * 2:
            idx = action - self.num_ingredients
            if self.quantities[idx] > 0:
                self.quantities[idx] -= 1
        else:
            terminated = True

        new_score = self._meal_score()
        reward = new_score - self.last_score
        self.last_score = new_score

        # small step cost
        reward -= 0.05

        # punish useless action
        if np.array_equal(old_quantities, self.quantities) and not terminated:
            reward -= 0.20

        # punish hovering around a near-optimal meal instead of finalizing
        n = self._nutrition()
        sc = self.current_scenario
        total_items = int(np.sum(self.quantities))

        near_target_calories = abs(n["cal"] - sc["target_calories"]) < 50
        within_sodium = n["sodium"] <= sc["max_sodium"]
        valid_portion = total_items <= sc["max_items"]

        if not terminated and near_target_calories and within_sodium and valid_portion:
            reward -= 0.20

        if terminated:
            reward += self._final_bonus()

        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 1.0

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _get_obs(self):
        n = self._nutrition()
        sc = self.current_scenario

        quantities_norm = self.quantities / self.max_quantity
        nutrition_norm = np.array([
            min(n["cal"] / 1000.0, 1.0),
            min(n["protein"] / 60.0, 1.0),
            min(n["carbs"] / 150.0, 1.0),
            min(n["fat"] / 60.0, 1.0),
            min(n["fiber"] / 30.0, 1.0),
            min(n["sodium"] / 2000.0, 1.0),
        ], dtype=np.float32)

        goal_vec = np.zeros(len(self.goals), dtype=np.float32)
        goal_vec[self.goals.index(sc["goal"])] = 1.0

        cond_vec = np.zeros(len(self.conditions), dtype=np.float32)
        cond_vec[self.conditions.index(sc["condition"])] = 1.0

        age_vec = np.zeros(len(self.age_groups), dtype=np.float32)
        age_vec[self.age_groups.index(sc["age_group"])] = 1.0

        targets = np.array([
            sc["target_calories"] / 1000.0,
            sc["max_sodium"] / 2000.0,
        ], dtype=np.float32)

        progress = np.array([self.current_step / self.max_steps], dtype=np.float32)

        return np.concatenate([
            quantities_norm.astype(np.float32),
            nutrition_norm,
            goal_vec,
            cond_vec,
            age_vec,
            targets,
            progress,
        ]).astype(np.float32)

    def _nutrition(self):
        totals = {
            "cal": 0.0,
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
            "fiber": 0.0,
            "sodium": 0.0,
        }
        for q, ing in zip(self.quantities, self.ingredients):
            totals["cal"] += q * ing["cal"]
            totals["protein"] += q * ing["protein"]
            totals["carbs"] += q * ing["carbs"]
            totals["fat"] += q * ing["fat"]
            totals["fiber"] += q * ing["fiber"]
            totals["sodium"] += q * ing["sodium"]
        return totals

    def _meal_score(self):
        n = self._nutrition()
        sc = self.current_scenario
        score = 0.0

        total_items = int(np.sum(self.quantities))
        diversity = int(np.sum(self.quantities > 0))

        if total_items == 0:
            return -2.0

        soy_idx = self._ingredient_index("soy_sauce")
        oil_idx = self._ingredient_index("oil")
        cabbage_idx = self._ingredient_index("cabbage")
        carrot_idx = self._ingredient_index("carrot")
        beans_idx = self._ingredient_index("beans")
        chicken_idx = self._ingredient_index("chicken")
        banana_idx = self._ingredient_index("banana")

        effective_protein = min(n["protein"], 40.0)
        effective_fiber = min(n["fiber"], 15.0)

        score += effective_protein * 0.10
        score += effective_fiber * 0.20
        score -= n["sodium"] * 0.005

        if n["fat"] > 20:
            score -= (n["fat"] - 20) * 0.12

        if diversity >= 3:
            score += 1.5
        elif diversity == 2:
            score += 0.5

        if total_items > sc["max_items"]:
            score -= (total_items - sc["max_items"]) * 1.5

        if n["sodium"] > sc["max_sodium"]:
            score -= (n["sodium"] - sc["max_sodium"]) * 0.03

        score -= self.quantities[soy_idx] * 2.0
        score += self.quantities[cabbage_idx] * 0.8
        score += self.quantities[carrot_idx] * 0.8
        score += self.quantities[beans_idx] * 0.5
        score += self.quantities[chicken_idx] * 0.5

        if self.quantities[oil_idx] > 1:
            score -= (self.quantities[oil_idx] - 1) * 1.5

        target_cal = sc["target_calories"]
        goal = sc["goal"]
        condition = sc["condition"]

        if goal == "weight_loss":
            if n["cal"] > target_cal:
                score -= (n["cal"] - target_cal) / 20.0
            else:
                score -= abs(n["cal"] - 350) / 100.0
            score += effective_protein * 0.08
            score += effective_fiber * 0.12
            score -= max(0.0, n["carbs"] - 45) * 0.08

        elif goal == "weight_gain":
            score -= abs(n["cal"] - target_cal) / 60.0
            if n["cal"] > 900:
                score -= (n["cal"] - 900) / 20.0
            score += effective_protein * 0.08

        elif goal == "maintenance":
            if n["cal"] > target_cal:
                score -= (n["cal"] - target_cal) / 20.0
            else:
                score -= abs(n["cal"] - target_cal) / 80.0

        elif goal == "baby_safe":
            score -= abs(n["cal"] - target_cal) / 60.0
            if n["cal"] > 350:
                score -= (n["cal"] - 350) / 15.0
            score -= n["sodium"] * 0.02
            score += self.quantities[banana_idx] * 1.2
            score += self.quantities[carrot_idx] * 1.2
            score += self.quantities[beans_idx] * 0.8
            score -= self.quantities[soy_idx] * 5.0
            score -= self.quantities[oil_idx] * 2.0

        if condition == "hypertension":
            score -= n["sodium"] * 0.02

        return score

    def _final_bonus(self):
        n = self._nutrition()
        sc = self.current_scenario
        bonus = 0.0

        total_items = int(np.sum(self.quantities))
        diversity = int(np.sum(self.quantities > 0))

        if total_items == 0:
            return -5.0

        if diversity >= 3:
            bonus += 1.5
        elif diversity == 2:
            bonus += 0.5

        if total_items <= sc["max_items"]:
            bonus += 2.0

        if total_items == sc["max_items"]:
            bonus += 1.0

        if n["sodium"] <= sc["max_sodium"]:
            bonus += 2.0

        target = sc["target_calories"]
        if n["cal"] <= target * 1.1 and n["cal"] >= target * 0.7:
            bonus += 4.0

        if n["protein"] >= 10 and n["fiber"] >= 4:
            bonus += 2.0

        return bonus

    def _ingredient_index(self, name):
        for i, ing in enumerate(self.ingredients):
            if ing["name"] == name:
                return i
        raise ValueError(f"Ingredient '{name}' not found.")

    def _get_info(self):
        return {
            "scenario_name": self.current_scenario["name"],
            "goal": self.current_scenario["goal"],
            "condition": self.current_scenario["condition"],
            "age_group": self.current_scenario["age_group"],
            "target_calories": self.current_scenario["target_calories"],
            "max_sodium": self.current_scenario["max_sodium"],
            "quantities": self.quantities.copy(),
            "nutrition": self._nutrition(),
            "score": self.last_score,
            "ingredient_names": [x["name"] for x in self.ingredients],
        }


    def render(self, save_path=None):
        info = self._get_info()
        return render_meal_state(info, step=self.current_step, save_path=save_path)
