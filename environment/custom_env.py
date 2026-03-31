import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environment.rendering import render_meal_state


class KitchenMealPlanningEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=20):
        super().__init__()
        self.max_steps = max_steps

        # Per-portion approximations. This is still a simplified prototype,
        # but the food set is more realistic and diverse than the original one.
        self.ingredients = [
            {"name": "rice", "group": "carb", "texture": "soft", "baby_ok": True,  "cal": 130, "protein": 2.7, "carbs": 28.0, "fat": 0.3, "fiber": 0.4, "sodium": 1},
            {"name": "beans", "group": "protein", "texture": "soft", "baby_ok": True,  "cal": 127, "protein": 8.7, "carbs": 22.8, "fat": 0.5, "fiber": 6.4, "sodium": 2},
            {"name": "chicken", "group": "protein", "texture": "soft", "baby_ok": True,  "cal": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "fiber": 0.0, "sodium": 74},
            {"name": "egg", "group": "protein", "texture": "soft", "baby_ok": True,  "cal": 78,  "protein": 6.3, "carbs": 0.6, "fat": 5.3, "fiber": 0.0, "sodium": 62},
            {"name": "yogurt", "group": "dairy", "texture": "soft", "baby_ok": True,  "cal": 59,  "protein": 10.0, "carbs": 3.6, "fat": 0.4, "fiber": 0.0, "sodium": 36},
            {"name": "oil", "group": "fat", "texture": "soft", "baby_ok": False, "cal": 119, "protein": 0.0, "carbs": 0.0, "fat": 13.5, "fiber": 0.0, "sodium": 0},
            {"name": "cabbage", "group": "vegetable", "texture": "soft", "baby_ok": True,  "cal": 25,  "protein": 1.3, "carbs": 5.8, "fat": 0.1, "fiber": 2.5, "sodium": 18},
            {"name": "carrot", "group": "vegetable", "texture": "soft", "baby_ok": True,  "cal": 41,  "protein": 0.9, "carbs": 10.0, "fat": 0.2, "fiber": 2.8, "sodium": 69},
            {"name": "spinach", "group": "vegetable", "texture": "soft", "baby_ok": True,  "cal": 23,  "protein": 2.9, "carbs": 3.6, "fat": 0.4, "fiber": 2.2, "sodium": 79},
            {"name": "sweet_potato", "group": "carb", "texture": "soft", "baby_ok": True,  "cal": 86,  "protein": 1.6, "carbs": 20.1, "fat": 0.1, "fiber": 3.0, "sodium": 55},
            {"name": "banana", "group": "fruit", "texture": "soft", "baby_ok": True,  "cal": 89,  "protein": 1.1, "carbs": 23.0, "fat": 0.3, "fiber": 2.6, "sodium": 1},
            {"name": "avocado", "group": "fat", "texture": "soft", "baby_ok": True,  "cal": 160, "protein": 2.0, "carbs": 8.5, "fat": 14.7, "fiber": 6.7, "sodium": 7},
            {"name": "soy_sauce", "group": "seasoning", "texture": "liquid", "baby_ok": False, "cal": 9,   "protein": 1.3, "carbs": 0.8, "fat": 0.0, "fiber": 0.1, "sodium": 879},
        ]

        self.num_ingredients = len(self.ingredients)
        self.max_quantity = 4

        self.goals = ["weight_loss", "weight_gain", "maintenance", "baby_safe"]
        self.conditions = ["none", "hypertension", "baby"]
        self.age_groups = ["adult", "infant"]

        self.scenarios = [
            {
                "name": "adult_weight_loss",
                "goal": "weight_loss",
                "condition": "none",
                "age_group": "adult",
                "target_calories": 450,
                "max_sodium": 500,
                "target_protein": 28,
                "target_fiber": 12,
                "max_fat": 22,
                "max_items": 6,
                "min_diversity": 3,
                "required_groups": ["protein", "vegetable"],
                "forbidden": [],
            },
            {
                "name": "adult_weight_gain",
                "goal": "weight_gain",
                "condition": "none",
                "age_group": "adult",
                "target_calories": 700,
                "max_sodium": 700,
                "target_protein": 35,
                "target_fiber": 10,
                "max_fat": 35,
                "max_items": 7,
                "min_diversity": 3,
                "required_groups": ["protein", "carb", "vegetable"],
                "forbidden": [],
            },
            {
                "name": "adult_hypertension",
                "goal": "maintenance",
                "condition": "hypertension",
                "age_group": "adult",
                "target_calories": 500,
                "max_sodium": 350,
                "target_protein": 25,
                "target_fiber": 12,
                "max_fat": 25,
                "max_items": 6,
                "min_diversity": 3,
                "required_groups": ["protein", "vegetable", "fruit"],
                "forbidden": ["soy_sauce"],
            },
            {
                "name": "baby_meal",
                "goal": "baby_safe",
                "condition": "baby",
                "age_group": "infant",
                "target_calories": 280,
                "max_sodium": 120,
                "target_protein": 12,
                "target_fiber": 5,
                "max_fat": 12,
                "max_items": 4,
                "min_diversity": 3,
                "required_groups": ["protein", "vegetable", "fruit_or_carb"],
                "forbidden": ["soy_sauce", "oil"],
            },
        ]

        self.action_space = spaces.Discrete(self.num_ingredients * 2 + 1)

        obs_size = (
            self.num_ingredients
            + 6
            + len(self.goals)
            + len(self.conditions)
            + len(self.age_groups)
            + 5
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

        reward -= 0.03

        if np.array_equal(old_quantities, self.quantities) and not terminated:
            reward -= 0.20

        if not terminated and self._near_good_meal():
            reward -= 0.15

        if terminated:
            reward += self._final_bonus()

        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 0.75

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _get_obs(self):
        n = self._nutrition()
        sc = self.current_scenario

        quantities_norm = self.quantities / self.max_quantity
        nutrition_norm = np.array([
            min(n["cal"] / 1000.0, 1.0),
            min(n["protein"] / 80.0, 1.0),
            min(n["carbs"] / 180.0, 1.0),
            min(n["fat"] / 60.0, 1.0),
            min(n["fiber"] / 40.0, 1.0),
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
            sc["target_protein"] / 80.0,
            sc["target_fiber"] / 40.0,
            sc["max_fat"] / 60.0,
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
        totals = {"cal": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0, "sodium": 0.0}
        for q, ing in zip(self.quantities, self.ingredients):
            totals["cal"] += q * ing["cal"]
            totals["protein"] += q * ing["protein"]
            totals["carbs"] += q * ing["carbs"]
            totals["fat"] += q * ing["fat"]
            totals["fiber"] += q * ing["fiber"]
            totals["sodium"] += q * ing["sodium"]
        return totals

    def _group_counts(self):
        counts = {}
        for q, ing in zip(self.quantities, self.ingredients):
            if q > 0:
                counts[ing["group"]] = counts.get(ing["group"], 0) + 1
        return counts

    def _diversity(self):
        return int(np.sum(self.quantities > 0))

    def _repetition_penalty(self):
        penalty = 0.0
        for q in self.quantities:
            if q > 2:
                penalty += (q - 2) * 1.25
        return penalty

    def _scenario_forbidden_penalty(self):
        sc = self.current_scenario
        penalty = 0.0
        for forbidden_name in sc.get("forbidden", []):
            idx = self._ingredient_index(forbidden_name)
            penalty += self.quantities[idx] * 4.0
        return penalty

    def _baby_unsuitable_penalty(self):
        if self.current_scenario["condition"] != "baby":
            return 0.0
        penalty = 0.0
        for q, ing in zip(self.quantities, self.ingredients):
            if q > 0 and not ing["baby_ok"]:
                penalty += q * 5.0
        return penalty

    def _group_bonus(self):
        sc = self.current_scenario
        present_groups = {ing["group"] for q, ing in zip(self.quantities, self.ingredients) if q > 0}
        bonus = 0.0
        for req in sc.get("required_groups", []):
            if req == "fruit_or_carb":
                if "fruit" in present_groups or "carb" in present_groups:
                    bonus += 1.0
            elif req in present_groups:
                bonus += 1.0
        return bonus

    def _target_alignment(self, value, target, tolerance, weight):
        diff = abs(value - target)
        return max(0.0, weight * (1.0 - diff / max(tolerance, 1e-6)))

    def _meal_score(self):
        n = self._nutrition()
        sc = self.current_scenario
        total_items = int(np.sum(self.quantities))
        diversity = self._diversity()

        if total_items == 0:
            return -2.5

        score = 0.0

        # Base target alignment
        score += self._target_alignment(n["cal"], sc["target_calories"], sc["target_calories"] * 0.45, 6.0)
        score += self._target_alignment(min(n["protein"], sc["target_protein"]), sc["target_protein"], sc["target_protein"], 3.0)
        score += self._target_alignment(min(n["fiber"], sc["target_fiber"]), sc["target_fiber"], max(sc["target_fiber"], 1.0), 2.5)

        # Hard penalties for exceeding safety limits
        if n["sodium"] > sc["max_sodium"]:
            score -= (n["sodium"] - sc["max_sodium"]) * 0.04
        else:
            score += 1.5

        if n["fat"] > sc["max_fat"]:
            score -= (n["fat"] - sc["max_fat"]) * 0.30

        if total_items > sc["max_items"]:
            score -= (total_items - sc["max_items"]) * 1.8

        # Encourage variety, but not random clutter
        if diversity >= sc["min_diversity"]:
            score += 2.5
        elif diversity == sc["min_diversity"] - 1:
            score += 1.0
        else:
            score -= 1.5

        score += self._group_bonus()
        score -= self._repetition_penalty()
        score -= self._scenario_forbidden_penalty()
        score -= self._baby_unsuitable_penalty()

        # Scenario-specific shaping
        if sc["goal"] == "weight_loss":
            if n["cal"] > sc["target_calories"] * 1.1:
                score -= (n["cal"] - sc["target_calories"] * 1.1) / 18.0

        elif sc["goal"] == "weight_gain":
            if n["cal"] < sc["target_calories"] * 0.85:
                score -= (sc["target_calories"] * 0.85 - n["cal"]) / 30.0
            if n["protein"] >= sc["target_protein"]:
                score += 1.5

        elif sc["goal"] == "maintenance":
            score -= abs(n["cal"] - sc["target_calories"]) / 80.0

        elif sc["goal"] == "baby_safe":
            # Gentle, diverse, baby-appropriate meals.
            if n["cal"] > 350:
                score -= (n["cal"] - 350) / 12.0
            if n["sodium"] > 120:
                score -= (n["sodium"] - 120) * 0.06
            # Extra reward for baby-friendly variety
            for name in ["banana", "carrot", "sweet_potato", "yogurt", "egg"]:
                idx = self._ingredient_index(name)
                if self.quantities[idx] > 0:
                    score += 0.8

        if sc["condition"] == "hypertension":
            score -= n["sodium"] * 0.02
            if self.quantities[self._ingredient_index("soy_sauce")] > 0:
                score -= 6.0

        return score

    def _near_good_meal(self):
        n = self._nutrition()
        sc = self.current_scenario
        total_items = int(np.sum(self.quantities))
        diversity = self._diversity()
        return (
            abs(n["cal"] - sc["target_calories"]) <= sc["target_calories"] * 0.12
            and n["sodium"] <= sc["max_sodium"]
            and total_items <= sc["max_items"]
            and diversity >= sc["min_diversity"]
        )

    def _final_bonus(self):
        n = self._nutrition()
        sc = self.current_scenario
        total_items = int(np.sum(self.quantities))
        diversity = self._diversity()

        if total_items == 0:
            return -5.0

        bonus = 0.0

        if total_items <= sc["max_items"]:
            bonus += 2.0
        if diversity >= sc["min_diversity"]:
            bonus += 2.5
        if n["sodium"] <= sc["max_sodium"]:
            bonus += 2.0
        if abs(n["cal"] - sc["target_calories"]) <= sc["target_calories"] * 0.12:
            bonus += 3.0
        if n["protein"] >= sc["target_protein"] * 0.8:
            bonus += 1.5
        if n["fiber"] >= sc["target_fiber"] * 0.8:
            bonus += 1.0
        if self._repetition_penalty() == 0:
            bonus += 1.0

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
            "target_protein": self.current_scenario["target_protein"],
            "target_fiber": self.current_scenario["target_fiber"],
            "max_fat": self.current_scenario["max_fat"],
            "min_diversity": self.current_scenario["min_diversity"],
            "quantities": self.quantities.copy(),
            "nutrition": self._nutrition(),
            "score": self.last_score,
            "ingredient_names": [x["name"] for x in self.ingredients],
        }

    def render(self, save_path=None):
        info = self._get_info()
        return render_meal_state(info, step=self.current_step, save_path=save_path)
