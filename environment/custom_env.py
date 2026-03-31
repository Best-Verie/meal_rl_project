import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environment.rendering import render_meal_state


class KitchenMealPlanningEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=18):
        super().__init__()
        self.max_steps = max_steps
        self.max_quantity = 3

        # Per-portion approximations. Simplified but intentionally diverse.
        self.ingredients = [
            {"name": "rice", "group": "carb", "baby_ok": True,  "cal": 130, "protein": 2.7, "carbs": 28.0, "fat": 0.3, "fiber": 0.4, "sodium": 1},
            {"name": "beans", "group": "protein", "baby_ok": True,  "cal": 127, "protein": 8.7, "carbs": 22.8, "fat": 0.5, "fiber": 6.4, "sodium": 2},
            {"name": "chicken", "group": "protein", "baby_ok": True,  "cal": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "fiber": 0.0, "sodium": 74},
            {"name": "egg", "group": "protein", "baby_ok": True,  "cal": 78,  "protein": 6.3, "carbs": 0.6, "fat": 5.3, "fiber": 0.0, "sodium": 62},
            {"name": "yogurt", "group": "dairy", "baby_ok": True,  "cal": 59,  "protein": 10.0, "carbs": 3.6, "fat": 0.4, "fiber": 0.0, "sodium": 36},
            {"name": "milk", "group": "dairy", "baby_ok": True,  "cal": 61,  "protein": 3.2, "carbs": 4.8, "fat": 3.3, "fiber": 0.0, "sodium": 43},
            {"name": "cabbage", "group": "vegetable", "baby_ok": True,  "cal": 25,  "protein": 1.3, "carbs": 5.8, "fat": 0.1, "fiber": 2.5, "sodium": 18},
            {"name": "carrot", "group": "vegetable", "baby_ok": True,  "cal": 41,  "protein": 0.9, "carbs": 10.0, "fat": 0.2, "fiber": 2.8, "sodium": 69},
            {"name": "spinach", "group": "vegetable", "baby_ok": True,  "cal": 23,  "protein": 2.9, "carbs": 3.6, "fat": 0.4, "fiber": 2.2, "sodium": 79},
            {"name": "sweet_potato", "group": "carb", "baby_ok": True,  "cal": 86,  "protein": 1.6, "carbs": 20.1, "fat": 0.1, "fiber": 3.0, "sodium": 55},
            {"name": "banana", "group": "fruit", "baby_ok": True,  "cal": 89,  "protein": 1.1, "carbs": 23.0, "fat": 0.3, "fiber": 2.6, "sodium": 1},
            {"name": "avocado", "group": "fat", "baby_ok": True,  "cal": 160, "protein": 2.0, "carbs": 8.5, "fat": 14.7, "fiber": 6.7, "sodium": 7},
            {"name": "oil", "group": "fat", "baby_ok": False, "cal": 119, "protein": 0.0, "carbs": 0.0, "fat": 13.5, "fiber": 0.0, "sodium": 0},
            {"name": "soy_sauce", "group": "seasoning", "baby_ok": False, "cal": 9, "protein": 1.3, "carbs": 0.8, "fat": 0.0, "fiber": 0.1, "sodium": 879},
        ]

        self.num_ingredients = len(self.ingredients)
        self.name_to_idx = {ing["name"]: i for i, ing in enumerate(self.ingredients)}

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
                "target_protein": 30,
                "target_fiber": 12,
                "max_fat": 22,
                "max_items": 5,
                "min_diversity": 3,
                "required_groups": ["protein", "vegetable"],
                "forbidden": [],
                "preferred_templates": ["lean_plate", "light_bowl"],
                "banned_templates": ["sauce_heavy"],
            },
            {
                "name": "adult_weight_gain",
                "goal": "weight_gain",
                "condition": "none",
                "age_group": "adult",
                "target_calories": 700,
                "max_sodium": 700,
                "target_protein": 38,
                "target_fiber": 10,
                "max_fat": 35,
                "max_items": 6,
                "min_diversity": 3,
                "required_groups": ["protein", "carb", "vegetable"],
                "forbidden": [],
                "preferred_templates": ["gain_plate", "protein_bowl"],
                "banned_templates": ["sauce_heavy"],
            },
            {
                "name": "adult_hypertension",
                "goal": "maintenance",
                "condition": "hypertension",
                "age_group": "adult",
                "target_calories": 500,
                "max_sodium": 350,
                "target_protein": 28,
                "target_fiber": 12,
                "max_fat": 25,
                "max_items": 5,
                "min_diversity": 3,
                "required_groups": ["protein", "vegetable", "fruit_or_carb"],
                "forbidden": ["soy_sauce"],
                "preferred_templates": ["hypertension_plate", "light_bowl"],
                "banned_templates": ["sauce_heavy"],
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
                "preferred_templates": ["baby_soft_meal", "baby_breakfast"],
                "banned_templates": ["sauce_heavy"],
            },
        ]

        # Meal templates discourage single-food exploitation and encourage plate-like meals.
        self.meal_templates = {
            "lean_plate": {
                "members": ["chicken", "carrot", "spinach", "rice", "sweet_potato"],
                "required_any": [["chicken", "beans", "egg", "yogurt"], ["carrot", "spinach", "cabbage"]],
                "bonus": 2.0,
            },
            "light_bowl": {
                "members": ["beans", "rice", "cabbage", "carrot", "banana"],
                "required_any": [["beans", "chicken", "egg"], ["cabbage", "carrot", "spinach"]],
                "bonus": 2.0,
            },
            "gain_plate": {
                "members": ["rice", "chicken", "beans", "avocado", "cabbage", "sweet_potato"],
                "required_any": [["rice", "sweet_potato"], ["chicken", "beans", "egg"], ["cabbage", "carrot", "spinach"]],
                "bonus": 2.8,
            },
            "protein_bowl": {
                "members": ["beans", "egg", "yogurt", "rice", "banana"],
                "required_any": [["beans", "egg", "yogurt"], ["rice", "banana", "sweet_potato"]],
                "bonus": 2.0,
            },
            "hypertension_plate": {
                "members": ["rice", "beans", "banana", "spinach", "cabbage", "chicken"],
                "required_any": [["beans", "chicken", "egg", "yogurt"], ["spinach", "cabbage", "carrot"], ["banana", "rice", "sweet_potato"]],
                "bonus": 2.5,
            },
            "baby_soft_meal": {
                "members": ["sweet_potato", "banana", "yogurt", "egg", "carrot", "rice"],
                "required_any": [["sweet_potato", "rice", "banana"], ["yogurt", "egg", "beans"], ["carrot", "banana"]],
                "bonus": 3.4,
            },
            "baby_breakfast": {
                "members": ["banana", "yogurt", "milk", "egg", "rice"],
                "required_any": [["banana", "rice"], ["yogurt", "milk", "egg"]],
                "bonus": 3.0,
            },
            "sauce_heavy": {
                "members": ["soy_sauce", "oil"],
                "required_any": [["soy_sauce"], ["oil"]],
                "bonus": -4.0,
            },
        }

        self.action_space = spaces.Discrete(self.num_ingredients * 2 + 1)

        obs_size = (
            self.num_ingredients
            + 6
            + len(self.goals)
            + len(self.conditions)
            + len(self.age_groups)
            + 5
            + 4
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
            reward -= 0.30

        if not terminated and self._near_good_meal():
            reward -= 0.18

        if terminated:
            reward += self._final_bonus()

        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 0.80

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

        meal_meta = np.array([
            min(self._diversity() / 5.0, 1.0),
            min(np.sum(self.quantities) / max(sc["max_items"], 1), 1.0),
            min(self._template_match_score() / 4.0, 1.0),
            min(self.current_step / self.max_steps, 1.0),
        ], dtype=np.float32)

        return np.concatenate([
            quantities_norm.astype(np.float32),
            nutrition_norm,
            goal_vec,
            cond_vec,
            age_vec,
            targets,
            meal_meta,
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

    def _diversity(self):
        return int(np.sum(self.quantities > 0))

    def _present_groups(self):
        return {ing["group"] for q, ing in zip(self.quantities, self.ingredients) if q > 0}

    def _ingredient_index(self, name):
        return self.name_to_idx[name]

    def _repetition_penalty(self):
        penalty = 0.0
        for q in self.quantities:
            if q > 1:
                penalty += (q - 1) * 1.0
            if q > 2:
                penalty += (q - 2) * 2.0
        return penalty

    def _single_food_penalty(self):
        diversity = self._diversity()
        total_items = int(np.sum(self.quantities))
        if total_items >= 2 and diversity == 1:
            return 8.0
        if total_items >= 3 and diversity == 2:
            return 2.5
        return 0.0

    def _scenario_forbidden_penalty(self):
        sc = self.current_scenario
        penalty = 0.0
        for forbidden_name in sc.get("forbidden", []):
            idx = self._ingredient_index(forbidden_name)
            penalty += self.quantities[idx] * 6.0
        return penalty

    def _baby_unsuitable_penalty(self):
        if self.current_scenario["condition"] != "baby":
            return 0.0

        penalty = 0.0
        for q, ing in zip(self.quantities, self.ingredients):
            if q > 0 and not ing["baby_ok"]:
                penalty += q * 7.0

        # Baby meals should not be dominated by one or two ingredients.
        for limited_name in ["beans", "yogurt", "milk"]:
            idx = self._ingredient_index(limited_name)
            if self.quantities[idx] > 1:
                penalty += (self.quantities[idx] - 1) * 2.8

        if self.quantities[self._ingredient_index("banana")] > 1:
            penalty += (self.quantities[self._ingredient_index("banana")] - 1) * 1.8

        return penalty

    def _group_bonus(self):
        sc = self.current_scenario
        present_groups = self._present_groups()
        bonus = 0.0

        for req in sc.get("required_groups", []):
            if req == "fruit_or_carb":
                if "fruit" in present_groups or "carb" in present_groups:
                    bonus += 1.5
            elif req in present_groups:
                bonus += 1.5

        return bonus

    def _missing_required_group_penalty(self):
        sc = self.current_scenario
        present_groups = self._present_groups()
        penalty = 0.0

        for req in sc.get("required_groups", []):
            if req == "fruit_or_carb":
                if "fruit" not in present_groups and "carb" not in present_groups:
                    penalty += 3.0
            elif req not in present_groups:
                penalty += 3.0

        return penalty

    def _diversity_bonus(self):
        sc = self.current_scenario
        diversity = self._diversity()

        if diversity >= sc["min_diversity"]:
            return 3.5
        if diversity == sc["min_diversity"] - 1:
            return 1.0
        return -3.0

    def _target_alignment(self, value, target, tolerance, weight):
        diff = abs(value - target)
        return max(0.0, weight * (1.0 - diff / max(tolerance, 1e-6)))

    def _macro_balance_bonus(self, nutrition):
        sc = self.current_scenario
        bonus = 0.0
        bonus += self._target_alignment(
            min(nutrition["protein"], sc["target_protein"]),
            sc["target_protein"],
            max(sc["target_protein"], 1.0),
            3.2,
        )
        bonus += self._target_alignment(
            min(nutrition["fiber"], sc["target_fiber"]),
            sc["target_fiber"],
            max(sc["target_fiber"], 1.0),
            2.7,
        )
        return bonus

    def _calorie_score(self, nutrition):
        sc = self.current_scenario
        return self._target_alignment(
            nutrition["cal"],
            sc["target_calories"],
            sc["target_calories"] * 0.16,
            7.5,
        )

    def _safety_score(self, nutrition):
        sc = self.current_scenario
        score = 0.0

        if nutrition["sodium"] <= sc["max_sodium"]:
            score += 2.0
        else:
            score -= (nutrition["sodium"] - sc["max_sodium"]) * 0.06

        if nutrition["fat"] <= sc["max_fat"]:
            score += 1.0
        else:
            score -= (nutrition["fat"] - sc["max_fat"]) * 0.40

        return score

    def _portion_penalty(self):
        sc = self.current_scenario
        total_items = int(np.sum(self.quantities))
        if total_items <= sc["max_items"]:
            return 0.0
        return (total_items - sc["max_items"]) * 2.5

    def _template_satisfied(self, template_name):
        template = self.meal_templates[template_name]
        present = {ing["name"] for q, ing in zip(self.quantities, self.ingredients) if q > 0}

        # Must include at least 2 items from members and satisfy all required_any clauses.
        member_hits = len([name for name in template["members"] if name in present])
        if member_hits < 2:
            return False

        for clause in template["required_any"]:
            if not any(name in present for name in clause):
                return False

        return True

    def _template_match_score(self):
        sc = self.current_scenario
        score = 0.0

        for template_name in sc.get("preferred_templates", []):
            if self._template_satisfied(template_name):
                score += self.meal_templates[template_name]["bonus"]

        for template_name in sc.get("banned_templates", []):
            if self._template_satisfied(template_name):
                score += self.meal_templates[template_name]["bonus"]

        return score

    def _scenario_specific_adjustment(self, nutrition):
        sc = self.current_scenario
        adjustment = 0.0
        present_groups = self._present_groups()

        if sc["goal"] == "weight_loss":
            if nutrition["cal"] > sc["target_calories"] * 1.08:
                adjustment -= (nutrition["cal"] - sc["target_calories"] * 1.08) / 18.0
            if "vegetable" in present_groups:
                adjustment += 0.8

        elif sc["goal"] == "weight_gain":
            if nutrition["cal"] < sc["target_calories"] * 0.90:
                adjustment -= (sc["target_calories"] * 0.90 - nutrition["cal"]) / 20.0
            if nutrition["protein"] >= sc["target_protein"]:
                adjustment += 1.6
            if "carb" in present_groups:
                adjustment += 0.9

        elif sc["goal"] == "maintenance":
            adjustment -= abs(nutrition["cal"] - sc["target_calories"]) / 100.0
            if sc["condition"] == "hypertension":
                adjustment -= nutrition["sodium"] * 0.015
                if self.quantities[self._ingredient_index("soy_sauce")] > 0:
                    adjustment -= 8.0

        elif sc["goal"] == "baby_safe":
            if nutrition["cal"] > 320:
                adjustment -= (nutrition["cal"] - 320) / 12.0
            if nutrition["protein"] >= 8:
                adjustment += 1.2
            for name in ["banana", "carrot", "sweet_potato", "yogurt", "egg", "rice", "milk"]:
                idx = self._ingredient_index(name)
                if self.quantities[idx] > 0:
                    adjustment += 0.7

        return adjustment

    def _meal_score(self):
        nutrition = self._nutrition()
        total_items = int(np.sum(self.quantities))

        if total_items == 0:
            return -3.5

        score = 0.0
        score += self._calorie_score(nutrition)
        score += self._macro_balance_bonus(nutrition)
        score += self._safety_score(nutrition)
        score += self._group_bonus()
        score += self._diversity_bonus()
        score += self._template_match_score()
        score += self._scenario_specific_adjustment(nutrition)

        score -= self._missing_required_group_penalty()
        score -= self._repetition_penalty()
        score -= self._single_food_penalty()
        score -= self._portion_penalty()
        score -= self._scenario_forbidden_penalty()
        score -= self._baby_unsuitable_penalty()

        return score

    def _near_good_meal(self):
        nutrition = self._nutrition()
        sc = self.current_scenario
        diversity = self._diversity()
        present_groups = self._present_groups()

        required_ok = True
        for req in sc.get("required_groups", []):
            if req == "fruit_or_carb":
                if "fruit" not in present_groups and "carb" not in present_groups:
                    required_ok = False
            elif req not in present_groups:
                required_ok = False

        return (
            abs(nutrition["cal"] - sc["target_calories"]) <= sc["target_calories"] * 0.10
            and nutrition["sodium"] <= sc["max_sodium"]
            and nutrition["fat"] <= sc["max_fat"]
            and diversity >= sc["min_diversity"]
            and required_ok
            and self._repetition_penalty() <= 1.5
            and self._single_food_penalty() == 0
        )

    def _final_bonus(self):
        nutrition = self._nutrition()
        sc = self.current_scenario
        total_items = int(np.sum(self.quantities))
        diversity = self._diversity()

        if total_items == 0:
            return -7.0

        bonus = 0.0

        if total_items <= sc["max_items"]:
            bonus += 2.0
        if diversity >= sc["min_diversity"]:
            bonus += 3.0
        if self._missing_required_group_penalty() == 0:
            bonus += 3.0
        if nutrition["sodium"] <= sc["max_sodium"]:
            bonus += 2.0
        if nutrition["fat"] <= sc["max_fat"]:
            bonus += 1.0
        if abs(nutrition["cal"] - sc["target_calories"]) <= sc["target_calories"] * 0.10:
            bonus += 3.0
        if nutrition["protein"] >= sc["target_protein"] * 0.85:
            bonus += 2.0
        if nutrition["fiber"] >= sc["target_fiber"] * 0.85:
            bonus += 1.5
        if self._repetition_penalty() <= 1.0:
            bonus += 1.5
        if self._single_food_penalty() == 0:
            bonus += 2.0
        if self._template_match_score() > 0:
            bonus += 2.0

        return bonus

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
