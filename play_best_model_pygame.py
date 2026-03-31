import argparse
import os
import sys
from pathlib import Path

import pygame
import torch
from stable_baselines3 import DQN, PPO

sys.path.append(os.path.abspath("."))

from environment.custom_env import KitchenMealPlanningEnv
from training.reinforce_training import load_model as load_reinforce


WINDOW_TITLE = "Kitchen Meal Planning - Best Model Play"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 780
FPS = 60
STEP_DELAY_MS = 1000

BACKGROUND = (242, 245, 247)
CARD = (255, 255, 255)
CARD_BORDER = (214, 220, 226)
TEXT = (32, 37, 43)
MUTED = (96, 108, 118)
GREEN = (76, 175, 80)
RED = (220, 80, 80)
YELLOW = (232, 181, 67)
BLUE = (90, 140, 220)
PURPLE = (145, 110, 215)
ORANGE = (237, 125, 49)
TEAL = (66, 166, 166)
SHADOW = (0, 0, 0, 18)

DEFAULTS = {
    "ppo": {
        "model_path": "models/ppo/kitchen_ppo_model.zip",
        "label": "PPO best model",
        "note": "Best overall result in your notebook runs.",
    },
    "dqn": {
        "model_path": "models/dqn/kitchen_dqn_model.zip",
        "label": "DQN best model",
        "note": "Best DQN checkpoint from your notebook runs.",
    },
    "reinforce": {
        "model_path": "models/reinforce/kitchen_reinforce_model.pt",
        "label": "REINFORCE best model",
        "note": "Best REINFORCE checkpoint from your notebook runs.",
    },
}


class BestModelPlayer:
    def __init__(self, algorithm: str, scenario_name: str, model_path: str | None = None):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("arial", 30, bold=True)
        self.h2_font = pygame.font.SysFont("arial", 22, bold=True)
        self.text_font = pygame.font.SysFont("arial", 18)
        self.small_font = pygame.font.SysFont("arial", 15)
        self.big_font = pygame.font.SysFont("arial", 24, bold=True)

        self.algorithm = algorithm.lower()
        self.scenario_name = scenario_name
        self.model_path = model_path or DEFAULTS[self.algorithm]["model_path"]
        self.env = KitchenMealPlanningEnv(max_steps=12)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

        self.total_reward = 0.0
        self.paused = False
        self.done = False
        self.truncated = False
        self.last_step_time = pygame.time.get_ticks()
        self.last_action_text = "None yet"
        self.last_reward = 0.0
        self.info = {}
        self.obs = None
        self.reset_env()

    def _load_model(self):
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        if self.algorithm == "ppo":
            return PPO.load(str(path))
        if self.algorithm == "dqn":
            return DQN.load(str(path))
        if self.algorithm == "reinforce":
            return load_reinforce(str(path), self.device)
        raise ValueError("Unsupported algorithm. Choose ppo, dqn, or reinforce.")

    def reset_env(self):
        self.obs, self.info = self.env.reset(options={"scenario_name": self.scenario_name})
        self.total_reward = 0.0
        self.paused = False
        self.done = False
        self.truncated = False
        self.last_step_time = pygame.time.get_ticks()
        self.last_action_text = "Environment reset"
        self.last_reward = 0.0

        print("\nBest-model play started")
        print(f"Algorithm: {self.algorithm.upper()}")
        print(f"Model path: {self.model_path}")
        print(f"Scenario: {self.info['scenario_name']}")
        print("Controls: SPACE pause/resume | R reset | N single model step | ESC quit")

    def predict_action(self):
        if self.algorithm in {"ppo", "dqn"}:
            action, _ = self.model.predict(self.obs, deterministic=True)
            return int(action)

        obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.model(obs_tensor)
        action = torch.argmax(dist.probs, dim=1).item()
        return int(action)

    def step_model(self):
        action = self.predict_action()
        self.obs, reward, self.done, self.truncated, self.info = self.env.step(action)
        self.total_reward += reward
        self.last_reward = reward
        self.last_action_text = self.decode_action(action)
        self.print_step(reward)

    def print_step(self, reward):
        score = self.info.get("score", 0.0)
        print(
            f"Step {self.env.current_step:02d} | Action: {self.last_action_text} | "
            f"Reward: {reward:.3f} | Total reward: {self.total_reward:.3f} | Score: {score:.3f}"
        )
        if self.done:
            print("Episode ended by STOP action.")
        if self.truncated:
            print("Episode truncated by max steps.")

    def decode_action(self, action):
        n = self.env.num_ingredients
        if action < n:
            return f"ADD {self.env.ingredients[action]['name']}"
        if action < 2 * n:
            idx = action - n
            return f"REMOVE {self.env.ingredients[idx]['name']}"
        return "STOP"

    def draw_text(self, text, x, y, font=None, color=TEXT):
        font = font or self.text_font
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, (x, y))

    def draw_shadowed_card(self, rect, radius=18):
        shadow = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow, SHADOW, shadow.get_rect(), border_radius=radius)
        self.screen.blit(shadow, (rect.x + 4, rect.y + 5))
        pygame.draw.rect(self.screen, CARD, rect, border_radius=radius)
        pygame.draw.rect(self.screen, CARD_BORDER, rect, width=1, border_radius=radius)

    def progress_color(self, ratio, reverse=False):
        if reverse:
            if ratio < 0.6:
                return GREEN
            if ratio < 0.9:
                return YELLOW
            return RED
        if ratio < 0.4:
            return RED
        if ratio < 0.75:
            return YELLOW
        return GREEN

    def draw_progress_bar(self, x, y, w, h, label, value, max_value, reverse=False):
        ratio = 0.0 if max_value <= 0 else max(0.0, min(value / max_value, 1.0))
        color = self.progress_color(ratio, reverse=reverse)
        self.draw_text(f"{label}: {value:.1f} / {max_value:.1f}", x, y, self.text_font)
        bar_y = y + 28
        pygame.draw.rect(self.screen, (232, 236, 240), (x, bar_y, w, h), border_radius=10)
        pygame.draw.rect(self.screen, color, (x, bar_y, int(w * ratio), h), border_radius=10)
        pygame.draw.rect(self.screen, CARD_BORDER, (x, bar_y, w, h), width=1, border_radius=10)

    def get_metric(self, key, default=0.0):
        nutrition = self.info.get("nutrition", {})
        mapping = {
            "calories": nutrition.get("cal", default),
            "sodium": nutrition.get("sodium", default),
            "protein": nutrition.get("protein", default),
            "fiber": nutrition.get("fiber", default),
            "fat": nutrition.get("fat", default),
        }
        return float(mapping.get(key, default))

    def get_ingredient_quantities(self):
        quantities = self.info.get("quantities", [])
        names = self.info.get("ingredient_names", [])
        return list(zip(names, quantities))

    def draw_header(self):
        self.draw_text("Kitchen Meal Planning Environment", 34, 24, self.title_font)
        subtitle = f"Scenario: {self.info.get('scenario_name', self.scenario_name)}"
        self.draw_text(subtitle, 36, 62, self.text_font, MUTED)

        status = "RUNNING"
        status_color = GREEN
        if self.paused:
            status = "PAUSED"
            status_color = YELLOW
        elif self.done:
            status = "FINISHED (STOP)"
            status_color = BLUE
        elif self.truncated:
            status = "FINISHED (MAX STEPS)"
            status_color = PURPLE

        pill = pygame.Rect(1030, 24, 210, 42)
        pygame.draw.rect(self.screen, status_color, pill, border_radius=20)
        label = self.text_font.render(status, True, (255, 255, 255))
        self.screen.blit(label, (pill.centerx - label.get_width() // 2, pill.y + 10))

    def draw_summary_cards(self):
        algo_label = DEFAULTS[self.algorithm]["label"]
        cards = [
            ("Algorithm", algo_label, BLUE),
            ("Last Action", self.last_action_text, TEAL),
            ("Last Reward", f"{self.last_reward:.2f}", ORANGE),
            ("Total Reward", f"{self.total_reward:.2f}", PURPLE),
        ]
        x = 34
        y = 100
        widths = [240, 360, 180, 200]
        for i, (title, value, accent) in enumerate(cards):
            rect = pygame.Rect(x, y, widths[i], 92)
            self.draw_shadowed_card(rect)
            pygame.draw.rect(self.screen, accent, (rect.x, rect.y, 8, rect.height), border_radius=18)
            self.draw_text(title, rect.x + 22, rect.y + 16, self.small_font, MUTED)
            value_font = self.big_font if i != 1 else self.text_font
            self.draw_text(value, rect.x + 22, rect.y + 42, value_font)
            x += widths[i] + 18

    def draw_left_panel(self):
        rect = pygame.Rect(34, 220, 560, 520)
        self.draw_shadowed_card(rect)
        self.draw_text("Nutrition Dashboard", rect.x + 22, rect.y + 18, self.h2_font)
        self.draw_text(DEFAULTS[self.algorithm]["note"], rect.x + 22, rect.y + 48, self.small_font, MUTED)

        calories = self.get_metric("calories")
        sodium = self.get_metric("sodium")
        protein = self.get_metric("protein")
        fiber = self.get_metric("fiber")
        fat = self.get_metric("fat")

        target_calories = float(self.info.get("target_calories", 500.0))
        max_sodium = float(self.info.get("max_sodium", 400.0))
        target_protein = 60.0
        target_fiber = 30.0
        max_fat = 40.0

        x = rect.x + 22
        y = rect.y + 90
        w = 510
        self.draw_progress_bar(x, y, w, 22, "Calories", calories, target_calories)
        self.draw_progress_bar(x, y + 88, w, 22, "Sodium", sodium, max_sodium, reverse=True)
        self.draw_progress_bar(x, y + 176, w, 22, "Protein", protein, target_protein)
        self.draw_progress_bar(x, y + 264, w, 22, "Fiber", fiber, target_fiber)
        self.draw_progress_bar(x, y + 352, w, 22, "Fat", fat, max_fat, reverse=True)

        score = float(self.info.get("score", 0.0))
        score_color = GREEN if score >= 0 else RED
        self.draw_text(f"Score: {score:.2f}", x, rect.bottom - 54, self.big_font, score_color)

    def draw_right_panel(self):
        rect = pygame.Rect(622, 220, 624, 520)
        self.draw_shadowed_card(rect)
        self.draw_text("Ingredient Quantities", rect.x + 22, rect.y + 18, self.h2_font)
        self.draw_text(f"Model path: {self.model_path}", rect.x + 22, rect.y + 48, self.small_font, MUTED)

        items = self.get_ingredient_quantities()
        x = rect.x + 22
        y = rect.y + 86
        col_w = 280
        row_h = 74

        for i, (name, qty) in enumerate(items):
            col = i % 2
            row = i // 2
            card_x = x + col * (col_w + 18)
            card_y = y + row * row_h
            item_rect = pygame.Rect(card_x, card_y, col_w, 56)
            pygame.draw.rect(self.screen, (245, 248, 251), item_rect, border_radius=16)
            pygame.draw.rect(self.screen, CARD_BORDER, item_rect, width=1, border_radius=16)

            badge_color = GREEN if qty > 0 else (210, 215, 220)
            badge_rect = pygame.Rect(card_x + 200, card_y + 12, 60, 30)
            pygame.draw.rect(self.screen, badge_color, badge_rect, border_radius=15)

            self.draw_text(name, card_x + 16, card_y + 17, self.text_font)
            qty_text = self.text_font.render(f"qty {int(qty)}", True, (255, 255, 255) if qty > 0 else TEXT)
            self.screen.blit(qty_text, (badge_rect.centerx - qty_text.get_width() // 2, badge_rect.y + 6))

    def draw_footer(self):
        self.draw_text("SPACE pause/resume   |   R reset   |   N single model step   |   ESC quit", 34, 752, self.small_font, MUTED)

    def draw(self):
        self.screen.fill(BACKGROUND)
        self.draw_header()
        self.draw_summary_cards()
        self.draw_left_panel()
        self.draw_right_panel()
        self.draw_footer()
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_env()
                elif event.key == pygame.K_n and not (self.done or self.truncated):
                    self.step_model()
        return True

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            now = pygame.time.get_ticks()
            if not self.paused and not (self.done or self.truncated) and now - self.last_step_time >= STEP_DELAY_MS:
                self.step_model()
                self.last_step_time = now
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["ppo", "dqn", "reinforce"], default="ppo")
    parser.add_argument("--scenario", default="adult_hypertension")
    parser.add_argument("--model-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    app = BestModelPlayer(args.algorithm, args.scenario, args.model_path)
    app.run()


if __name__ == "__main__":
    main()
