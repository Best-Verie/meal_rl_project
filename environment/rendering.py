import os
import pygame


WIDTH = 1000
HEIGHT = 700

BG_COLOR = (245, 245, 245)
TEXT_COLOR = (30, 30, 30)
BOX_COLOR = (220, 235, 255)
BAR_BG = (220, 220, 220)
BAR_FILL = (100, 180, 120)
WARN_FILL = (220, 100, 100)
LINE_COLOR = (80, 80, 80)


def init_pygame():
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()


def get_font(size=24):
    return pygame.font.SysFont("arial", size)


def draw_text(surface, text, x, y, size=24, color=TEXT_COLOR):
    font = get_font(size)
    img = font.render(str(text), True, color)
    surface.blit(img, (x, y))


def draw_bar(surface, label, value, max_value, x, y, w=320, h=24, warn=False):
    pygame.draw.rect(surface, BAR_BG, (x, y, w, h))
    ratio = 0 if max_value <= 0 else min(max(value / max_value, 0), 1)
    fill_w = int(ratio * w)
    fill_color = WARN_FILL if warn else BAR_FILL
    pygame.draw.rect(surface, fill_color, (x, y, fill_w, h))
    pygame.draw.rect(surface, LINE_COLOR, (x, y, w, h), 2)
    draw_text(surface, f"{label}: {value:.1f} / {max_value}", x, y - 26, size=20)


def render_meal_state(info, step=None, save_path=None):
    init_pygame()

    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill(BG_COLOR)

    nutrition = info["nutrition"]
    quantities = info["quantities"]
    ingredient_names = info["ingredient_names"]

    # Header
    draw_text(surface, "Kitchen Meal Planning Environment", 30, 20, size=34)
    draw_text(surface, f"Scenario: {info['scenario_name']}", 30, 75, size=24)
    draw_text(surface, f"Goal: {info['goal']}", 30, 110, size=24)
    draw_text(surface, f"Condition: {info['condition']}", 30, 145, size=24)
    draw_text(surface, f"Age Group: {info['age_group']}", 30, 180, size=24)

    if step is not None:
        draw_text(surface, f"Step: {step}", 30, 215, size=24)

    draw_text(surface, f"Target Calories: {info['target_calories']}", 30, 250, size=22)
    draw_text(surface, f"Max Sodium: {info['max_sodium']}", 30, 280, size=22)

    # Nutrition bars
    draw_bar(
        surface,
        "Calories",
        nutrition["cal"],
        max(info["target_calories"] * 1.5, 1),
        30,
        340,
        warn=nutrition["cal"] > info["target_calories"] * 1.1,
    )

    draw_bar(
        surface,
        "Sodium",
        nutrition["sodium"],
        max(info["max_sodium"], 1),
        30,
        410,
        warn=nutrition["sodium"] > info["max_sodium"],
    )

    draw_bar(surface, "Protein", nutrition["protein"], 60, 30, 480)
    draw_bar(surface, "Fiber", nutrition["fiber"], 30, 30, 550)
    draw_bar(surface, "Fat", nutrition["fat"], 40, 30, 620)

    # Ingredient panel
    draw_text(surface, "Ingredients", 500, 70, size=30)

    start_y = 120
    for i, (name, qty) in enumerate(zip(ingredient_names, quantities)):
        box_y = start_y + i * 60
        pygame.draw.rect(surface, BOX_COLOR, (500, box_y, 430, 45))
        pygame.draw.rect(surface, LINE_COLOR, (500, box_y, 430, 45), 2)
        draw_text(surface, name, 515, box_y + 8, size=22)
        draw_text(surface, f"qty: {qty}", 840, box_y + 8, size=22)

    draw_text(surface, f"Score: {info['score']:.2f}", 500, 620, size=28)

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        pygame.image.save(surface, save_path)

    return surface
