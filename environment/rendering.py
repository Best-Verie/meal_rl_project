import pygame


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 780
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


def _ensure_pygame():
    if not pygame.get_init():
        pygame.init()
        pygame.font.init()


def _draw_text(surface, text, x, y, font, color=TEXT):
    img = font.render(str(text), True, color)
    surface.blit(img, (x, y))


def _draw_shadowed_card(surface, rect, radius=18):
    shadow = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(shadow, SHADOW, shadow.get_rect(), border_radius=radius)
    surface.blit(shadow, (rect.x + 4, rect.y + 5))
    pygame.draw.rect(surface, CARD, rect, border_radius=radius)
    pygame.draw.rect(surface, CARD_BORDER, rect, width=1, border_radius=radius)


def _progress_color(ratio, reverse=False):
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


def _draw_progress_bar(surface, x, y, w, h, label, value, max_value, font, reverse=False):
    ratio = 0.0 if max_value <= 0 else max(0.0, min(value / max_value, 1.0))
    color = _progress_color(ratio, reverse=reverse)
    _draw_text(surface, f"{label}: {value:.1f} / {max_value:.1f}", x, y, font)
    bar_y = y + 28
    pygame.draw.rect(surface, (232, 236, 240), (x, bar_y, w, h), border_radius=10)
    pygame.draw.rect(surface, color, (x, bar_y, int(w * ratio), h), border_radius=10)
    pygame.draw.rect(surface, CARD_BORDER, (x, bar_y, w, h), width=1, border_radius=10)


def _status_from_info(info, step):
    score = float(info.get("score", 0.0))
    quantities = info.get("quantities", [])
    if step == 0:
        return "READY", BLUE
    if sum(int(q) for q in quantities) == 0 and score <= 0:
        return "EMPTY", YELLOW
    return "IN PROGRESS", TEAL


def render_meal_state(info, step=0, save_path=None):
    _ensure_pygame()

    surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    surface.fill(BACKGROUND)

    title_font = pygame.font.SysFont("arial", 30, bold=True)
    h2_font = pygame.font.SysFont("arial", 22, bold=True)
    text_font = pygame.font.SysFont("arial", 18)
    small_font = pygame.font.SysFont("arial", 15)
    big_font = pygame.font.SysFont("arial", 24, bold=True)

    scenario_name = info.get("scenario_name", "unknown_scenario")
    goal = info.get("goal", "unknown")
    condition = info.get("condition", "unknown")
    age_group = info.get("age_group", "unknown")
    score = float(info.get("score", 0.0))

    nutrition = info.get("nutrition", {})
    calories = float(nutrition.get("cal", 0.0))
    sodium = float(nutrition.get("sodium", 0.0))
    protein = float(nutrition.get("protein", 0.0))
    fiber = float(nutrition.get("fiber", 0.0))
    fat = float(nutrition.get("fat", 0.0))

    target_calories = float(info.get("target_calories", 500.0))
    max_sodium = float(info.get("max_sodium", 400.0))
    target_protein = float(info.get("target_protein", 25.0))
    target_fiber = float(info.get("target_fiber", 10.0))
    max_fat = float(info.get("max_fat", 25.0))
    min_diversity = int(info.get("min_diversity", 3))

    ingredient_names = info.get("ingredient_names", [])
    quantities = info.get("quantities", [])
    items = list(zip(ingredient_names, quantities))

    diversity = sum(1 for q in quantities if int(q) > 0)
    total_items = sum(int(q) for q in quantities)
    status, status_color = _status_from_info(info, step)

    _draw_text(surface, "Kitchen Meal Planning Environment", 34, 24, title_font)
    _draw_text(surface, f"Scenario: {scenario_name}", 36, 62, text_font, MUTED)

    pill = pygame.Rect(1030, 24, 210, 42)
    pygame.draw.rect(surface, status_color, pill, border_radius=20)
    label = text_font.render(status, True, (255, 255, 255))
    surface.blit(label, (pill.centerx - label.get_width() // 2, pill.y + 10))

    cards = [
        ("Goal", goal.replace("_", " "), BLUE),
        ("Condition", condition.replace("_", " "), TEAL),
        ("Age Group", age_group.replace("_", " "), ORANGE),
        ("Step", str(step), PURPLE),
    ]
    x = 34
    y = 100
    widths = [240, 240, 220, 140]
    for i, (title, value, accent) in enumerate(cards):
        rect = pygame.Rect(x, y, widths[i], 92)
        _draw_shadowed_card(surface, rect)
        pygame.draw.rect(surface, accent, (rect.x, rect.y, 8, rect.height), border_radius=18)
        _draw_text(surface, title, rect.x + 22, rect.y + 16, small_font, MUTED)
        _draw_text(surface, value, rect.x + 22, rect.y + 42, big_font if i != 3 else text_font)
        x += widths[i] + 18

    rect = pygame.Rect(34, 220, 560, 520)
    _draw_shadowed_card(surface, rect)
    _draw_text(surface, "Nutrition Dashboard", rect.x + 22, rect.y + 18, h2_font)
    _draw_text(surface, f"Score: {score:.2f}   |   Diversity: {diversity}/{min_diversity}   |   Portions: {total_items}", rect.x + 22, rect.y + 48, small_font, MUTED)

    x = rect.x + 22
    y = rect.y + 90
    w = 510
    _draw_progress_bar(surface, x, y, w, 22, "Calories", calories, target_calories, text_font)
    _draw_progress_bar(surface, x, y + 88, w, 22, "Sodium", sodium, max_sodium, text_font, reverse=True)
    _draw_progress_bar(surface, x, y + 176, w, 22, "Protein", protein, target_protein, text_font)
    _draw_progress_bar(surface, x, y + 264, w, 22, "Fiber", fiber, target_fiber, text_font)
    _draw_progress_bar(surface, x, y + 352, w, 22, "Fat", fat, max_fat, text_font, reverse=True)

    score_color = GREEN if score >= 0 else RED
    _draw_text(surface, f"Meal score: {score:.2f}", x, rect.bottom - 54, big_font, score_color)

    rect = pygame.Rect(622, 220, 624, 520)
    _draw_shadowed_card(surface, rect)
    _draw_text(surface, "Ingredient Quantities", rect.x + 22, rect.y + 18, h2_font)
    _draw_text(surface, "Expanded ingredient set for more realistic meals", rect.x + 22, rect.y + 48, small_font, MUTED)

    x = rect.x + 22
    y = rect.y + 86
    col_w = 280
    row_h = 70

    for i, (name, qty) in enumerate(items):
        col = i % 2
        row = i // 2
        card_x = x + col * (col_w + 18)
        card_y = y + row * row_h
        item_rect = pygame.Rect(card_x, card_y, col_w, 54)
        pygame.draw.rect(surface, (245, 248, 251), item_rect, border_radius=16)
        pygame.draw.rect(surface, CARD_BORDER, item_rect, width=1, border_radius=16)

        badge_color = GREEN if int(qty) > 0 else (210, 215, 220)
        badge_rect = pygame.Rect(card_x + 194, card_y + 12, 70, 30)
        pygame.draw.rect(surface, badge_color, badge_rect, border_radius=15)

        _draw_text(surface, name, card_x + 16, card_y + 15, text_font)
        qty_text = text_font.render(f"qty {int(qty)}", True, (255, 255, 255) if int(qty) > 0 else TEXT)
        surface.blit(qty_text, (badge_rect.centerx - qty_text.get_width() // 2, badge_rect.y + 6))

    _draw_text(surface, "Used by env.render(), random play, and screenshot capture.", 34, 752, small_font, MUTED)

    if save_path:
        pygame.image.save(surface, save_path)

    return surface
