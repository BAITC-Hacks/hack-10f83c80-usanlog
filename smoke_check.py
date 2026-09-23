"""Offline smoke checks for the three user-facing recommendation outcomes."""
import server

# Keep this check free of network calls even when the shell has an API key.
server.API_KEY = ""

GOOD = {
    "city": "Алматы", "event_date": "2026-09-24", "event_type": "свадьба",
    "category": "Фотограф", "budget_kzt": 500000,
}
ANNA_BUSY = {
    "city": "Алматы", "event_date": "2026-09-25", "event_type": "день рождения",
    "category": "Ведущий", "budget_kzt": 1000000, "duration_hours": 4,
}


def require(condition, message):
    if not condition:
        raise RuntimeError("Проверка не пройдена: " + message)


recommended = server.recommend(GOOD)
require(recommended["status"] == "recommendations", "должны найтись фотографы на 24 сентября")
require(1 <= len(recommended["cards"]) <= 3, "в выдаче должно быть от одной до трёх карточек")
require(recommended["count"] >= len(recommended["cards"]), "общее количество кандидатов не меньше числа карточек")
require([c["id"] for c in recommended["cards"]] == [c["id"] for c in server.recommend(GOOD)["cards"]],
        "одинаковый запрос должен возвращать одинаковый порядок")

missing = server.recommend({**GOOD, "city": "Кызылорда"})
require(missing["status"] == "no_catalog", "город без нужной категории должен давать no_catalog")

blocked = server.recommend(ANNA_BUSY)
require(blocked["status"] == "no_match", "доступные исходные карточки с занятыми датами должны давать no_match")
anna = next(p for p in server.PROVIDERS if p["id"] == "HK-29829")
require("2026-09-25" in anna["busy_list"], "25 сентября должно быть отмечено занятым для Ани Форджер")
require(all(card["id"] != "HK-29829" for card in blocked["cards"]), "Аня не должна попасть в выдачу на занятую дату")

print("OK: рекомендации, отсутствие категории, отсутствие подходящих кандидатов, стабильный порядок и календарь Ани Форджер.")
