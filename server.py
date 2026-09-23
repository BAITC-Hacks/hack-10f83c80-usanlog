"""Deterministic contractor recommendations with optional OpenAI-compatible APIs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import threading
import urllib.error
import urllib.request
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DATA_FILE = Path(os.getenv("PROVIDERS_CSV", ROOT / "data" / "providers.csv"))
CACHE_FILE = ROOT / ".cache" / "embeddings.json"
API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "8000"))
API_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "18"))


def truth(value: str) -> bool:
    return value.strip().casefold() in {"true", "1", "yes", "да"}


def split_values(value: str) -> List[str]:
    return [part.strip() for part in (value or "").split("|") if part.strip()]


def norm(value: str) -> str:
    value = (value or "").strip().casefold().replace("ё", "е")
    return re.sub(r"\s+", " ", value)


def load_providers() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        raise RuntimeError("Не найден каталог подрядчиков: " + str(DATA_FILE))
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    providers = []
    for row in rows:
        row["categories_list"] = split_values(row.get("categories", ""))
        row["formats_list"] = split_values(row.get("event_formats", ""))
        row["languages_list"] = split_values(row.get("languages", ""))
        row["busy_list"] = split_values(row.get("busy_dates", ""))
        try:
            row["price"] = int(float(row["price_from_kzt"])) if row.get("price_from_kzt", "").strip() else None
        except ValueError:
            row["price"] = None
        try:
            row["max_hours_value"] = float(row["max_hours"]) if row.get("max_hours", "").strip() else None
        except ValueError:
            row["max_hours_value"] = None
        row["city_is_imputed"] = truth(row.get("city_imputed", ""))
        row["price_is_imputed"] = truth(row.get("price_imputed", ""))
        row["is_synthetic"] = truth(row.get("synthetic", ""))
        providers.append(row)
    return providers


PROVIDERS = load_providers()


def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        API_BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


_cache_lock = threading.Lock()


def load_embedding_cache() -> Dict[str, List[float]]:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def text_key(text: str) -> str:
    return hashlib.sha256((EMBED_MODEL + "\0" + text).encode("utf-8")).hexdigest()


def embed(texts: List[str]) -> List[List[float]]:
    if not API_KEY:
        return []
    response = api_post("/embeddings", {"model": EMBED_MODEL, "input": texts})
    return [item["embedding"] for item in sorted(response["data"], key=lambda item: item["index"])]


def get_provider_embeddings(providers: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    with _cache_lock:
        cache = load_embedding_cache()
        missing = []
        for provider in providers:
            description = provider.get("description", "").strip()
            key = text_key(description)
            if description and key not in cache:
                missing.append((key, description))
        if missing:
            # Modest batches keep the first request reliable on hosted APIs.
            for start in range(0, len(missing), 32):
                batch = missing[start:start + 32]
                vectors = embed([text for _, text in batch])
                for (key, _), vector in zip(batch, vectors):
                    cache[key] = vector
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
        return {provider["id"]: cache[text_key(provider.get("description", "").strip())]
                for provider in providers if text_key(provider.get("description", "").strip()) in cache}


def cosine(left: List[float], right: List[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


ALIASES = {
    "астана": {"астана", "нур-султан", "нурсултан"},
    "алматы": {"алматы", "алмата"},
    "день рождения": {"день рождения", "др", "birthday"},
    "корпоратив": {"корпоратив", "корпоративное мероприятие", "corporate"},
}


def equivalent(value: str, candidate: str) -> bool:
    a, b = norm(value), norm(candidate)
    return b == a or b in ALIASES.get(a, {a}) or a in ALIASES.get(b, {b})


def request_text(req: Dict[str, Any]) -> str:
    parts = [req["event_type"], req["category"], req["city"]]
    if req.get("language"):
        parts.append(req["language"])
    if req.get("duration_hours") is not None:
        parts.append("мероприятие длительностью " + str(req["duration_hours"]) + " часов")
    return ". ".join(parts)


def validate_request(body: Dict[str, Any]) -> Dict[str, Any]:
    required = ["city", "event_date", "event_type", "category", "budget_kzt"]
    missing = [key for key in required if body.get(key) in (None, "")]
    if missing:
        raise ValueError("Заполните обязательные поля: " + ", ".join(missing))
    try:
        parsed_date = date.fromisoformat(str(body["event_date"]))
        budget = int(body["budget_kzt"])
        duration = float(body["duration_hours"]) if body.get("duration_hours") not in (None, "") else None
    except (ValueError, TypeError):
        raise ValueError("Проверьте дату, бюджет и длительность мероприятия.")
    if budget < 0 or (duration is not None and duration <= 0):
        raise ValueError("Бюджет должен быть неотрицательным, а длительность — больше нуля.")
    return {
        "city": str(body["city"]).strip(), "event_date": parsed_date.isoformat(),
        "event_type": str(body["event_type"]).strip(), "category": str(body["category"]).strip(),
        "budget_kzt": budget, "duration_hours": duration,
        "language": str(body.get("language") or "").strip(),
    }


def candidate_failures(provider: Dict[str, Any], req: Dict[str, Any]) -> List[str]:
    reasons = []
    if provider["is_synthetic"]:
        return ["synthetic"]
    if req["event_date"] in provider["busy_list"]:
        reasons.append("busy")
    if provider["price"] is None:
        reasons.append("price_unknown")
    elif provider["price"] > req["budget_kzt"]:
        reasons.append("over_budget")
    if not provider["formats_list"] or not any(equivalent(req["event_type"], item) for item in provider["formats_list"]):
        reasons.append("format")
    if req.get("language"):
        if not provider["languages_list"]:
            reasons.append("language_unknown")
        elif not any(equivalent(req["language"], item) for item in provider["languages_list"]):
            reasons.append("language")
    if req.get("duration_hours") is not None:
        if provider["max_hours_value"] is None:
            reasons.append("duration_unknown")
        elif provider["max_hours_value"] < req["duration_hours"]:
            reasons.append("duration")
    return reasons


FAIL_LABELS = {
    "busy": "заняты на выбранную дату", "over_budget": "цена выше бюджета",
    "price_unknown": "в каталоге не указана цена", "format": "не подтвердили работу с таким форматом",
    "language": "не указан нужный язык", "language_unknown": "язык не указан в каталоге",
    "duration": "не работают нужное количество часов", "duration_unknown": "длительность работы не указана",
    "synthetic": "синтетические записи каталога исключены",
}


def evidence_for(provider: Dict[str, Any], req: Dict[str, Any]) -> List[Dict[str, str]]:
    evidence = [
        {"id": "city", "fact": ("город " + provider["city"] + (" (указан ориентировочно)" if provider["city_is_imputed"] else ""))},
        {"id": "category", "fact": "категория " + ", ".join(provider["categories_list"])},
        {"id": "format", "fact": "форматы " + ", ".join(provider["formats_list"])},
        {"id": "date", "fact": "дата " + req["event_date"] + " отсутствует в списке занятых дат"},
    ]
    if provider["price"] is not None:
        evidence.append({"id": "price", "fact": "цена от {:,} ₸{}".format(provider["price"], " (ориентировочная)" if provider["price_is_imputed"] else "").replace(",", " ")})
    if provider["languages_list"]:
        evidence.append({"id": "language", "fact": "языки " + ", ".join(provider["languages_list"])})
    if provider["max_hours_value"] is not None:
        evidence.append({"id": "duration", "fact": "до {} ч".format(provider["max_hours_value"])})
    if provider.get("description", "").strip():
        evidence.append({"id": "description", "fact": provider["description"].strip()[:1200]})
    return evidence


def fallback_explanation(provider: Dict[str, Any], req: Dict[str, Any]) -> str:
    price = "Цена от {:,} ₸{}".format(provider["price"], " (ориентировочная)" if provider["price_is_imputed"] else "").replace(",", " ")
    first = "{} подходит по формату «{}» и укладывается в бюджет {} ₸.".format(provider["anon_name"], req["event_type"], req["budget_kzt"])
    second = [price]
    if req.get("language"):
        second.append("язык: " + ", ".join(provider["languages_list"]))
    if req.get("duration_hours") is not None:
        second.append("до {} ч при запросе на {} ч".format(provider["max_hours_value"], req["duration_hours"]))
    if provider["city_is_imputed"]:
        second.append("город указан ориентировочно")
    return first + " " + "; ".join(second) + "."


def llm_explanations(cards: List[Dict[str, Any]], req: Dict[str, Any]) -> None:
    if not API_KEY or not cards:
        return
    payload_cards = [{"id": card["id"], "name": card["name"], "evidence": card["evidence"]} for card in cards]
    prompt = (
        "Составь по-русски объяснения для карточек. Для каждой карточки 1–2 предложения, "
        "конкретно сравнивая запрос и факты каталога. Не используй общие похвалы. "
        "Не добавляй фактов, которых нет во входных данных. Не считай непроверенную цену точной. "
        "Верни JSON-объект вида {\"cards\":[{\"id\":\"...\",\"explanation\":\"...\"}]}, по одному объекту на каждую карточку.\n"
        "Запрос: " + json.dumps(req, ensure_ascii=False) + "\nКарточки и факты: " + json.dumps(payload_cards, ensure_ascii=False)
    )
    try:
        result = api_post("/chat/completions", {
            "model": CHAT_MODEL, "temperature": 0,
            "messages": [{"role": "system", "content": "Ты пишешь только проверяемые объяснения по предоставленным фактам."}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        })
        content = result["choices"][0]["message"]["content"]
        decoded = json.loads(content)
        generated = decoded if isinstance(decoded, list) else decoded.get("cards", decoded.get("results", []))
        by_id = {str(item.get("id")): item.get("explanation", "") for item in generated if isinstance(item, dict)}
        for card in cards:
            explanation = str(by_id.get(card["id"], "")).strip()
            if explanation and len(explanation) <= 550 and "отличный выбор" not in explanation.casefold():
                card["explanation"] = explanation
        for card in cards:
            card["explanation_source"] = "llm" if "explanation" in card else "rules"
    except Exception:
        for card in cards:
            card["explanation_source"] = "rules"


def recommend(body: Dict[str, Any]) -> Dict[str, Any]:
    req = validate_request(body)
    city_category = [p for p in PROVIDERS if not p["is_synthetic"]
                     and equivalent(req["city"], p.get("city", ""))
                     and any(equivalent(req["category"], c) for c in p["categories_list"])]
    if not city_category:
        return {"status": "no_catalog", "count": 0, "cards": [],
                "message": "В каталоге нет подрядчиков этой категории в городе {}.".format(req["city"]), "reasons": {}}

    rejected: Dict[str, int] = {}
    eligible = []
    for provider in city_category:
        failures = candidate_failures(provider, req)
        if failures:
            for failure in failures:
                rejected[failure] = rejected.get(failure, 0) + 1
        else:
            eligible.append(provider)
    if not eligible:
        reason_summary = ["{} — {}".format(count, FAIL_LABELS[key]) for key, count in sorted(rejected.items(), key=lambda pair: (-pair[1], pair[0]))]
        return {"status": "no_match", "count": 0, "cards": [], "city_category_count": len(city_category),
                "message": "В городе есть {} подрядчика этой категории, но ни один не прошёл условия.".format(len(city_category)),
                "reasons": rejected, "reason_labels": reason_summary}

    query_vector = None
    semantic = {}
    semantic_note = ""
    if API_KEY:
        try:
            vectors = get_provider_embeddings(eligible)
            query_vector = embed([request_text(req)])[0]
            semantic = {p["id"]: cosine(query_vector, vectors[p["id"]]) for p in eligible if p["id"] in vectors}
            if len(semantic) < len(eligible):
                semantic_note = "Семантическая оценка доступна не для всех карточек."
        except Exception:
            semantic_note = "API эмбеддингов сейчас недоступен; применена стабильная сортировка по каталогу."

    def score(provider: Dict[str, Any]) -> Tuple[float, int, str]:
        # The embedding only breaks ties within fully eligible candidates.
        return (round(semantic.get(provider["id"], 0.0), 6), -(provider["price"] or 0), provider["id"])

    eligible.sort(key=score, reverse=True)
    chosen = eligible[:3]
    cards = []
    for p in chosen:
        evidence = evidence_for(p, req)
        cards.append({
            "id": p["id"], "name": p["anon_name"], "category": p.get("categories", ""), "city": p.get("city", ""),
            "price_kzt": p["price"], "price_imputed": p["price_is_imputed"],
            "city_imputed": p["city_is_imputed"], "languages": p["languages_list"],
            "max_hours": p["max_hours_value"], "explanation": fallback_explanation(p, req),
            "evidence": evidence, "explanation_source": "rules",
            "semantic_score": round(semantic[p["id"]], 4) if p["id"] in semantic else None,
        })
    llm_explanations(cards, req)
    count = len(eligible)
    message = "Подобрали {} подрядчика.".format(min(count, 3))
    if count < 3:
        message += " Подходящих меньше трёх: по всем заданным условиям нашлось только {}.".format(count)
    return {"status": "recommendations", "count": count, "cards": cards,
            "message": message, "reasons": {}, "api_enabled": bool(API_KEY), "api_note": semantic_note}


def options() -> Dict[str, List[str]]:
    providers = [p for p in PROVIDERS if not p["is_synthetic"]]
    return {
        "cities": sorted({p["city"] for p in providers if p.get("city")}),
        "categories": sorted({c for p in providers for c in p["categories_list"]}, key=norm),
        "event_types": sorted({c for p in providers for c in p["formats_list"]}, key=norm),
        "languages": sorted({c for p in providers for c in p["languages_list"]}, key=norm),
        "api_enabled": bool(API_KEY),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/options":
            self._send(200, json.dumps(options(), ensure_ascii=False).encode(), "application/json; charset=utf-8")
            return
        if path in ("/", "/index.html"):
            self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/recommend":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 20000:
                raise ValueError("Слишком большой запрос.")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            result = recommend(body)
            self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        except ValueError as exc:
            self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        except Exception:
            self._send(500, json.dumps({"error": "Не удалось обработать запрос. Проверьте каталог и настройки API."}, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


if __name__ == "__main__":
    print("Каталог загружен: {} записей".format(len(PROVIDERS)))
    print("Откройте http://localhost:{}".format(PORT))
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
