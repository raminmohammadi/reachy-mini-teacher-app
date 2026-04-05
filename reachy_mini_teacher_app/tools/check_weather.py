"""Weather tool using Open-Meteo (free, no API key required).

Fetches current weather for a city and returns a Farsi-language description
that the AI can read aloud to the user.

Data sources
------------
- Geocoding : https://geocoding-api.open-meteo.com/v1/search
- Forecast  : https://api.open-meteo.com/v1/forecast
"""

import asyncio
import logging
from typing import Any, Dict

import requests

from reachy_mini_teacher_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

# ── WMO weather-code → Farsi description ──────────────────────────────────────
_WMO_FARSI: Dict[int, str] = {
    0: "آفتابی و صاف",
    1: "عمدتاً صاف",
    2: "کمی ابری",
    3: "ابری",
    45: "مه‌آلود",
    48: "مه یخبندان",
    51: "نم‌نم باران خفیف",
    53: "نم‌نم باران متوسط",
    55: "نم‌نم باران شدید",
    61: "باران خفیف",
    63: "باران متوسط",
    65: "باران شدید",
    71: "برف خفیف",
    73: "برف متوسط",
    75: "برف سنگین",
    77: "دانه‌های برف",
    80: "رگبار خفیف",
    81: "رگبار متوسط",
    82: "رگبار شدید",
    85: "رگبار برف خفیف",
    86: "رگبار برف سنگین",
    95: "طوفان رعد و برق",
    96: "طوفان با تگرگ",
    99: "طوفان با تگرگ سنگین",
}

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_S = 8


def _wmo_to_farsi(code: int) -> str:
    """Return the Farsi weather description for a WMO weather code."""
    return _WMO_FARSI.get(code, "نامشخص")


def _geocode(city: str) -> tuple[float, float, str]:
    """Return (lat, lon, resolved_name) for a city name, or raise ValueError."""
    resp = requests.get(
        _GEOCODING_URL,
        params={"name": city, "count": 1, "language": "fa", "format": "json"},
        timeout=_TIMEOUT_S,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        raise ValueError(f"شهر «{city}» پیدا نشد.")
    r = results[0]
    name = r.get("name", city)
    country = r.get("country", "")
    resolved = f"{name}، {country}" if country else name
    return float(r["latitude"]), float(r["longitude"]), resolved


def _fetch_weather(lat: float, lon: float) -> Dict[str, Any]:
    """Return the current-weather block from Open-Meteo."""
    resp = requests.get(
        _FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": ",".join([
                "temperature_2m",
                "apparent_temperature",
                "relative_humidity_2m",
                "weather_code",
                "wind_speed_10m",
            ]),
            "timezone": "auto",
            "wind_speed_unit": "kmh",
        },
        timeout=_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["current"]


def _build_farsi_report(city_name: str, current: Dict[str, Any]) -> str:
    """Build a natural Farsi weather report from the Open-Meteo current block."""
    temp = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    code = int(current.get("weather_code", -1))
    condition = _wmo_to_farsi(code)

    parts = [f"آب‌وهوای {city_name}:"]
    parts.append(f"وضعیت: {condition}.")
    if temp is not None:
        parts.append(f"دما: {temp:.0f} درجه سانتی‌گراد.")
    if feels is not None and feels != temp:
        parts.append(f"احساس می‌شود: {feels:.0f} درجه.")
    if humidity is not None:
        parts.append(f"رطوبت: {humidity:.0f} درصد.")
    if wind is not None:
        parts.append(f"سرعت باد: {wind:.0f} کیلومتر بر ساعت.")

    return " ".join(parts)


class CheckWeather(Tool):
    """Get the current weather for a city and describe it in Farsi."""

    name = "check_weather"
    description = (
        "Get the current weather for any city and describe it in Farsi. "
        "Use this when the user asks about the weather in a city. "
        "Returns a ready-to-speak Farsi weather report."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": (
                    "Name of the city to check weather for, "
                    "e.g. 'Boston', 'Paris', 'New York'."
                ),
            },
        },
        "required": ["city"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Fetch current weather and return a Farsi description."""
        city: str = (kwargs.get("city") or "").strip()
        if not city:
            return {"error": "نام شهر وارد نشده است.", "farsi_report": "لطفاً نام شهر را بگویید."}

        logger.info("Tool call: check_weather city=%s", city)

        try:
            lat, lon, resolved = await asyncio.to_thread(_geocode, city)
            current = await asyncio.to_thread(_fetch_weather, lat, lon)
            report = _build_farsi_report(resolved, current)
            logger.info("check_weather result: %s", report)
            return {"farsi_report": report, "city": resolved}

        except ValueError as e:
            msg = str(e)
            logger.warning("check_weather geocoding failed: %s", msg)
            return {"error": msg, "farsi_report": msg}

        except requests.RequestException as e:
            logger.error("check_weather HTTP error: %s", e)
            fallback = "متأسفم، در دریافت اطلاعات آب‌وهوا مشکلی پیش آمد. لطفاً دوباره امتحان کنید."
            return {"error": str(e), "farsi_report": fallback}

        except Exception as e:
            logger.exception("check_weather unexpected error: %s", e)
            fallback = "خطایی رخ داد و نتوانستم آب‌وهوا را بررسی کنم."
            return {"error": str(e), "farsi_report": fallback}
