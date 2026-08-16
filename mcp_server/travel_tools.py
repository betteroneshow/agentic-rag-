# -*- coding: utf-8 -*-
"""实时天气与地图路线 MCP Server。

默认使用无需密钥的 Open-Meteo 和 OSRM；配置 AMAP_API_KEY 后，
国内地点解析及驾车/步行/公交路线会优先使用高德 Web Service API。
"""

from __future__ import annotations

import math
import os
from datetime import date
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

AMAP_API_KEY = os.getenv("AMAP_API_KEY", "").strip()
HTTP_TIMEOUT = float(os.getenv("MCP_HTTP_TIMEOUT", "20"))
USER_AGENT = os.getenv("MCP_USER_AGENT", "AgentiRAG-MCP/1.0")

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
AMAP_GEOCODING_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_DRIVING_URL = "https://restapi.amap.com/v3/direction/driving"
AMAP_WALKING_URL = "https://restapi.amap.com/v3/direction/walking"
AMAP_TRANSIT_URL = "https://restapi.amap.com/v3/direction/transit/integrated"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"

mcp = FastMCP(
    "agentic-rag-travel-tools",
    instructions="提供实时天气、天气预报、地点解析、距离计算和路线规划。",
)

WEATHER_CODES = {
    0: "晴朗", 1: "大部晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "雾凇", 51: "小毛毛雨", 53: "中等毛毛雨",
    55: "强毛毛雨", 56: "轻微冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "轻微冻雨",
    67: "强冻雨", 71: "小雪", 73: "中雪", 75: "大雪",
    77: "米雪", 80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "强阵雪", 95: "雷暴", 96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=HTTP_TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        follow_redirects=True,
    )


def _get_json(url: str, params: dict[str, Any]) -> Any:
    try:
        with _client() as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"外部服务请求失败: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("外部服务返回的不是有效 JSON") from exc


def _amap_geocode(location: str, city: str | None = None) -> dict | None:
    if not AMAP_API_KEY:
        return None
    params = {"key": AMAP_API_KEY, "address": location, "output": "JSON"}
    if city:
        params["city"] = city
    payload = _get_json(AMAP_GEOCODING_URL, params)
    if payload.get("status") != "1" or not payload.get("geocodes"):
        return None
    item = payload["geocodes"][0]
    longitude, latitude = map(float, item["location"].split(","))
    return {
        "name": item.get("formatted_address") or location,
        "latitude": latitude,
        "longitude": longitude,
        "country": "中国",
        "admin1": item.get("province") or "",
        "admin2": item.get("city") or "",
        "district": item.get("district") or "",
        "provider": "amap",
        "coordinate_system": "GCJ-02",
    }


def _open_meteo_geocode(location: str, count: int = 5) -> list[dict]:
    payload = _get_json(
        OPEN_METEO_GEOCODING_URL,
        {"name": location, "count": count, "language": "zh", "format": "json"},
    )
    results = []
    for item in payload.get("results") or []:
        results.append({
            "name": item.get("name") or location,
            "latitude": float(item["latitude"]),
            "longitude": float(item["longitude"]),
            "country": item.get("country") or "",
            "admin1": item.get("admin1") or "",
            "admin2": item.get("admin2") or "",
            "district": item.get("admin3") or "",
            "timezone": item.get("timezone") or "auto",
            "provider": "open-meteo",
            "coordinate_system": "WGS-84",
        })
    return results


def _coordinate_location(value: str) -> dict | None:
    """接受 `纬度,经度`，避免 POI 服务不可用时产生猜测。"""
    try:
        latitude_text, longitude_text = [part.strip() for part in value.split(",")]
        latitude, longitude = float(latitude_text), float(longitude_text)
    except (ValueError, TypeError):
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("坐标超出范围，请使用 纬度,经度 格式")
    return {
        "name": value, "latitude": latitude, "longitude": longitude,
        "country": "", "admin1": "", "admin2": "", "district": "",
        "provider": "coordinates", "coordinate_system": "WGS-84",
    }


def _nominatim_geocode(location: str, city: str | None = None, count: int = 5) -> list[dict]:
    query = f"{location}, {city}" if city and city not in location else location
    payload = _get_json(
        NOMINATIM_SEARCH_URL,
        {
            "q": query, "format": "jsonv2", "limit": min(max(count, 1), 10),
            "accept-language": "zh-CN", "addressdetails": 1,
            "countrycodes": "cn" if any("\u4e00" <= char <= "\u9fff" for char in query) else "",
        },
    )
    results = []
    for item in payload if isinstance(payload, list) else []:
        address = item.get("address") or {}
        results.append({
            "name": item.get("display_name") or item.get("name") or location,
            "latitude": float(item["lat"]), "longitude": float(item["lon"]),
            "country": address.get("country") or "",
            "admin1": address.get("state") or address.get("province") or "",
            "admin2": address.get("city") or address.get("municipality") or address.get("county") or "",
            "district": address.get("city_district") or address.get("district") or "",
            "category": item.get("category") or "", "type": item.get("type") or "",
            "importance": item.get("importance"), "provider": "OpenStreetMap Nominatim",
            "coordinate_system": "WGS-84",
        })
    return results


def _resolve_location(location: str, city: str | None = None, prefer_amap: bool = False) -> dict:
    if not location.strip():
        raise ValueError("地点不能为空")
    coordinate = _coordinate_location(location)
    if coordinate:
        return coordinate
    if prefer_amap:
        result = _amap_geocode(location, city)
        if result:
            return result
    results = _nominatim_geocode(location, city, count=1)
    if not results:
        raise ValueError(f"没有找到地点：{location}")
    return results[0]


def _haversine_km(origin: dict, destination: dict) -> float:
    lat1, lon1 = math.radians(origin["latitude"]), math.radians(origin["longitude"])
    lat2, lon2 = math.radians(destination["latitude"]), math.radians(destination["longitude"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(value))


def _seconds_text(seconds: float | int) -> str:
    minutes = max(0, round(float(seconds) / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours}小时{minutes}分钟" if hours else f"{minutes}分钟"


def _weather_location(location: str) -> dict:
    # Open-Meteo 天气数据使用 WGS-84；Nominatim 对中文城市和 POI 更准确。
    return _resolve_location(location, prefer_amap=False)


@mcp.tool()
def geocode_location(location: str, city: str | None = None) -> dict:
    """将地点名称解析为经纬度。适合在路线规划前确认同名地点。

    Args:
        location: 城市、景点、车站或详细地址。
        city: 可选的城市限定词，例如“长沙”。
    """
    primary = _resolve_location(location, city, prefer_amap=bool(AMAP_API_KEY))
    alternatives = _nominatim_geocode(location, city, count=5)
    return {"query": location, "best_match": primary, "alternatives": alternatives}


@mcp.tool()
def get_current_weather(location: str) -> dict:
    """查询指定城市或景点当前的实时天气、体感温度、降水和风况。"""
    place = _weather_location(location)
    payload = _get_json(
        OPEN_METEO_WEATHER_URL,
        {
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "timezone": "auto",
        },
    )
    current = payload.get("current") or {}
    units = payload.get("current_units") or {}
    return {
        "location": place,
        "observed_at": current.get("time"),
        "weather": WEATHER_CODES.get(current.get("weather_code"), "未知"),
        "temperature": current.get("temperature_2m"),
        "temperature_unit": units.get("temperature_2m", "°C"),
        "apparent_temperature": current.get("apparent_temperature"),
        "relative_humidity_percent": current.get("relative_humidity_2m"),
        "precipitation_mm": current.get("precipitation"),
        "rain_mm": current.get("rain"),
        "cloud_cover_percent": current.get("cloud_cover"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "wind_direction_degree": current.get("wind_direction_10m"),
        "wind_gusts_kmh": current.get("wind_gusts_10m"),
        "is_day": bool(current.get("is_day")),
        "provider": "Open-Meteo",
    }


@mcp.tool()
def get_weather_forecast(location: str, days: int = 3, start_date: str | None = None) -> dict:
    """查询未来1至14天逐日天气预报。

    Args:
        location: 城市或景点名称。
        days: 返回天数，范围1到14。
        start_date: 可选开始日期，格式 YYYY-MM-DD，必须在可预报范围内。
    """
    if not 1 <= days <= 14:
        raise ValueError("days 必须在 1 到 14 之间")
    if start_date:
        date.fromisoformat(start_date)
    place = _weather_location(location)
    params: dict[str, Any] = {
        "latitude": place["latitude"], "longitude": place["longitude"],
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max",
        "timezone": "auto", "forecast_days": days,
    }
    if start_date:
        params["start_date"] = start_date
        params.pop("forecast_days")
        from datetime import timedelta
        params["end_date"] = (date.fromisoformat(start_date) + timedelta(days=days - 1)).isoformat()
    payload = _get_json(OPEN_METEO_WEATHER_URL, params)
    daily = payload.get("daily") or {}
    forecasts = []
    for index, day in enumerate(daily.get("time") or []):
        value = lambda key: (daily.get(key) or [None] * (index + 1))[index]
        code = value("weather_code")
        forecasts.append({
            "date": day, "weather": WEATHER_CODES.get(code, "未知"),
            "temperature_max_c": value("temperature_2m_max"),
            "temperature_min_c": value("temperature_2m_min"),
            "apparent_temperature_max_c": value("apparent_temperature_max"),
            "apparent_temperature_min_c": value("apparent_temperature_min"),
            "precipitation_sum_mm": value("precipitation_sum"),
            "precipitation_probability_max_percent": value("precipitation_probability_max"),
            "wind_speed_max_kmh": value("wind_speed_10m_max"),
            "wind_gusts_max_kmh": value("wind_gusts_10m_max"),
            "sunrise": value("sunrise"), "sunset": value("sunset"),
        })
    return {"location": place, "forecast": forecasts, "provider": "Open-Meteo"}


def _amap_route(origin: dict, destination: dict, mode: str, city: str | None) -> dict:
    coordinates = {
        "origin": f"{origin['longitude']:.6f},{origin['latitude']:.6f}",
        "destination": f"{destination['longitude']:.6f},{destination['latitude']:.6f}",
        "key": AMAP_API_KEY, "output": "JSON",
    }
    if mode == "transit":
        if not city:
            city = origin.get("admin2") or origin.get("admin1")
        if not city:
            raise ValueError("公交路线需要提供 city 参数")
        payload = _get_json(AMAP_TRANSIT_URL, {**coordinates, "city": city})
        if payload.get("status") != "1" or not (payload.get("route") or {}).get("transits"):
            raise RuntimeError(f"高德公交路线失败: {payload.get('info') or '无可用方案'}")
        item = payload["route"]["transits"][0]
        instructions = []
        for segment in item.get("segments") or []:
            walking = segment.get("walking") or {}
            instructions.extend(step.get("instruction") for step in walking.get("steps") or [] if step.get("instruction"))
            for busline in (segment.get("bus") or {}).get("buslines") or []:
                instructions.append(
                    f"乘坐{busline.get('name', '公共交通')}，从{(busline.get('departure_stop') or {}).get('name', '起点站')}到{(busline.get('arrival_stop') or {}).get('name', '终点站')}"
                )
        return {
            "distance_meters": int(float(item.get("distance") or 0)),
            "duration_seconds": int(float(item.get("duration") or 0)),
            "cost_yuan": item.get("cost"), "walking_distance_meters": item.get("walking_distance"),
            "instructions": instructions[:30], "provider": "Amap",
        }
    endpoint = AMAP_WALKING_URL if mode == "walking" else AMAP_DRIVING_URL
    payload = _get_json(endpoint, coordinates)
    paths = (payload.get("route") or {}).get("paths") or []
    if payload.get("status") != "1" or not paths:
        raise RuntimeError(f"高德路线失败: {payload.get('info') or '无可用方案'}")
    item = paths[0]
    return {
        "distance_meters": int(float(item.get("distance") or 0)),
        "duration_seconds": int(float(item.get("duration") or 0)),
        "tolls_yuan": item.get("tolls"),
        "instructions": [step["instruction"] for step in item.get("steps") or [] if step.get("instruction")][:30],
        "provider": "Amap",
    }


def _osrm_route(origin: dict, destination: dict) -> dict:
    coordinates = (
        f"{origin['longitude']:.6f},{origin['latitude']:.6f};"
        f"{destination['longitude']:.6f},{destination['latitude']:.6f}"
    )
    payload = _get_json(
        f"{OSRM_ROUTE_URL}/{coordinates}",
        {"overview": "false", "steps": "true", "alternatives": "false"},
    )
    if payload.get("code") != "Ok" or not payload.get("routes"):
        raise RuntimeError(f"OSRM 路线失败: {payload.get('message') or payload.get('code')}")
    route = payload["routes"][0]
    instructions = []
    for leg in route.get("legs") or []:
        for step in leg.get("steps") or []:
            maneuver = step.get("maneuver") or {}
            road = step.get("name") or "未命名道路"
            instructions.append(f"{maneuver.get('type', '行驶')}：{road}（{round(step.get('distance', 0))}米）")
    return {
        "distance_meters": round(route["distance"]),
        "duration_seconds": round(route["duration"]),
        "instructions": instructions[:30], "provider": "OSRM",
    }


@mcp.tool()
def plan_route(
    origin: str,
    destination: str,
    mode: Literal["driving", "walking", "transit", "straight"] = "driving",
    city: str | None = None,
) -> dict:
    """计算两地直线距离，并规划驾车、步行或公交路线。

    Args:
        origin: 起点名称或详细地址。
        destination: 终点名称或详细地址。
        mode: driving驾车、walking步行、transit公交、straight仅直线距离。
        city: 同名地点或公交路线的城市限定，例如“长沙”。
    """
    use_amap = bool(AMAP_API_KEY) and mode != "straight"
    start = _resolve_location(origin, city, prefer_amap=use_amap)
    end = _resolve_location(destination, city, prefer_amap=use_amap)
    straight_km = round(_haversine_km(start, end), 3)
    result = {
        "origin": start, "destination": end, "mode": mode,
        "straight_distance_km": straight_km,
    }
    if mode == "straight":
        result["provider"] = "Haversine"
        return result
    if use_amap:
        route = _amap_route(start, end, mode, city)
    else:
        if mode != "driving":
            raise ValueError("步行和公交路线需要配置 AMAP_API_KEY；未配置时仅支持 driving 或 straight")
        route = _osrm_route(start, end)
    result.update(route)
    result["distance_km"] = round(result["distance_meters"] / 1000, 3)
    result["duration_text"] = _seconds_text(result["duration_seconds"])
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
