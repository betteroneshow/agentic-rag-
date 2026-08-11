# -*- coding: utf-8 -*-
"""Clean the travel guide workbook into destination-level and attraction-level data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ATTRACTION_PATTERN = re.compile(
    r"(?:^|\s*)(?:\d+[\.、])\s*(?:\*\*)?([^：:\n*]+?)(?:\*\*)?[：:]\s*"
    r"(.*?)(?=(?:\s*\d+[\.、]\s*(?:\*\*)?[^：:\n*]+(?:\*\*)?[：:])|$)",
    re.S,
)
QUOTED_NAME_PATTERN = re.compile(r"[『「“《]([^』」”》]{2,20})[』」”》]")
SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？；;]\s*")
LEADING_NAME_PATTERN = re.compile(
    r"^([^，,。！!？?、]{2,20}?)(?:是|一定|也很|绝对|最|可谓|算是|值得|超级|特别|非常|不容|必去|必打卡)"
)
REGION_PREFIXES = [
    "内蒙古",
    "黑龙江",
    "新疆",
    "广西",
    "宁夏",
    "西藏",
    "北京",
    "天津",
    "上海",
    "重庆",
    "河北",
    "山西",
    "辽宁",
    "吉林",
    "江苏",
    "浙江",
    "安徽",
    "福建",
    "江西",
    "山东",
    "河南",
    "湖北",
    "湖南",
    "广东",
    "海南",
    "四川",
    "贵州",
    "云南",
    "陕西",
    "甘肃",
    "青海",
    "台湾",
    "香港",
    "澳门",
]
CITY_REGION_PREFIXES = {
    "黄山": "安徽",
    "丽江": "云南",
    "张家界": "湖南",
    "敦煌": "甘肃",
    "桂林": "广西",
    "昆明": "云南",
    "杭州": "浙江",
    "成都": "四川",
    "青岛": "山东",
    "腾冲": "云南",
    "厦门": "福建",
    "西安": "陕西",
    "苏州": "江苏",
    "拉萨": "西藏",
    "福州": "福建",
    "甘南": "甘肃",
    "阿勒泰": "新疆",
    "呼伦贝尔": "内蒙古",
    "张掖": "甘肃",
    "哈尔滨": "黑龙江",
    "三亚": "海南",
    "张家口": "河北",
    "深圳": "广东",
    "兰州": "甘肃",
    "大连": "辽宁",
    "喀什": "新疆",
    "九寨沟": "四川",
    "阿坝": "四川",
    "香格里拉": "云南",
    "阿尔山": "内蒙古",
    "西双版纳": "云南",
    "西宁": "青海",
    "呼和浩特": "内蒙古",
    "千岛湖": "浙江",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def split_destination(raw_destination: str) -> tuple[str, str]:
    destination = clean_text(raw_destination)
    parts = destination.split()
    if len(parts) >= 2:
        return parts[0], parts[-1]
    for prefix in REGION_PREFIXES:
        if destination.startswith(prefix) and len(destination) > len(prefix):
            return prefix, destination[len(prefix) :]
    for prefix, region in CITY_REGION_PREFIXES.items():
        if destination.startswith(prefix) or prefix in destination:
            return region, destination
    return "未知地区", destination


def parse_attractions(text: str) -> list[tuple[str, str]]:
    text = clean_text(text)
    matches = ATTRACTION_PATTERN.findall(text)
    if matches:
        return [(clean_text(name), clean_text(desc)) for name, desc in matches]

    attractions: list[tuple[str, str]] = []
    seen = set()

    for sentence in SENTENCE_SPLIT_PATTERN.split(text):
        sentence = clean_text(sentence)
        if not sentence:
            continue

        names = [clean_text(name) for name in QUOTED_NAME_PATTERN.findall(sentence)]
        leading_match = LEADING_NAME_PATTERN.search(sentence)
        if leading_match:
            names.insert(0, clean_text(leading_match.group(1)))

        for name in names:
            if not name or name in seen:
                continue
            seen.add(name)
            attractions.append((name, sentence))

    if attractions:
        return attractions

    # Last-resort fallback: keep one attraction-like record so the destination can
    # still be retrieved from the attraction knowledge base.
    return [("必打卡景点", text)] if text else []


def build_destination_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for idx, row in df.iterrows():
        region, destination_name = split_destination(row["目的地"])
        attractions = parse_attractions(row.get("必打卡景点", ""))
        rows.append(
            {
                "doc_id": f"destination_{idx + 1:04d}",
                "doc_type": "destination",
                "region": region,
                "destination": destination_name,
                "full_destination": clean_text(row["目的地"]),
                "transportation": clean_text(row.get("交通安排", "")),
                "accommodation": clean_text(row.get("住宿推荐", "")),
                "attraction_names": "、".join(name for name, _ in attractions),
                "food": clean_text(row.get("美食推荐", "")),
                "tips": clean_text(row.get("实用小贴士", "")),
                "travel_notes": clean_text(row.get("旅行感悟", "")),
                "source_row": int(idx) + 2,
            }
        )
    return rows


def build_attraction_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for idx, row in df.iterrows():
        region, destination_name = split_destination(row["目的地"])
        attractions = parse_attractions(row.get("必打卡景点", ""))
        for attraction_idx, (name, description) in enumerate(attractions, start=1):
            rows.append(
                {
                    "doc_id": f"attraction_{idx + 1:04d}_{attraction_idx:02d}",
                    "doc_type": "attraction",
                    "region": region,
                    "destination": destination_name,
                    "full_destination": clean_text(row["目的地"]),
                    "attraction": name,
                    "description": description,
                    "transportation_context": clean_text(row.get("交通安排", "")),
                    "food_context": clean_text(row.get("美食推荐", "")),
                    "tips_context": clean_text(row.get("实用小贴士", "")),
                    "source_row": int(idx) + 2,
                    "source_attraction_index": attraction_idx,
                }
            )
    return rows


def write_outputs(source: Path, output_dir: Path) -> None:
    df = pd.read_excel(source).dropna(how="all")
    output_dir.mkdir(parents=True, exist_ok=True)

    destination_df = pd.DataFrame(build_destination_rows(df))
    attraction_df = pd.DataFrame(build_attraction_rows(df))

    destination_path = output_dir / "travel_destinations.xlsx"
    attraction_path = output_dir / "travel_attractions.xlsx"
    destination_csv_path = output_dir / "travel_destinations.csv"
    attraction_csv_path = output_dir / "travel_attractions.csv"

    destination_df.to_excel(destination_path, index=False)
    attraction_df.to_excel(attraction_path, index=False)
    destination_df.to_csv(destination_csv_path, index=False, encoding="utf-8-sig")
    attraction_df.to_csv(attraction_csv_path, index=False, encoding="utf-8-sig")

    print(f"source rows: {len(df)}")
    print(f"destination rows: {len(destination_df)} -> {destination_path}")
    print(f"attraction rows: {len(attraction_df)} -> {attraction_path}")
    print(f"csv mirrors: {destination_csv_path}, {attraction_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="travel_guide.xlsx")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    write_outputs(Path(args.source), Path(args.output_dir))


if __name__ == "__main__":
    main()
