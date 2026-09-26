"""
ML Challenge 2026: Business Entity Resolution
Ultra-Fast Normalization & Text Processing Module
"""

import re
import anyascii

LEGAL_TERMS = {
    # US / UK / Common
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "private", "llp", "co", "company", "plc", "gmbh",
    # India
    "private limited", "pvt ltd", "p limited", "pvtltd",
    # France
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "gie",
    # Generic entity descriptors
    "holding", "holdings", "group", "groupe", "enterprises", "enterprise",
    "solutions", "services", "technologies", "technology", "associates",
    "international", "sons", "fils", "brothers", "freres", "centre", "center",
    "club", "ecole", "foundation", "trust", "society", "association"
}

US_STATES = {
    "new york": "ny", "california": "ca", "texas": "tx", "pennsylvania": "pa",
    "florida": "fl", "illinois": "il", "ohio": "oh", "georgia": "ga",
    "north carolina": "nc", "michigan": "mi", "new jersey": "nj", "virginia": "va",
    "washington": "wa", "arizona": "az", "massachusetts": "ma", "tennessee": "tn",
    "indiana": "in", "missouri": "mo", "maryland": "md", "wisconsin": "wi",
    "colorado": "co", "minnesota": "mn", "south carolina": "sc", "alabama": "al",
    "louisiana": "la", "kentucky": "ky", "oregon": "or", "oklahoma": "ok",
    "connecticut": "ct", "utah": "ut", "iowa": "ia", "nevada": "nv",
    "arkansas": "ar", "mississippi": "ms", "kansas": "ks", "new mexico": "nm",
    "nebraska": "ne", "idaho": "id", "west virginia": "wv", "hawaii": "hi",
    "new hampshire": "nh", "maine": "me", "montana": "mt", "rhode island": "ri",
    "delaware": "de", "south dakota": "sd", "north dakota": "nd", "alaska": "ak",
    "vermont": "vt", "wyoming": "wy"
}

INDIA_STATES = {
    "tamil nadu": "tn", "maharashtra": "mh", "uttar pradesh": "up", "karnataka": "ka",
    "delhi": "dl", "haryana": "hr", "gujarat": "gj", "west bengal": "wb",
    "rajasthan": "rj", "telangana": "tg", "andhra pradesh": "ap", "kerala": "kl",
    "punjab": "pb", "bihar": "br", "odisha": "od", "orissa": "od", "jharkhand": "jh",
    "assam": "as", "uttarakhand": "uk", "chhattisgarh": "cg", "goa": "ga",
    "himachal pradesh": "hp", "jammu and kashmir": "jk", "chandigarh": "ch"
}

ADDR_EXPANSIONS = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "lane": "ln", "drive": "dr", "court": "ct", "highway": "hwy",
    "floor": "fl", "apartment": "apt", "building": "bldg", "suite": "ste",
    "rue": "rd", "r": "rd", "boulevard": "blvd", "bd": "blvd", "bvd": "blvd",
    "allee": "ln", "all": "ln", "chemin": "rd", "ch": "rd", "place": "pl",
    "cours": "ave", "cs": "ave", "route": "rd", "rt": "rd",
    "bis": "b", "ter": "t"
}

ADDR_STOPWORDS = {
    "near", "opp", "opposite", "floor", "road", "rd", "street", "st", "avenue", "ave",
    "lane", "ln", "block", "blk", "building", "bldg", "apartment", "apt", "flat", "plot",
    "sector", "sec", "phase", "nagar", "colony", "town", "city", "district", "dist",
    "state", "north", "south", "east", "west", "cross", "main", "first", "second", "third",
    "rue", "r", "boulevard", "bd", "all", "allee", "chemin", "place", "pl", "cours", "cs",
    "hwy", "ct", "dr"
}

DOMAIN_SUFFIX_REGEX = re.compile(r"\.(?:com|org|net|in|co\.in|fr|io|biz|info|gov|edu|eu)$")
DIGITS_REGEX = re.compile(r"\d+")
CLEAN_PUNCT_REGEX = re.compile(r"[^a-z0-9\s]")

US_STATE_REGEX = re.compile(r"\b(" + "|".join(map(re.escape, sorted(US_STATES.keys(), key=len, reverse=True))) + r")\b")
INDIA_STATE_REGEX = re.compile(r"\b(" + "|".join(map(re.escape, sorted(INDIA_STATES.keys(), key=len, reverse=True))) + r")\b")


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = anyascii.anyascii(text).lower()
    text = re.sub(r"^(?:https?:\/\/)?(?:www\.)?", "", text)
    text = DOMAIN_SUFFIX_REGEX.sub("", text)
    text = CLEAN_PUNCT_REGEX.sub(" ", text)
    tokens = [w for w in text.split() if w]
    return " ".join(tokens)


def get_core_name(text: str) -> str:
    cleaned = clean_text(text)
    tokens = [w for w in cleaned.split() if w not in LEGAL_TERMS]
    return " ".join(tokens) if tokens else cleaned


def get_compact_name(cn: str) -> str:
    return cn.replace(" ", "")


def extract_name_tokens(cn: str) -> list[str]:
    return [t for t in cn.split() if len(t) >= 3]


def normalize_address(addr: str, country: str = "") -> str:
    if not addr or not isinstance(addr, str):
        return ""
    cleaned = clean_text(addr)
    if country == "US":
        cleaned = US_STATE_REGEX.sub(lambda m: US_STATES[m.group(0)], cleaned)
    elif country == "India":
        cleaned = INDIA_STATE_REGEX.sub(lambda m: INDIA_STATES[m.group(0)], cleaned)
    words = cleaned.split()
    norm_words = [ADDR_EXPANSIONS.get(w, w) for w in words]
    return " ".join(norm_words)


def extract_address_digits(norm_addr: str) -> set[str]:
    if not norm_addr:
        return set()
    raw_nums = DIGITS_REGEX.findall(norm_addr)
    res = set()
    for n in raw_nums:
        if len(n) <= 6:
            try:
                val = int(n)
                if val > 0:
                    res.add(str(val))
            except ValueError:
                pass
    return res


def extract_address_keys(norm_addr: str) -> list[str]:
    if not norm_addr:
        return []
    tokens = norm_addr.split()
    digits = extract_address_digits(norm_addr)
    words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS]
    keys = []
    for num in list(digits)[:2]:
        for w in words[:3]:
            keys.append(f"{num}_{w}")
    for i in range(min(len(words) - 1, 2)):
        keys.append(f"{words[i]}_{words[i+1]}")
    return keys
