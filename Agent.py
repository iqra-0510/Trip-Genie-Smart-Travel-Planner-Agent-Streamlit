import json
import math
import os
import time
from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-120b"


@lru_cache(maxsize=1)
def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    return Groq(api_key=key)


# ======================================================================
# TOOLS (live data, with offline fallbacks)
# ======================================================================
def http_get_json(url, params=None, timeout=8):
    if params:
        url += "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "travel-planner-agent/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


FALLBACK_RATES = {  # units per 1 USD, used only if the live API is down
    "USD": 1, "PKR": 280, "EUR": 0.92, "GBP": 0.79, "AED": 3.67,
    "SAR": 3.75, "TRY": 34, "THB": 35, "MYR": 4.5,
}


@lru_cache(maxsize=1)
def get_rates():
    try:
        data = http_get_json("https://open.er-api.com/v6/latest/USD")
        if data.get("result") == "success":
            return data["rates"], "live rates"
    except Exception:
        pass
    return FALLBACK_RATES, "offline demo rates"


def convert(amount, from_cur, to_cur):
    rates, _ = get_rates()
    f, t = from_cur.upper(), to_cur.upper()
    if f not in rates or t not in rates:
        raise ValueError(f"Unsupported currency: {f} or {t}")
    return float(amount) * rates[t] / rates[f]


def convert_currency(amount, from_cur, to_cur):
    value = convert(amount, from_cur, to_cur)
    return f"{float(amount):,.2f} {from_cur.upper()} = {value:,.2f} {to_cur.upper()} ({get_rates()[1]})"


@lru_cache(maxsize=32)
def _geocode(city):
    hit = http_get_json("https://geocoding-api.open-meteo.com/v1/search",
                        {"name": city, "count": 1})["results"][0]
    return hit["latitude"], hit["longitude"], hit["name"], hit.get("country", "")


def get_weather(city, days=3):
    check_city(city)
    days = max(1, min(int(days), 14))
    try:
        lat, lon, name, country = _geocode(city)
        d = http_get_json("https://api.open-meteo.com/v1/forecast", {
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": days,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        })["daily"]
    except Exception as e:
        return f"Weather unavailable for {city} ({e}). Do not invent values; say it is unknown."
    lines = [f"Forecast for {name}, {country}:"]
    for i, day in enumerate(d["time"]):
        lines.append(f"Day {i + 1} ({day}): {d['temperature_2m_min'][i]}-{d['temperature_2m_max'][i]}°C, "
                     f"rain chance {d['precipitation_probability_max'][i]}%")
    return "\n".join(lines)


# Demo cost data in USD (per person; hotel per room). Flights = return from Karachi.
COSTS_USD = {
    "dubai":    dict(flight=260, hotel=110, food=40, transport=15, activities=45),
    "istanbul": dict(flight=380, hotel=80,  food=30, transport=10, activities=30),
    "bangkok":  dict(flight=330, hotel=55,  food=20, transport=8,  activities=25),
    "london":   dict(flight=650, hotel=170, food=55, transport=20, activities=40),
    "baku":     dict(flight=300, hotel=65,  food=25, transport=8,  activities=25),
    "kuala lumpur": dict(flight=340, hotel=60, food=22, transport=8, activities=25),
    "paris":    dict(flight=700, hotel=180, food=60, transport=18, activities=45),
}
DEFAULT_COSTS = dict(flight=450, hotel=100, food=35, transport=12, activities=35)

# Only these cities are supported (every tool and agent is limited to them)
SUPPORTED_CITIES = sorted(COSTS_USD)


def check_city(city):
    """Returns the normalised city name or raises if the city is not supported."""
    c = str(city).strip().lower()
    if c not in COSTS_USD:
        raise ValueError(f"'{city}' is not supported. Supported cities: "
                         + ", ".join(x.title() for x in SUPPORTED_CITIES))
    return c
STYLES = {"budget": (0.9, 0.6), "standard": (1.0, 1.0), "luxury": (2.2, 2.2)}  # (flight, rest)


def estimate_trip_costs(destination, days, people, budget_pkr, style="standard"):
    days, people, budget_pkr = int(days), int(people), float(budget_pkr)
    if style.lower() not in STYLES:
        raise ValueError(f"style must be one of {list(STYLES)}")
    c = COSTS_USD[check_city(destination)]
    f_mult, o_mult = STYLES[style.lower()]
    rooms, nights = math.ceil(people / 2), max(days - 1, 1)
    usd = {
        "flights": c["flight"] * people * f_mult,
        "hotel": c["hotel"] * rooms * nights * o_mult,
        "food": c["food"] * people * days * o_mult,
        "transport": c["transport"] * people * days * o_mult,
        "activities": c["activities"] * people * days * o_mult,
    }
    pkr = {k: round(convert(v, "USD", "PKR")) for k, v in usd.items()}
    total = sum(pkr.values())
    return json.dumps({
        "style": style.lower(), "rooms": rooms, "nights": nights, "costs_pkr": pkr,
        "total_pkr": total, "budget_pkr": round(budget_pkr),
        "remaining_pkr": round(budget_pkr - total), "within_budget": total <= budget_pkr,
    })


ATTRACTIONS = {
    "dubai": [("Burj Khalifa - At The Top", "indoor"), ("Dubai Mall & Fountain Show", "indoor"),
              ("Museum of the Future", "indoor"), ("Jumeirah Beach", "outdoor"),
              ("Gold & Spice Souks + Abra ride", "outdoor"), ("Desert Safari", "outdoor"),
              ("Dubai Marina Walk", "outdoor")],
    "istanbul": [("Hagia Sophia", "indoor"), ("Grand Bazaar", "indoor"), ("Topkapi Palace", "indoor"),
                 ("Blue Mosque", "indoor"), ("Bosphorus Cruise", "outdoor"), ("Galata Tower", "outdoor")],
    "bangkok": [("Jim Thompson House", "indoor"), ("Siam Paragon", "indoor"), ("Grand Palace", "outdoor"),
                ("Wat Pho", "outdoor"), ("Chatuchak Market", "outdoor"), ("Chao Phraya River Boat", "outdoor")],
    "london": [("British Museum", "indoor"), ("Tower of London", "indoor"), ("Borough Market", "indoor"),
               ("Westminster & Big Ben", "outdoor"), ("London Eye", "outdoor"), ("Hyde Park", "outdoor")],
    "baku": [("Heydar Aliyev Center", "indoor"), ("Carpet Museum", "indoor"), ("Old City (Icherisheher)", "outdoor"),
             ("Flame Towers", "outdoor"), ("Baku Boulevard", "outdoor"), ("Gobustan & Mud Volcanoes", "outdoor")],
    "kuala lumpur": [("Petronas Towers & KLCC", "indoor"), ("Islamic Arts Museum", "indoor"),
                     ("Pavilion KL Mall", "indoor"), ("Batu Caves", "outdoor"),
                     ("Jalan Alor Food Street", "outdoor"), ("KL Tower", "outdoor")],
    "paris": [("Louvre Museum", "indoor"), ("Musee d'Orsay", "indoor"), ("Eiffel Tower", "outdoor"),
              ("Seine River Cruise", "outdoor"), ("Montmartre & Sacre-Coeur", "outdoor"), ("Versailles", "outdoor")],
}


def get_attractions(destination, kind="all"):
    items = ATTRACTIONS[check_city(destination)]
    if kind != "all":
        items = [i for i in items if i[1] == kind]
    return "\n".join(f"- {name} ({k})" for name, k in items)


def packing_list(city, days=3):
    """Builds a packing checklist from the real forecast of the destination."""
    check_city(city)
    days = max(1, min(int(days), 14))
    try:
        lat, lon, name, country = _geocode(city)
        d = http_get_json("https://api.open-meteo.com/v1/forecast", {
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": days,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        })["daily"]
    except Exception as e:
        return f"Weather unavailable for {city} ({e}). Give a generic packing list and say the weather is unknown."
    lo, hi = min(d["temperature_2m_min"]), max(d["temperature_2m_max"])
    rain = max(d["precipitation_probability_max"])
    items = ["Passport + visa/ticket copies", "Phone charger & power bank", "Travel adapter",
             "Some cash in local currency", "Basic medicines"]
    if hi >= 30:
        items += ["Light cotton clothes", "Sunscreen & sunglasses", "Hat/cap", "Water bottle"]
    if lo <= 15:
        items += ["Warm jacket", "Scarf & gloves"]
    elif lo <= 22:
        items += ["Light jacket or sweater"]
    if rain >= 40:
        items += ["Umbrella / rain jacket", "Waterproof shoes"]
    items.append("Comfortable walking shoes")
    return (f"Packing list for {name}, {country} ({lo}-{hi}°C, max rain chance {rain}%):\n"
            + "\n".join(f"- {i}" for i in items))


def _schema(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = {
    "get_weather": (get_weather, _schema(
        "get_weather", "Real daily weather forecast for a city.",
        {"city": {"type": "string"}, "days": {"type": "integer"}}, ["city"])),
    "convert_currency": (convert_currency, _schema(
        "convert_currency", "Convert an amount between currencies (e.g. USD to PKR).",
        {"amount": {"type": "number"}, "from_cur": {"type": "string"}, "to_cur": {"type": "string"}},
        ["amount", "from_cur", "to_cur"])),
    "estimate_trip_costs": (estimate_trip_costs, _schema(
        "estimate_trip_costs", "Trip cost split in PKR plus whether it fits the budget.",
        {"destination": {"type": "string"}, "days": {"type": "integer"}, "people": {"type": "integer"},
         "budget_pkr": {"type": "number"}, "style": {"type": "string", "enum": list(STYLES)}},
        ["destination", "days", "people", "budget_pkr"])),
    "packing_list": (packing_list, _schema(
        "packing_list", "Packing checklist based on the real weather forecast of a city.",
        {"city": {"type": "string"}, "days": {"type": "integer"}}, ["city"])),
    "get_attractions": (get_attractions, _schema(
        "get_attractions", "Attractions for a destination tagged indoor/outdoor.",
        {"destination": {"type": "string"}, "kind": {"type": "string", "enum": ["all", "indoor", "outdoor"]}},
        ["destination"])),
}


def call_tool(name, args):
    if name not in TOOLS:
        return f"ERROR: unknown tool '{name}'"
    try:
        return str(TOOLS[name][0](**args))
    except Exception as e:
        return f"ERROR: {e}"


# ======================================================================
# AGENTS
# ======================================================================
def llm(messages, retries=4, **kwargs):
    delay = 2
    for attempt in range(1, retries + 1):
        try:
            return get_client().chat.completions.create(
                model=MODEL, messages=messages, max_completion_tokens=2500, **kwargs
            ).choices[0].message
        except Exception:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2


def parse_json(text):
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


class Agent:
    def __init__(self, name, role, tools=(), json_mode=False):
        self.name, self.tools, self.json_mode = name, tools, json_mode
        self.schemas = [TOOLS[t][1] for t in tools]
        self.system = f"You are the {name}. {role}"
        if tools:
            self.system += "\nNever guess weather, prices or currency values - always use your tools."

    def run(self, task, context="", log=print):
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": f"Context:\n{context}\n\nTask: {task}"}]

        if not self.tools:  # plain single call
            extra = {"response_format": {"type": "json_object"}} if self.json_mode else {}
            return (llm(messages, **extra).content or "").strip()

        used_tool = False
        for _ in range(8):  # tool loop; first turn must use a tool
            msg = llm(messages, tools=self.schemas, tool_choice="auto" if used_tool else "required")
            if not msg.tool_calls:
                return (msg.content or "").strip()

            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = call_tool(tc.function.name, args)
                used_tool = True
                log(f"🔧 {self.name} used `{tc.function.name}` -> {result[:100]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return "Stopped: max tool steps reached"


budget_agent = Agent(
    "Budget Agent",
    "Produce a cost split in PKR (flights, hotel, food, transport, activities, total, remaining). "
    "Call estimate_trip_costs with the exact trip facts and the travel style from TRIP FACTS. If over budget, retry with style 'budget' "
    "and suggest savings. Use a short markdown table. Keep it under 200 words.",
    tools=("estimate_trip_costs", "convert_currency"),
)
writer_agent = Agent(
    "Writer",
    "Produce a clear day-by-day trip plan (Day 1, Day 2, ...) with morning/afternoon/evening. "
    "Check the weather with the tool, then use get_attractions: indoor places for hot or rainy days, "
    "outdoor for pleasant ones. Respect the budget in the context. Keep it under 200 words.",
    tools=("get_weather", "get_attractions"),
)
reviewer_agent = Agent(
    "Reviewer",
    "Check that (1) the budget is within the trip budget, in PKR, and adds up, and (2) the plan covers "
    "every day, matches the weather and fits the budget. "
    'Reply ONLY with JSON: {"approved": true or false, "budget_issues": "...", "plan_issues": "..."}. '
    "Leave an issues string empty when that part is fine.",
    json_mode=True,
)
WORKERS = {"budget": budget_agent, "writer": writer_agent}

PLANNER = """You are the orchestrator. Split the goal into steps for these workers: budget, writer.
The budget step must come first, then the writer step.
Reply ONLY with JSON:
{"steps": [{"worker": "budget", "task": "..."}, {"worker": "writer", "task": "..."}]}"""


def make_plan(goal):
    default = [{"worker": "budget", "task": f"Create the PKR cost split for: {goal}"},
               {"worker": "writer", "task": f"Create the day-by-day plan for: {goal}"}]
    try:
        steps = parse_json(llm([{"role": "system", "content": PLANNER},
                                {"role": "user", "content": goal}],
                               response_format={"type": "json_object"}).content)["steps"]
        steps = [s for s in steps if s.get("worker") in WORKERS and s.get("task")]
        if {s["worker"] for s in steps} != set(WORKERS):
            return default
        return sorted(steps, key=lambda s: list(WORKERS).index(s["worker"]))
    except Exception:
        return default


def parse_review(text):
    try:
        d = parse_json(text)
        return bool(d.get("approved")), str(d.get("budget_issues", "")).strip(), str(d.get("plan_issues", "")).strip()
    except Exception:
        return True, "", ""  # bad reviewer output -> accept current draft


# ======================================================================
# MAIN ENTRY POINT (call this from Streamlit)
# ======================================================================
def plan_trip(origin, destination, days, people, budget_pkr, max_revisions=2, on_event=None, style="standard"):
    """
    Returns {"budget": str, "plan": str, "status": str}.
    on_event: optional function(str) for live progress, e.g. st.write or status.write.
    """
    log = on_event or (lambda msg: None)
    check_city(destination)
    goal = (f"Plan a {days}-day trip to {destination} for {people} people from {origin}. "
            f"Budget is {budget_pkr} PKR. Travel style: {style}.")
    facts = (f"TRIP FACTS: origin={origin}, destination={destination}, days={days}, "
             f"people={people}, budget_pkr={budget_pkr}, style={style}")
    context, results = facts, {}

    # 1. Orchestrator delegates to workers
    for step in make_plan(goal):
        worker = WORKERS[step["worker"]]
        log(f"➡️ Hand-off to **{worker.name}**: {step['task']}")
        results[step["worker"]] = worker.run(step["task"], context, log)
        context += f"\n\n[{worker.name}]\n{results[step['worker']]}"

    # 2. Reviewer loop - each agent fixes only its own part
    status = "Not reviewed"
    for i in range(1, max_revisions + 1):
        bundle = f"{facts}\n\n[BUDGET]\n{results['budget']}\n\n[PLAN]\n{results['writer']}"
        approved, b_issues, p_issues = parse_review(reviewer_agent.run("Review the budget and the plan.", bundle))
        log(f"🔍 Reviewer round {i}: {'approved ✅' if approved else 'changes needed'}")

        if approved or not (b_issues or p_issues):
            status = f"Approved after {i - 1} revision(s)"
            break
        if b_issues:
            results["budget"] = budget_agent.run(
                f"Fix these issues and return the FULL budget in PKR:\n{b_issues}",
                f"{facts}\n\n[BUDGET]\n{results['budget']}", log)
        if p_issues:
            results["writer"] = writer_agent.run(
                f"Fix these issues and return the FULL day-by-day plan (not the budget):\n{p_issues}",
                f"{facts}\n\n[BUDGET]\n{results['budget']}\n\n[PLAN]\n{results['writer']}", log)
        status = f"Revised {i}x"

    return {"budget": results["budget"], "plan": results["writer"], "status": status}


if __name__ == "__main__":  # quick test: python travel_planner.py
    out = plan_trip("Karachi", "Dubai", 3, 2, 400000, on_event=print)
    print("\n", out["budget"], "\n\n", out["plan"], "\n\n", out["status"])