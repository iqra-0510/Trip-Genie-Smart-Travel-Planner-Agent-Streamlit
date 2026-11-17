"""
TripGenie - Smart Travel Planner Chatbot (Streamlit)

Run:  streamlit run app.py
Needs Agent.py in the same folder and GROQ_API_KEY in .env
(or in .streamlit/secrets.toml, or paste it in the sidebar).

How it works:
  - A chat agent (LLM + tools) talks to the user and REMEMBERS the conversation.
  - If trip details are missing (destination, days, people, budget) it asks for them.
  - Once it has everything it calls the `plan_trip` tool, which runs your multi-agent
    pipeline (Orchestrator -> Budget -> Writer -> Reviewer).
  - For quick questions it uses weather / attractions / currency / cost tools directly.
  - Follow-ups like "make it 5 days" or "cheaper please" reuse the remembered trip.
"""
import json
import os

import streamlit as st

st.set_page_config(page_title="TripGenie - Smart Travel Planner", page_icon="🧭", layout="centered")

# ---- API key: .env / env var -> Streamlit secrets -> sidebar input ---------
if not os.getenv("GROQ_API_KEY"):
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

from Agent import SUPPORTED_CITIES, TOOLS, call_tool, check_city, llm, plan_trip  # noqa: E402

CITY_LIST = ", ".join(c.title() for c in SUPPORTED_CITIES)

MAX_HISTORY = 20   # last N chat messages sent to the model (conversation memory)
MAX_AGENT_STEPS = 6

GREETING = (
    "Hello! I'm **TripGenie**, your smart travel planning assistant 🧭\n\n"
    "Tell me where you want to go and I'll build a **budget** and a **day-by-day plan**. "
    "I can also check the weather, suggest attractions, make a packing list and convert currencies.\n\n"
    f"**Supported destinations:** {CITY_LIST}.\n\n"
    "Try: *\"Plan a 3-day trip to Dubai for 2 people, budget 400000 PKR\"*"
)
EXAMPLES = [
    "Plan a 3-day trip to Dubai for 2 people, budget 400000 PKR",
    "What's the weather like in Istanbul this week?",
    "I want to go to Bangkok",
    "What should I pack for Istanbul for 4 days?",
]

# ---- session state (this is the chatbot's memory) --------------------------
ss = st.session_state
ss.setdefault("messages", [{"role": "assistant", "content": GREETING}])  # what the UI shows
ss.setdefault("history", [])      # clean text turns sent to the LLM
ss.setdefault("trip", None)       # last planned trip details
ss.setdefault("last_plan", None)  # last generated budget + plan (markdown)


def reset_chat():
    ss.messages = [{"role": "assistant", "content": GREETING}]
    ss.history, ss.trip, ss.last_plan = [], None, None


def set_pending(text):
    ss.pending_prompt = text


# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    if not os.getenv("GROQ_API_KEY"):
        key = st.text_input("Groq API key", type="password")
        if key:
            os.environ["GROQ_API_KEY"] = key
    st.slider("Reviewer rounds", 0, 3, 2, key="revisions",
              help="How many times the Reviewer agent may send the plan back for fixes (0 = skip review).")
    st.selectbox("Travel style", ["budget", "standard", "luxury"], index=1, key="style",
                 help="Budget = cheaper hotels/food, Luxury = premium everything.")
    st.toggle("Show agent activity", value=True, key="show_logs")
    st.button("🗑️ New chat", on_click=reset_chat, use_container_width=True)

    st.divider()
    st.subheader("🧠 Trip memory")
    if ss.trip:
        t = ss.trip
        st.markdown(f"**From:** {t['origin']}  \n**To:** {t['destination']}  \n"
                    f"**Days:** {t['days']}  \n**People:** {t['people']}  \n"
                    f"**Budget:** {t['budget_pkr']:,} PKR  \n**Style:** {t.get('style', 'standard')}")
        st.metric("Trip budget", f"{t['budget_pkr']:,} PKR",
                  help="Per person: " + f"{t['budget_pkr'] // t['people']:,} PKR")
    else:
        st.caption("No trip planned yet.")
    if ss.last_plan:
        st.download_button("⬇️ Download plan", ss.last_plan, file_name="trip_plan.md",
                           mime="text/markdown", use_container_width=True)

st.title("🧭 TripGenie")
st.caption("Your multi-agent trip planner: live weather, exchange rates and smart budgeting")

st.sidebar.caption("TripGenie v1.1 · Powered by Groq")

if not os.getenv("GROQ_API_KEY"):
    st.info("Add your **GROQ_API_KEY** (in `.env` or the sidebar) to start chatting.")
    st.stop()

# ---- agent tools -------------------------------------------------------------
PLAN_TOOL = {"type": "function", "function": {
    "name": "plan_trip",
    "description": "Run the full multi-agent pipeline (budget + day-by-day plan + review). "
                   "Call ONLY when destination, days, people and budget are all known.",
    "parameters": {"type": "object", "properties": {
        "origin": {"type": "string", "description": "Departure city, default Karachi"},
        "destination": {"type": "string"},
        "days": {"type": "integer"},
        "people": {"type": "integer"},
        "budget_pkr": {"type": "number", "description": "Total trip budget in PKR"},
        "style": {"type": "string", "enum": ["budget", "standard", "luxury"],
                  "description": "Travel style, use the sidebar default unless the user asks otherwise"},
    }, "required": ["destination", "days", "people", "budget_pkr"]},
}}
CHAT_TOOLS = [PLAN_TOOL] + [TOOLS[n][1] for n in
                            ("get_weather", "get_attractions", "convert_currency", "estimate_trip_costs", "packing_list")]


def system_prompt():
    trip = json.dumps(ss.trip) if ss.trip else "none yet"
    last = ss.last_plan or "none yet"
    return f"""You are a friendly, concise travel-planning assistant. Users are usually travelling from Pakistan \
(default origin: Karachi, currency: PKR). Format answers in markdown.

RULES
- To plan a trip you need: destination, number of days, number of people, total budget. \
If any is missing, ask for ALL missing ones in one short message. Never invent them. Origin defaults to Karachi.
- If the budget is given in another currency, convert it to PKR with convert_currency first.
- When you have everything, call plan_trip. It shows the full budget and itinerary to the user itself, \
so afterwards just write a short summary (2-3 sentences: total vs budget) and offer tweaks. Do NOT repeat the plan.
- For changes ("make it 5 days", "cheaper", "add one more person") call plan_trip again with the \
remembered trip details plus the change.
- The default travel style is "{ss.style}" (budget / standard / luxury). Pass it to plan_trip unless the user asks for another.
- For quick questions use get_weather, get_attractions, convert_currency, estimate_trip_costs or packing_list. \
Never guess weather, prices or exchange rates - use the tools.
- For other travel questions (tips, food, transport) answer from general knowledge. For visas and entry \
rules say they should verify on the official embassy/immigration website.
- Questions about the previously generated plan: answer from LAST PLAN below.
- ONLY these destinations are supported: {CITY_LIST}. If the user asks about any other city or country \
(plans, weather, attractions, tips, food, anything), do NOT answer about it and do not call tools for it. \
Politely say it is not supported and list the supported cities.
- Politely steer off-topic questions back to travel.

REMEMBERED TRIP: {trip}

LAST PLAN:
{last}"""


def run_plan_tool(args, status, attachments):
    """Executes the plan_trip tool and stores results in memory."""
    try:
        origin = str(args.get("origin") or "Karachi").strip()
        destination = str(args["destination"]).strip()
        check_city(destination)
        days = max(1, min(int(args["days"]), 14))
        people = max(1, min(int(args["people"]), 20))
        budget = int(float(args["budget_pkr"]))
        style = str(args.get("style") or ss.style).lower()
        if style not in ("budget", "standard", "luxury"):
            style = "standard"
        if budget <= 0:
            raise ValueError("budget must be positive")
    except Exception as e:
        return f"ERROR: invalid or missing details ({e}). Ask the user for valid details; only these cities are supported: {CITY_LIST}."

    status.update(label=f"Planning {days}-day trip to {destination}...", state="running")
    try:
        res = plan_trip(origin, destination, days, people, budget,
                        max_revisions=ss.revisions, on_event=status.write, style=style)
    except Exception as e:
        return f"ERROR: planning failed ({e}). Apologise and suggest trying again."

    ss.trip = {"origin": origin, "destination": destination, "days": days,
               "people": people, "budget_pkr": budget, "style": style}
    md = (f"### 💰 Budget (PKR): {style} style\n{res['budget']}\n\n"
          f"### 🗓️ Day-by-day plan: {destination}\n{res['plan']}\n\n"
          f"*Review status: {res['status']}*")
    ss.last_plan = md
    attachments.append(md)
    return ("The full budget and itinerary were already shown to the user. Do not repeat them. "
            f"Budget summary for your reply:\n{res['budget']}")


def run_agent(user_text, status):
    """One chat turn: LLM decides whether to ask questions, call tools, or answer."""
    msgs = ([{"role": "system", "content": system_prompt()}]
            + ss.history[-MAX_HISTORY:]
            + [{"role": "user", "content": user_text}])
    attachments, final = [], None

    for _ in range(MAX_AGENT_STEPS):
        msg = llm(msgs, tools=CHAT_TOOLS, tool_choice="auto")
        if not msg.tool_calls:
            final = (msg.content or "").strip()
            break

        msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if tc.function.name == "plan_trip":
                result = run_plan_tool(args, status, attachments)
            else:
                result = call_tool(tc.function.name, args)
                status.write(f"🔧 `{tc.function.name}` {args} -> {result[:120]}")
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    if final is None:
        final = "Sorry, I got stuck on that one. Could you rephrase or give me a bit more detail?"
    if not final and attachments:
        final = "Let me know if you'd like any changes: more days, a lower budget, different activities. 😊"

    ss.history += [{"role": "user", "content": user_text},
                   {"role": "assistant", "content": final}]
    return "\n\n".join(attachments + ([final] if final else []))


# ---- chat UI -----------------------------------------------------------------
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if len(ss.messages) == 1 and "pending_prompt" not in ss:
    st.caption("Quick start:")
    for ex in EXAMPLES:
        st.button(ex, on_click=set_pending, args=(ex,))

prompt = st.chat_input("Plan a trip, check weather, convert currency...")
if not prompt:
    prompt = ss.pop("pending_prompt", None)

if prompt:
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=ss.show_logs)
        try:
            reply = run_agent(prompt, status)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            reply = f"⚠️ Something went wrong: {e}\n\nPlease try again."
            status.update(label="Error", state="error")
        st.markdown(reply)
    ss.messages.append({"role": "assistant", "content": reply})
    st.rerun()  # refresh sidebar memory + hide quick-start buttons
