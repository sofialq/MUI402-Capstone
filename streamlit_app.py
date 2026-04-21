import streamlit as st
from anthropic import Anthropic

# client
if "claude_client" not in st.session_state:
    st.session_state.claude_client = Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])

client: Anthropic = st.session_state.claude_client

# constants
MODEL = "claude-opus-4-5"
MAX_TOKENS = 4096
TOKEN_BUFFER = 2000
SUMMARY_AFTER = 15   # exchanges before auto-summary

SYSTEM_PROMPT = f"""\
You are TourBot, an expert tour organizer involved in the planning and logistics of bands 
and artists' touring schedules. The user will ask you to help plan tours around festivals, sporting events,
and graduation ceremonies (if relevant) worldwide. Respond as a tour planner working with a band manager to build exciting, 
efficient itineraries that hit the best events for the band's fanbase.

Use the web_search tool to find up-to-date information on events, lineups, and schedules, as well as more information on the artist's background.

You have access to a web_search tool. Look up the listed information and use it to avoid dates in these cities if there are overlapping events:
- The artist, their genre, and fanbase
- Current festival lineups, dates, and locations
- Sporting event schedules, venues, and location details
- Tour routing between multiple events
- Graduation dates for surrounding cities, if band has a college-based fanbase; avoid these dates

When choosing cities, consider where the artist's fanbase is strongest and where they have the most demand. Also consider if travel
is possible for fans who's cities aren't hosting an event, but are nearby a city that is.

Do not send artist to a festival or event that doesn't fit their genre or fanbase. 
Prioritise events that are a strong match, and flag any scheduling conflicts or logistical issues with proposed itineraries.

For every event you mention, include:
  • Event name and type (festival / sport / music)
  • Confirmed dates and city/venue
  • Why it's worth attending
  • Practical travel tip (nearest airport, booking lead time, etc.)

When building an itinerary:
  - Order events chronologically
  - Suggest a logical routing to minimise travel
  - Flag scheduling conflicts
  - Give rough travel times between stops

Keep your tone enthusiastic and conversational. Assume budget and travel styles based on 
previous tour data.

After {SUMMARY_AFTER} exchanges, offer a structured tour summary with all stops, dates,
and the full travel flow.
"""

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}

# initialise session state
if "history" not in st.session_state:
    st.session_state.history = []          # list[dict] – raw API messages
if "display" not in st.session_state:
    st.session_state.display = []          # list[dict] – {role, text}
if "exchanges" not in st.session_state:
    st.session_state.exchanges = 0
if "summary" not in st.session_state:
    st.session_state.summary = ""

# helper functions
def token_trimmed_history(history: list, max_words: int = TOKEN_BUFFER) -> list:
    """Keep as many recent messages as fit within the word budget."""
    kept, budget = [], max_words
    for msg in reversed(history):
        content = msg["content"]
        # content is usually a list of blocks; approximate length via str()
        words = len(str(content))
        if budget - words < 0:
            break
        kept.insert(0, msg)
        budget -= words
    return kept


def extract_text(content) -> str:
    """Pull plain text out of a Claude content block (list or string)."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
        elif hasattr(block, "type") and block.type == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def call_claude(user_text: str) -> str:
    """Send the conversation to Claude (with web search) and return the reply."""
    st.session_state.history.append({
        "role": "user",
        "content": [{"type": "text", "text": user_text}],
    })

    trimmed = token_trimmed_history(st.session_state.history)

    system = SYSTEM_PROMPT
    if st.session_state.summary:
        system += f"\n\nConversation summary so far:\n{st.session_state.summary}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=[WEB_SEARCH_TOOL],
        messages=trimmed,
    )

    reply = extract_text(response.content)

    # Store the raw content block so tool-use turns survive history trimming
    st.session_state.history.append(
        {"role": "assistant", "content": response.content}
    )
    st.session_state.exchanges += 1
    return reply


def generate_summary() -> str:
    convo = "\n".join(
        f"{m['role']}: {extract_text(m['content'])}"
        for m in st.session_state.history
    )
    result = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=(
            "Summarise the following tour-planning conversation in 4–5 sentences, "
            "highlighting events discussed, destinations, and any itinerary agreed."
        ),
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": convo}],
        }],
    )
    return extract_text(result.content)

# user interface
st.set_page_config(page_title="TourBot", page_icon="🗺️", layout="centered")

st.markdown("""
<style>
.chip-row { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:1rem; }
.chip {
    display:inline-block; padding:4px 12px;
    border-radius:20px; font-size:13px; cursor:pointer;
    border:1px solid #ccc; background:#f7f7f7;
}
</style>
""", unsafe_allow_html=True)

st.title("🗺️ TourBot")
st.caption("Plan tours around festivals & sporting events · powered by Claude + web search")

# chat history display 
if not st.session_state.display:
    with st.chat_message("assistant"):
        st.markdown(
            "Hi! I'm **TourBot** — your personal tour organizer for your artist's concerts.\n\n"
            "Tell me more about your artist and what your goals are with this tour, and I can help you plan an exciting itinerary that hits the best events for your fanbase!"
        )
else:
    for msg in st.session_state.display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

# sidebar inputs- tour details
with st.sidebar:
    st.title("Tour Detail")
    st.caption("Fill in these details to help guide your tour routing.")
    st.divider()

    st.subheader("Artist description")
    artist = st.text_input("Artist/band name:")
    artist_genre = st.text_input("Genre:")

    st.subheader("Timeframe:")
    timeframe_options = [
        "Summer 2026 (Jun–Aug)",
        "Fall 2026 (Sep–Nov)",
        "Winter 2026 (Dec–Feb)",
        "Spring 2027 (Mar–May)",
        "Custom range",
        "no preference",
    ]
    timeframe = st.selectbox("When are you looking to tour?", timeframe_options)
    if timeframe == "Custom range":
        col1, col2 = st.columns(2)
        with col1:
            start_month = st.date_input("Start date")
        with col2:
            end_month = st.date_input("End date")

    st.subheader("Tour Length:")
    tour_length = st.slider(
        "How many cities/stops are you looking to include?",
        min_value=1, max_value=30, value=10, step=1
    )

    st.subheader("Region")
    region = st.radio(
        "Where are we focusing?",
        options=[
            "US only",
            "Europe",
            "Mix of international destinations",
            "Specific countries or regions",
        ],
    )
    specific_regions = ""
    if region == "Specific countries or regions":
        specific_regions = st.text_input(
            "List countries or regions",
            placeholder="e.g., UK, Germany, Australia",
        )

    st.subheader("Targeted Fanbase")
    fanbase = st.text_area(
        "Describe your core audience",
        placeholder="e.g., college-age listeners 18–28",
        height=100,
    )

    st.subheader("Must-Hit Cities or Events")
    must_hit = st.text_area(
        "Places your artist has always wanted to play, or markets with the strongest fanbase",
        placeholder="e.g., Austin TX, Nashville TN, Glastonbury UK...",
        height=100,
    )

# create tour plan 
if st.button("Create my tour plan."):
    prompt = ( {SYSTEM_PROMPT} +
        f"Your artist is {artist or 'the artist'} and their music is best described as "
        f"{artist_genre or 'their genre'}. The target fanbase is {fanbase or 'their fans'}.\n"
        f"The tour should focus on {region}"
        f"{' (' + specific_regions + ')' if specific_regions else ''} during {timeframe}.\n"
        f"Make sure to include any must-hit cities or events: {must_hit or 'none specified'}.\n"
        f"The tour should be around {tour_length} stops long. "
        f"Can you help me plan an exciting tour itinerary based on this information?"
    )

    st.session_state.display.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the web…"):
            reply = call_claude(prompt)
        st.markdown(reply)

    st.session_state.display.append({"role": "assistant", "text": reply})

    # auto-summary after a set number of exchanges to keep conversation manageable and give user a checkpoint
    if st.session_state.exchanges > 0 and st.session_state.exchanges % SUMMARY_AFTER == 0:
        with st.chat_message("assistant"):
            with st.spinner("Generating conversation summary…"):
                summary_text = generate_summary()
            st.session_state.summary = summary_text
            summary_msg = f"**Tour summary so far:**\n\n{summary_text}"
            st.markdown(summary_msg)
            st.session_state.display.append({"role": "assistant", "text": summary_msg})

            # compress history to summary only (as a message)
            st.session_state.history = [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"Summary so far:\n{summary_text}"}],
                }
            ]


