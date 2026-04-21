import streamlit as st
from anthropic import Anthropic
import re
import textwrap
from io import StringIO

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
is possible for fans whose cities aren't hosting an event, but are nearby a city that is.

Do not send the artist to a festival or event that doesn't fit their genre or fanbase. 
Prioritise events that are a strong match, and clearly flag any scheduling conflicts or logistical issues with proposed itineraries.

When building an itinerary:
  - Order events chronologically
  - Suggest a logical routing to minimise travel and avoid backtracking
  - Prefer routes with reasonable travel times between stops
  - Flag scheduling conflicts
  - Give rough travel times between stops (flight/train/drive)

For every event you mention, include:
  • Event name and type (festival / sport / music)
  • Confirmed dates and city/venue
  • Why it's worth attending
  • Practical travel tip (nearest airport, booking lead time, etc.)

After {SUMMARY_AFTER} exchanges, offer a structured tour summary with all stops, dates,
and the full travel flow.
"""

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}

# inirialize session state
if "history" not in st.session_state:
    st.session_state.history = []          # list[dict] – raw API messages
if "display" not in st.session_state:
    st.session_state.display = []          # list[dict] – {role, text}
if "exchanges" not in st.session_state:
    st.session_state.exchanges = 0
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None
if "last_reply" not in st.session_state:
    st.session_state.last_reply = None

# helper functions
def token_trimmed_history(history: list, max_words: int = TOKEN_BUFFER) -> list:
    """Keep as many recent messages as fit within the word budget."""
    kept, budget = [], max_words
    for msg in reversed(history):
        content = msg["content"]
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


def strip_citations(text: str) -> str:
    """Remove <cite> tags while keeping inner text."""
    return re.sub(r"</?cite[^>]*>", "", text).strip()


def build_dynamic_context(
    artist: str,
    artist_genre: str,
    fanbase: str,
    region: str,
    specific_regions: str,
    timeframe: str,
    must_hit: str,
    tour_length: int,
) -> str:
    return textwrap.dedent(
        f"""
        Artist: {artist or "unknown"}
        Genre: {artist_genre or "unknown"}
        Fanbase: {fanbase or "unspecified"}
        Region: {region}
        Specific regions: {specific_regions or "none"}
        Timeframe: {timeframe}
        Must-hit cities/events: {must_hit or "none"}
        Target tour length: {tour_length} stops

        Routing constraints:
        - Minimise backtracking between cities
        - Prefer geographically efficient sequences
        - Avoid overlapping major events in the same city unless strategically beneficial
        - Only include festivals/events that fit the artist's genre and fanbase
        """
    ).strip()


def call_claude(user_text: str,
                artist: str,
                artist_genre: str,
                fanbase: str,
                region: str,
                specific_regions: str,
                timeframe: str,
                must_hit: str,
                tour_length: int) -> str:
    """Send the conversation to Claude (with web search) and return the reply."""

    # add user message
    st.session_state.history.append({
        "role": "user",
        "content": [{"type": "text", "text": user_text}],
    })

    trimmed = token_trimmed_history(st.session_state.history)

    # dynamic system prompt with tour context
    dynamic_context = build_dynamic_context(
        artist=artist,
        artist_genre=artist_genre,
        fanbase=fanbase,
        region=region,
        specific_regions=specific_regions,
        timeframe=timeframe,
        must_hit=must_hit,
        tour_length=tour_length,
    )

    system = SYSTEM_PROMPT + "\n\n" + dynamic_context

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
    reply = strip_citations(reply)

    st.session_state.history.append({
        "role": "assistant",
        "content": response.content
    })
    st.session_state.exchanges += 1

    st.session_state.last_prompt = user_text
    st.session_state.last_reply = reply

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
            "highlighting events discussed, destinations, routing logic, and any itinerary agreed."
        ),
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": convo}],
        }],
    )
    return extract_text(result.content)


def build_markdown_tour_summary(summary_text: str, last_reply: str | None) -> str:
    """Create a markdown export of the tour summary + latest itinerary."""
    md = ["# Tour Summary\n"]
    if summary_text:
        md.append(summary_text)
        md.append("\n")
    if last_reply:
        md.append("## Latest Itinerary Draft\n")
        md.append(last_reply)
    return "\n".join(md).strip()


# ui setup
st.set_page_config(page_title="TourBot", page_icon="🗺️", layout="wide")

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

# layout: main chat + right summary panel
col_main, col_summary = st.columns([2.5, 1.5])

with col_main:
    st.title("🗺️ TourBot")
    st.caption("Help plan tours with the fans in mind · powered by Claude + web search")

# sidebar inputs
with st.sidebar:
    st.title("Tour Detail")
    st.caption("Fill in these details to guide your tour routing.")
    st.divider()

    st.subheader("Artist description")
    artist = st.text_input("Artist/band name:")
    artist_genre = st.text_input("Genre:")

    st.subheader("Timeframe")
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

    st.subheader("Tour Length")
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

    st.divider()
    if st.button("🗑️ Clear conversation"):
        for key in ["history", "display", "exchanges", "summary", "last_prompt", "last_reply"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

# main chat interface
with col_main:
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

    # create tour plan
    create_clicked = st.button("Create my tour plan", type="primary")
    regen_clicked = st.button("Regenerate itinerary", disabled=st.session_state.last_prompt is None)

    if create_clicked or regen_clicked:
        if not artist or not artist_genre:
            with st.chat_message("assistant"):
                st.markdown(
                    "To build a strong tour plan, I need at least the **artist name** and **genre**. "
                    "Fill those in on the left and click again."
                )
        else:
            if create_clicked:
                prompt = (
                    f"Your artist is {artist} and their music is best described as "
                    f"{artist_genre}. The target fanbase is {fanbase or 'their fans'}.\n"
                    f"The tour should focus on {region}"
                    f"{' (' + specific_regions + ')' if specific_regions else ''} during {timeframe}.\n"
                    f"Make sure to include any must-hit cities or events: {must_hit or 'none specified'}.\n"
                    f"The tour should be around {tour_length} stops long. "
                    f"Please propose a genre-appropriate, geographically efficient tour itinerary that balances festivals and headline shows."
                )
            else:
                # regenerate using same context but explicit instruction
                base_prompt = st.session_state.last_prompt or ""
                prompt = (
                    base_prompt
                    + "\n\nPlease regenerate an alternative tour itinerary with different routing and event choices, "
                      "while still respecting genre fit, fanbase, and travel efficiency."
                )

            st.session_state.display.append({"role": "user", "text": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Planning your tour…"):
                    reply = call_claude(
                        user_text=prompt,
                        artist=artist,
                        artist_genre=artist_genre,
                        fanbase=fanbase,
                        region=region,
                        specific_regions=specific_regions,
                        timeframe=timeframe,
                        must_hit=must_hit,
                        tour_length=tour_length,
                    )
                st.markdown(reply)

            st.session_state.display.append({"role": "assistant", "text": reply})

            # auto-summary
            if st.session_state.exchanges > 0 and st.session_state.exchanges % SUMMARY_AFTER == 0:
                with st.chat_message("assistant"):
                    with st.spinner("Generating conversation summary…"):
                        summary_text = generate_summary()
                    st.session_state.summary = summary_text
                    summary_msg = f"**Tour summary so far:**\n\n{summary_text}"
                    st.markdown(summary_msg)
                    st.session_state.display.append({"role": "assistant", "text": summary_msg})

                    st.session_state.history = [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": f"Summary so far:\n{summary_text}"}],
                        }
                    ]

# tour summary panel + export
with col_summary:
    st.subheader("Tour Summary: ")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.caption("Once you've had a few exchanges, a high-level tour summary will appear here.")

    st.divider()
    st.subheader("Export")

    if st.session_state.summary or st.session_state.last_reply:
        md_content = build_markdown_tour_summary(
            st.session_state.summary,
            st.session_state.last_reply,
        )
        buffer = StringIO(md_content)
        st.download_button(
            label="Download tour plan (Markdown)",
            data=buffer.getvalue(),
            file_name="tour_plan.md",
            mime="text/markdown",
        )
    else:
        st.caption("Generate a tour plan first to enable export.")
