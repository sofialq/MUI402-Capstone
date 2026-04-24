import streamlit as st
from anthropic import Anthropic, RateLimitError  
import re
import textwrap
import time 
from io import StringIO

# client
if "claude_client" not in st.session_state:
    st.session_state.claude_client = Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])

client: Anthropic = st.session_state.claude_client

# constants
MODEL = "claude-sonnet-4-6"
SUMMARY_MODEL = "claude-haiku-4-5-20251001"  # cheaper model for summaries
MAX_TOKENS = 2000
TOKEN_BUFFER = 2000
SUMMARY_AFTER = 5
SUMMARY_HISTORY_LIMIT = 10  # only send last N messages to summarizer

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

For every event you mention, use 2-3 sentences max per field, be concise and include:
  • Event name and type (festival / sport / music)
  • Confirmed dates and city/venue
  • Why it's worth attending

After {SUMMARY_AFTER} exchanges, offer a structured tour summary with all stops, dates,
and the full travel flow. Don't return artist profile. Never exceed 800 generated tokens in a single response. "If the itinerary has more than 5 stops, 
ALWAYS split into parts and ask before continuing. Never generate more than 5 stops in a single response."
"""

WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}

# initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "display" not in st.session_state:
    st.session_state.display = []
if "exchanges" not in st.session_state:
    st.session_state.exchanges = 0
if "pending_summary" not in st.session_state:
    st.session_state.pending_summary = False
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None
if "last_reply" not in st.session_state:
    st.session_state.last_reply = None


# helper functions
def token_trimmed_history(history: list, max_words: int = TOKEN_BUFFER) -> list:
    kept, budget = [], max_words
    for msg in reversed(history):
        content = msg["content"]
        words = len(str(content))
        if budget - words < 0 and kept:
            break
        kept.insert(0, msg)
        budget -= words
    return kept


def extract_text(content) -> str:
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
    return re.sub(r"</?cite[^>]*>", "", text).strip()


# retry wrapper with exponential backoff
def call_with_retry(fn, retries=3, base_delay=10):
    for attempt in range(retries):
        try:
            return fn()
        except RateLimitError:
            if attempt == retries - 1:
                raise
            wait = base_delay * (2 ** attempt)  # 10s → 20s → 40s
            st.toast(f"Rate limit hit — retrying in {wait}s…", icon="⏳")
            time.sleep(wait)


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

    st.session_state.history.append({
        "role": "user",
        "content": [{"type": "text", "text": user_text}],
    })

    trimmed = token_trimmed_history(st.session_state.history)

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

    # wrap the API call with retry logic
    response = call_with_retry(lambda: client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=[WEB_SEARCH_TOOL],
        messages=trimmed,
    ))

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
    # only send the last N messages to keep input tokens low
    recent = st.session_state.history[-SUMMARY_HISTORY_LIMIT:]
    convo = "\n".join(
        f"{m['role']}: {extract_text(m['content'])}"
        for m in recent
    )

    result = call_with_retry(lambda: client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=400,
        system=(
            "Summarise the following tour-planning conversation in 4–5 sentences, "
            "highlighting events discussed, destinations, routing logic, and any itinerary agreed."
        ),
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": convo}],
        }],
    ))
    return extract_text(result.content)


def build_markdown_tour_summary(summary_text: str, last_reply: str | None) -> str:
    md = ["# Tour Summary\n"]
    if summary_text:
        md.append(summary_text)
        md.append("\n")
    if last_reply:
        md.append("## Latest Itinerary Draft\n")
        md.append(last_reply)
    return "\n".join(md).strip()


# ui setup
st.set_page_config(page_title="TourBot", layout="wide")

# layout
col_main, col_summary = st.columns([2.5, 1.5])

with col_main:
    # deferred summary- runs after main call
    if st.session_state.pending_summary:
        st.session_state.pending_summary = False
        time.sleep(5) 
        try:
            summary_text = generate_summary()
            st.session_state.summary = summary_text
            st.session_state.history = [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"Summary so far:\n{summary_text}"}],
                }
            ]
        except RateLimitError:
            st.toast("Summary skipped — rate limit hit. Will retry next exchange.")
        st.rerun()  # rerun once more so the summary panel updates

    st.title("TourBot")
    st.caption("Help plan tours with the fans in mind · powered by Claude + web search")

# sidebar
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
    if st.button("Clear conversation"):
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
        for i, msg in enumerate(st.session_state.display):
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])

                if msg["role"] == "assistant" and "Would you like" in msg["text"]:
                    col1, col2 = st.columns(2)

                    if col1.button("Yes, continue", key=f"yes_{i}"):
                        user_reply = "Yes, continue."

                        st.session_state.display.append({"role": "user", "text": user_reply})
                        st.session_state.history.append({
                            "role": "user",
                            "content": [{"type": "text", "text": user_reply}],
                        })

                        with st.chat_message("assistant"):
                            with st.spinner("Continuing…"):
                                reply = call_claude(
                                    user_text=user_reply,
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
                        st.rerun()

                    if col2.button("No, stop", key=f"no_{i}"):
                        user_reply = "No, stop."
                        st.session_state.display.append({"role": "user", "text": user_reply})
                        st.session_state.history.append({
                            "role": "user",
                            "content": [{"type": "text", "text": user_reply}],
                        })
                        st.rerun()

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
                    try:
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
                    except RateLimitError:
                        st.warning("Rate limit reached mid-response. Please wait 30 seconds and click Regenerate.")
                        st.stop()

            # summary flag
            st.session_state.display.append({"role": "assistant", "text": reply})

            if st.session_state.exchanges == 1 or (
                st.session_state.exchanges > 0 and st.session_state.exchanges % SUMMARY_AFTER == 0
            ):
                st.session_state.pending_summary = True

# summary panel
with col_summary:
    st.subheader("Tour Summary: ")
    if st.session_state.summary:
        with st.expander("Show / Hide Summary", expanded = False):
            st.markdown(st.session_state.summary)
    else:
        st.caption("Once you've generated a tour plan, a high-level tour summary will appear here.")

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

