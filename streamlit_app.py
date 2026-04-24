import streamlit as st
from anthropic import Anthropic, RateLimitError  
import re
import time 
from io import StringIO

# client
if "claude_client" not in st.session_state:
    st.session_state.claude_client = Anthropic(api_key=st.secrets["CLAUDE_API_KEY"])

client: Anthropic = st.session_state.claude_client

# constants
MODEL = "claude-sonnet-4-6"
SUMMARY_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS_ITINERARY = 750
MAX_TOKENS_CHAT = 400
SUMMARY_AFTER = 5
SUMMARY_HISTORY_LIMIT = 6
MAX_HISTORY_MESSAGES = 6

SYSTEM_PROMPT = f"""\
You are TourBot, a tour logistics expert for bands and artists. Plan tours around festivals, \
sports events, and graduation ceremonies worldwide. Use web_search for current event data, \
lineups, schedules, and artist background.

STRICT OUTPUT FORMAT — copy this exactly for every stop, nothing else:

## STOP N — City, State
**Type** | Date | Venue | 🚗 Xh from [prev city] by [method]
Why: [max 20 words on genre/fanbase fit]

HARD RULES:
- "Why:" field is capped at 20 words. Count them. Cut if over.
- No routing notes. No extra paragraphs. No blockquotes. No corrections sections.
- Chronological order only — if a date conflict exists, silently fix the order.
- Max 5 stops per response, then ask: "Would you like me to continue with stops 6–10?"
- After {SUMMARY_AFTER} exchanges, offer a structured tour summary.
- Genre-appropriate events only. Flag scheduling conflicts inline, not in separate sections.
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
def trim_history(history: list) -> list:
    """Keep only the last MAX_HISTORY_MESSAGES messages."""
    return history[-MAX_HISTORY_MESSAGES:]


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


def call_with_retry(fn, retries=3, base_delay=10):
    for attempt in range(retries):
        try:
            return fn()
        except RateLimitError:
            if attempt == retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            st.toast(f"Rate limit hit — retrying in {wait}s…", icon="⏳")
            time.sleep(wait)


def build_dynamic_context(
    artist, artist_genre, fanbase, region, specific_regions, timeframe, must_hit, tour_length
) -> str:
    return (
        f"Artist: {artist or 'unknown'} | Genre: {artist_genre or 'unknown'} | "
        f"Fanbase: {fanbase or 'unspecified'}\n"
        f"Region: {region}{' (' + specific_regions + ')' if specific_regions else ''} | "
        f"Timeframe: {timeframe} | Stops: {tour_length}\n"
        f"Must-hit: {must_hit or 'none'}\n"
        f"Routing: minimize backtracking, genre-fit only, no city overlaps unless strategic."
    ).strip()


def call_claude(
    user_text: str,
    artist: str,
    artist_genre: str,
    fanbase: str,
    region: str,
    specific_regions: str,
    timeframe: str,
    must_hit: str,
    tour_length: int,
    is_itinerary: bool = False,
) -> str:

    # append strict format reminder to every user message
    user_text_with_reminder = (
        user_text
        + "\n\n[FORMAT ENFORCEMENT: Use the exact 3-line stop format. "
        "'Why:' field = 20 words max. No extra paragraphs, routing notes, blockquotes, "
        "or corrections sections. Chronological order only. Max 5 stops.]"
    )

    st.session_state.history.append({
        "role": "user",
        "content": [{"type": "text", "text": user_text_with_reminder}],
    })

    trimmed = trim_history(st.session_state.history)

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

    max_tokens = MAX_TOKENS_ITINERARY if is_itinerary else MAX_TOKENS_CHAT

    response = call_with_retry(lambda: client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        tools=[WEB_SEARCH_TOOL],
        messages=trimmed,
    ))

    reply = extract_text(response.content)
    reply = strip_citations(reply)

    # store clean text only — never raw response.content
    st.session_state.history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": reply}],
    })
    st.session_state.exchanges += 1

    st.session_state.last_prompt = user_text
    st.session_state.last_reply = reply

    return reply


def generate_summary() -> str:
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

col_main, col_summary = st.columns([2.5, 1.5])

with col_main:
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
        st.rerun()

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
                "Tell me more about your artist and what your goals are with this tour, and I can "
                "help you plan an exciting itinerary that hits the best events for your fanbase!"
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
                                    is_itinerary=True,
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
                    f"Please propose a genre-appropriate, geographically efficient tour itinerary "
                    f"that balances festivals and headline shows."
                )
            else:
                base_prompt = st.session_state.last_prompt or ""
                prompt = (
                    base_prompt
                    + "\n\nPlease regenerate an alternative tour itinerary with different routing "
                    "and event choices, while still respecting genre fit, fanbase, and travel efficiency."
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
                            is_itinerary=True,
                        )
                        st.markdown(reply)
                    except RateLimitError:
                        st.warning("Rate limit reached. Please wait 30 seconds and click Regenerate.")
                        st.stop()

            st.session_state.display.append({"role": "assistant", "text": reply})

            if st.session_state.exchanges == 1 or (
                st.session_state.exchanges > 0 and st.session_state.exchanges % SUMMARY_AFTER == 0
            ):
                st.session_state.pending_summary = True

# summary panel
with col_summary:
    st.subheader("Tour Summary:")
    if st.session_state.summary:
        with st.expander("Show / Hide Summary", expanded=False):
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