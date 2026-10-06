# 9. Past chats in the sidebar, memory as markdown files, one time zone

**Status:** Accepted (2026-10-06)

## Context

After long-term memory (ADR 0008), three things were missing:

- **No way back to an old chat.** Chats had no titles, and the UI only knew the current session.
- **Memory you could see but not shape.** It was shown item by item, so you could delete a fact but not correct it, add one, or say anything about yourself.
- **Dates that were wrong in two places:**
  - Past-chat summaries were dated and ordered by **when they were summarized**. The background summarizer (ADR 0008) often summarizes a chat days later.
  - The prompt used the **UTC** date, which is the previous day in India between 00:00 and 05:30.
- **The agent wasn't told today's date.**

ShopSense runs for one person, so there's a single user and no login.

## Decision

**Chats:**
- `sessions` gets `title`, `title_source` (`placeholder` | `llm` | `user`) and `deleted_at`.
- **Titles:** a chat is titled by its first message, then named once by the small model after its first answer. It runs in the background like memory extraction, and only replaces a placeholder, so a rename always wins.
- **The list** (`GET /sessions`) shows chats with a message that aren't deleted, most recently active first, with title search. `PATCH` renames.
- **`DELETE` is a soft delete:** the rows stay, so memories learned from the chat keep their source. Deleted chats aren't summarized.
- **Titles aren't activity:** setting one doesn't change `updated_at`, which orders the list and decides when a chat needs a new summary.
- **Existing databases** get the new columns at startup, and each existing chat gets its first message as a title.

**`user.md`:**
- **One row** holding a short template ("# About me / Name / Call me / Pronouns / City / Notes").
- **Written only by the user.** It's shown to the agent, wrapped as `<user.md>`, once something is filled in.
- **Known to the extractor,** so its facts aren't copied into memories.

**Memory files** (`app/agent/memory_files.py`, `GET/PUT/DELETE /memory/files/{name}`):
- **`memory.md`, `preferences.md` and the day files are views of the database rows,** generated on every read, so they always match what the agent uses.
- **Saving applies the edit:**
  - `memory.md` (headings are categories): a removed line is forgotten, a new line is stored as stated by the user, and toggling "(inferred)" changes the confidence. An unchanged fact keeps its id and timestamps.
  - `preferences.md`: `- key: value` lines.
  - `YYYY-MM-DD.md`: one section per chat, tied to it by a `<!-- chat 1a2b3c4d -->` line. Editing a summary rewrites it; deleting a section forgets that chat's summary for good.
- **Clearing:** `user.md` goes back to the template; the other files are forgotten.

**Time:**
- **Storage stays UTC.** A `TIMEZONE` setting (IANA, default `Asia/Kolkata`) is used for every date anyone sees: the agent's "Today" line, the day a chat belongs to, the day files, and the UI's labels. `/health` reports it, so the UI never guesses.
- **Past chats are dated and ordered by the chat's last activity,** not by when they were summarized.
- **`tzdata` is declared,** because Windows has no time zone database.

**UI:**
- **The sidebar:** New chat, search, the chat list (open, rename, delete), the cart, then Settings.
- **The Settings dialog:** a "You" tab (`user.md`) and a "Memory" tab (Long term, Preferences, Short term with a day picker). Each editor shows "Updated OCT 6, 2026 | 10:15 AM" and has Save, plus Clear behind a confirmation.

## Consequences

**Measured end to end** (uvicorn on a seeded temporary database, the real Streamlit app driven with `AppTest`):
- The list ordered correctly: "2 hr ago", "Yesterday", "3 days ago", "24 Sep".
- Opening, renaming, deleting with the confirmation, and search all worked.
- Edits to `memory.md`, `user.md` and `preferences.md` were applied.
- A real `gemma4:31b` turn got the title "Coding keyboard under 6000".
- Chats from 5 Oct, 3 Oct and 24 Sep that were summarized on 6 Oct appeared under their own days.

**Tested on day boundaries:**
- 18:45 UTC on 5 Oct is 00:15 on 6 Oct in India.
- 23:30 yesterday is "Yesterday", though only 11 hours ago.
- A summary made days later stays on the chat's day.
- An old database migrates.

**The cost:** one more small-model call per chat, for its title. The sidebar re-fetches up to 30 chats on every rerun; "Show more" adds 30.

**Limits:**
- One user, no login: every chat in the database is listed.
- A Streamlit rerun is needed before a generated title shows, which is normally the next interaction.
- Deleting a chat keeps what was learned from it (shown in Settings).
- Editing a day file needs the `<!-- chat … -->` lines kept: a section without one is treated as deleted.
