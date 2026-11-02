# NoteRecall — Command Cheat Sheet

Always run commands from the project folder:

```
cd ~/Desktop/rerag
```

Your gold file lives at `data/rerag-data/gold/gold.json` (inside the project, not the `rerag-data` folder on your Desktop).

---

## 1. Setup (one time, or after pulling new code)

| Command | What it does |
|---|---|
| `curl -LsSf https://astral.sh/uv/0.11.32/install.sh \| sh` | Installs `uv`, the tool that runs the project (only once per computer) |
| `git pull` | Gets the latest code from GitHub |
| `make setup` | Installs the project's Python packages |
| `make sync` | Reads your notebooks, cleans them and stores them in the database |
| `make check` | Runs all tests and code checks (should end with no errors) |

---

## 2. Writing queries

| Command | What it does |
|---|---|
| `uv run noterecall label` | Starts writing new queries. Each one is saved after `/done` |
| `uv run noterecall label --edit q007` | Re-opens query `q007` to change its marked cells |
| `uv run noterecall label --gold-path /tmp/dry.json` | Practice mode: saves to a temporary file, not your real gold file |

Press **Ctrl+C** to stop. Everything saved with `/done` is kept.

When you start a query, the tool asks:
1. **query text** — type your question the way you'd really ask it
2. **type** — pick a number:
   1. `concept` — explain an idea ("what does -1 mean in reshape")
   2. `code_identifier` — about a specific function ("how does astype work")
   3. `how_did_i` — how *you* did something in a notebook ("how did I pick k for KMeans")
   4. `cross_notebook` — answer needs several notebooks ("ways to handle missing values")
   5. `unanswerable` — not in your notes at all (mark nothing)

---

## 3. Inside the tool (at the `label >` prompt)

| Type this | What it does |
|---|---|
| `reshape dimension` (plain words) | Searches cells and shows numbered results |
| `/grep astype` | Finds cells containing that exact text |
| `/more` | Shows the next page of results |
| `/nb` | Lists all notebooks with numbers |
| `/nb 12` | Shows notebook 12 cell by cell |
| `/mark 2 5` | Marks results 2 and 5 as relevant |
| `/markall 3` | Marks result 3 and its identical copies in other notebooks |
| `/unmark 5` | Removes a mark |
| `/marked` | Shows what you've marked |
| `/done` | Saves this query |
| `/abort` | Throws this query away without saving |
| `/help` | Shows this list |

**Typical flow:**
```
label > reshape -1 dimension
label > /mark 2 4
label > /grep reshape(
label > /mark 1
label > /marked
label > /done
```

---

## 4. Checking your gold set

| Command | What it does |
|---|---|
| `uv run noterecall gold stats` | Counts your queries by type |
| `uv run noterecall gold validate` | Checks for mistakes (missing labels, duplicates, broken cells) |
| `uv run noterecall gold validate --ready` | Same, plus checks you have enough queries to freeze |
| `uv run noterecall gold freeze` | Splits queries into dev/test **once**. Do this only when finished |

After freezing you can only change marked cells (`--edit`), not query text or type.

---

## 5. Saving your work to GitHub

```
cd ~/Desktop/rerag/data/rerag-data
git add gold/gold.json
git commit -m "Gold set: work in progress"
git push
cd ~/Desktop/rerag
```

Do this after every labelling session so nothing is lost.

---

## 6. Rules of thumb

- **Target:** 60 minimum, 80 is better; at least 10% `unanswerable`.
- **Mark a cell only if you'd need it to answer the question.** Usually 1–5 cells.
- **Partial answers count:** mark the cells that cover the part your notes answer.
- **Unanswerable:** search 2–3 times with different words; mark nothing; `/done`.
- **Use different words from your notes** in some queries (e.g. "pick clusters" instead of "elbow method").
- **To fix text/type before freezing:** edit `gold.json` by hand, then run `gold validate`.

---

## 7. When finished

1. `uv run noterecall gold validate --ready`
2. `uv run noterecall gold freeze`
3. Push `gold.json` (section 5)
4. Tell the agent: *"Gold set frozen and pushed to rerag-data. Plan M4."*
