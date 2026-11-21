"""
Deep · portfolio RAG — ingestion.

Collects everything Deep should know, chunks it, embeds only what changed,
and syncs Supabase:

  * every public, non-fork repo of GITHUB_USER   → all *.md files + a repo card
  * the portfolio repo (PORTFOLIO_REPO)           → also index.html sections and every PDF
                                                    (resume, case study, articles)

Chunk ids are content hashes, so:
  new/edited text  → embedded + inserted
  unchanged text   → skipped (no API call)
  deleted text     → removed from Supabase

Usage
  python ingest/ingest.py                    # normal run (GitHub Action)
  python ingest/ingest.py --dry-run          # build chunks, print stats, touch nothing
  python ingest/ingest.py --dry-run --local-portfolio ./Portfolio-main --no-github
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

# ----------------------------------------------------------------- config
GITHUB_USER     = os.getenv("GITHUB_USER", "Deepaksangh41git")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN", "")
PORTFOLIO_REPO  = os.getenv("PORTFOLIO_REPO", "Portfolio")
SKIP_TOPIC      = os.getenv("SKIP_TOPIC", "no-rag")           # add this topic to a repo to exclude it
SKIP_REPOS      = {r.strip().lower() for r in os.getenv("SKIP_REPOS", "").split(",") if r.strip()}
INCLUDE_FORKS   = os.getenv("INCLUDE_FORKS", "false").lower() == "true"

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM       = int(os.getenv("EMBED_DIM", "768"))
EMBED_BATCH     = int(os.getenv("EMBED_BATCH", "20"))
EMBED_PAUSE_S   = float(os.getenv("EMBED_PAUSE_S", "21"))     # keeps us under free-tier TPM
MAX_EMBED_RUN   = int(os.getenv("MAX_EMBED_PER_RUN", "900"))   # free tier ≈ 1,000 requests/day

SUPABASE_URL    = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY    = os.getenv("SUPABASE_SECRET_KEY", "")          # secret / service_role key

TARGET_CHARS    = 1800    # ≈ 450 tokens
MAX_CHARS       = 2400
OVERLAP_CHARS   = 250
MIN_CHARS       = 40      # drop near-empty chunks

SKIP_MD = re.compile(r"(^|/)(node_modules|\.github|vendor|dist|build|site-packages|\.venv|venv)/"
                     r"|(^|/)(LICENSE|CODE_OF_CONDUCT|CONTRIBUTING|SECURITY|CHANGELOG)\.md$", re.I)


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------- models
@dataclass
class Doc:
    source_key: str
    source_type: str          # markdown | pdf | portfolio | repo
    repo: str | None
    title: str
    url: str
    text: str = ""
    sections: list[tuple[str, str, str]] = field(default_factory=list)  # (heading, text, url) pre-split


@dataclass
class Chunk:
    id: str
    source_key: str
    source_type: str
    repo: str | None
    title: str
    heading: str
    url: str
    content: str

    def row(self, emb: list[float]) -> dict:
        return {
            "id": self.id, "source_key": self.source_key, "source_type": self.source_type,
            "repo": self.repo, "title": self.title, "heading": self.heading or None,
            "url": self.url, "content": self.content,
            "embedding": "[" + ",".join(f"{x:.7f}" for x in emb) + "]",
        }


# ----------------------------------------------------------------- github
def gh_client() -> httpx.Client:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "deep-portfolio-rag"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return httpx.Client(headers=h, timeout=30, follow_redirects=True)


def list_repos(gh: httpx.Client) -> list[dict]:
    repos, page = [], 1
    while True:   # /users/{u}/repos returns PUBLIC repos only — private repos never leak in
        r = gh.get(f"https://api.github.com/users/{GITHUB_USER}/repos",
                   params={"per_page": 100, "page": page, "type": "owner", "sort": "updated"})
        r.raise_for_status()
        batch = r.json()
        repos += batch
        if len(batch) < 100:
            break
        page += 1
    keep = []
    for rp in repos:
        if rp.get("fork") and not INCLUDE_FORKS:
            continue
        if rp["name"].lower() in SKIP_REPOS or SKIP_TOPIC in (rp.get("topics") or []):
            continue
        keep.append(rp)
    return keep


def repo_tree(gh: httpx.Client, rp: dict) -> list[str]:
    r = gh.get(f"https://api.github.com/repos/{rp['full_name']}/git/trees/{rp['default_branch']}",
               params={"recursive": "1"})
    if r.status_code in (404, 409):           # 409 = empty repo
        return []
    r.raise_for_status()
    return [t["path"] for t in r.json().get("tree", []) if t["type"] == "blob"]


def raw_bytes(gh: httpx.Client, rp: dict, path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{rp['full_name']}/{rp['default_branch']}/{quote(path)}"
    r = gh.get(url)
    r.raise_for_status()
    return r.content


def blob_url(rp: dict, path: str) -> str:
    return f"{rp['html_url']}/blob/{rp['default_branch']}/{quote(path)}"


# ----------------------------------------------------------------- text helpers
def gh_slug(heading: str) -> str:
    s = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return s.replace(" ", "-")


def clean_md(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", md)    # linked badges
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)                 # images
    md = re.sub(r"\[\s*\]\([^)]*\)", "", md)                      # empty links left behind
    md = re.sub(r"<img[^>]*>", "", md, flags=re.I)
    md = re.sub(r"</?(p|div|br|span|a|picture|source|center|details|summary)[^>]*>", "", md, flags=re.I)
    md = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", md)      # keep link text + target
    md = re.sub(r"[ \t]+\n", "\n", md)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def md_sections(md: str) -> list[tuple[list[str], str]]:
    """Split markdown on headings (ignoring '#' inside code fences)."""
    out, path, buf, in_code = [], [], [], False
    for line in md.splitlines():
        if line.strip().startswith(("```", "~~~")):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            if "\n".join(buf).strip():
                out.append((list(path), "\n".join(buf).strip()))
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            while len(path) < level:            # skipped heading levels
                path.insert(len(path) - 1, "")
            path = [p for p in path if p] or [m.group(2).strip()]
            buf = []
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        out.append((list(path), "\n".join(buf).strip()))
    return out


def _hard_split(p: str) -> list[str]:
    """Split one oversized block on lines, then sentences, then (last resort) words."""
    units: list[str] = []
    for line in (l for l in p.split("\n") if l.strip()):
        if len(line) <= TARGET_CHARS:
            units.append(line)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            while len(sent) > TARGET_CHARS:               # giant sentence: cut on a space
                cut = sent.rfind(" ", 0, TARGET_CHARS)
                cut = cut if cut > TARGET_CHARS // 2 else TARGET_CHARS
                units.append(sent[:cut]); sent = sent[cut:].lstrip()
            if sent:
                units.append(sent)
    out, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > TARGET_CHARS:
            out.append(cur)
            tail = cur[-OVERLAP_CHARS:]
            tail = tail.split(" ", 1)[-1] if " " in tail else tail
            cur = tail + " " + u                          # overlap with previous piece
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur.strip():
        out.append(cur)
    return out


def split_long(text: str) -> list[str]:
    """Split text > MAX_CHARS into ~TARGET_CHARS pieces with overlap."""
    if len(text) <= MAX_CHARS:
        return [text]
    blocks: list[str] = []
    for p in re.split(r"\n\s*\n", text):
        blocks += _hard_split(p) if len(p) > TARGET_CHARS else [p]
    pieces, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) + 2 > TARGET_CHARS:
            pieces.append(cur.strip())
            cur = b
        else:
            cur = f"{cur}\n\n{b}" if cur else b
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def make_chunk(doc: Doc, heading: str, body: str, url: str) -> Chunk | None:
    body = body.strip()
    if len(body) < MIN_CHARS:
        return None
    header = f"Source: {doc.title}" + (f"\nSection: {heading}" if heading else "")
    content = f"{header}\n\n{body}"
    cid = hashlib.sha256(f"{doc.source_key}\n{content}".encode()).hexdigest()[:40]
    return Chunk(cid, doc.source_key, doc.source_type, doc.repo, doc.title, heading, url, content)


def chunk_doc(doc: Doc) -> list[Chunk]:
    chunks: list[Chunk] = []

    if doc.sections:                                       # already split (portfolio html)
        for heading, text, url in doc.sections:
            for piece in split_long(text):
                c = make_chunk(doc, heading, piece, url)
                if c: chunks.append(c)
        return chunks

    if doc.source_type == "markdown":
        secs = md_sections(clean_md(doc.text))
        # pack small neighbouring sections together up to TARGET_CHARS
        packed: list[tuple[str, str, str]] = []
        for path, body in secs:
            heading = " > ".join(path)
            anchor = f"{doc.url}#{gh_slug(path[-1])}" if path else doc.url
            text = (f"## {path[-1]}\n{body}" if path else body)
            prev_len = len(packed[-1][1]) if packed else 0
            # pack small neighbours together; also fold a tiny section into the previous chunk
            if packed and (prev_len + len(text) < TARGET_CHARS
                           or (len(text) < 300 and prev_len % TARGET_CHARS + len(text) < MAX_CHARS)):
                h, t, u = packed[-1]
                packed[-1] = (h or heading, t + "\n\n" + text, u)
            else:
                packed.append((heading, text, anchor))
        for heading, text, url in packed:
            for piece in split_long(text):
                c = make_chunk(doc, heading, piece, url)
                if c: chunks.append(c)
        return chunks

    for piece in split_long(doc.text):                     # pdf / plain
        c = make_chunk(doc, "", piece, doc.url)
        if c: chunks.append(c)
    return chunks


# ----------------------------------------------------------------- source parsers
def pdf_text(data: bytes) -> tuple[str, str | None]:
    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    meta = reader.metadata.title if reader.metadata else None
    return re.sub(r"\n{3,}", "\n\n", text).strip(), meta


def pdf_title(path: str, text: str, meta_title: str | None) -> str:
    if re.search(r"resume|\bcv\b", Path(path).stem, re.I):
        return "Resume"
    if meta_title and len(meta_title.strip()) > 4:
        return meta_title.strip()
    stem = Path(path).stem.replace("_", " ").replace("-", " ").strip()
    if not re.fullmatch(r"\d+", stem):
        return stem
    # "1.pdf": the title is usually the first run of short, non-ALL-CAPS lines on page 1
    words: list[str] = []
    for line in (l.strip() for l in text.splitlines()[:25]):
        if not line or line.endswith(":") or line.isupper():
            if words: break
            continue
        if len(line) > 40:
            break
        words.append(line)
    return " ".join(words)[:100] or f"{Path(path).parent.name} {stem}".strip()


def portfolio_sections(html: str) -> list[tuple[str, str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "svg", "canvas", "noscript", "button", "video", "iframe"]):
        t.decompose()

    def text_of(node) -> str:
        for br in node.find_all(["br"]):
            br.replace_with("\n")
        txt = node.get_text("\n", strip=True)
        lines = [l for l in (x.strip() for x in txt.splitlines()) if l]
        # join tiny fragments (chips, tags) onto one line
        out, buf = [], []
        for l in lines:
            if len(l) < 40 and not l.endswith((".", ":", "?")):
                buf.append(l)
            else:
                if buf: out.append(" · ".join(buf)); buf = []
                out.append(l)
        if buf: out.append(" · ".join(buf))
        return "\n".join(out)

    secs: list[tuple[str, str, str]] = []
    for tpl in soup.find_all("template"):                  # project detail modals
        frag = BeautifulSoup(tpl.decode_contents(), "html.parser")   # template text is skipped by get_text()
        title = frag.find(["h2", "h3", "h4"])
        name = title.get_text(" ", strip=True) if title else tpl.get("id", "Project")
        secs.append((f"Project: {name}", text_of(frag), "#projects"))
        tpl.decompose()
    for sec in soup.find_all("section", id=True):
        sid = sec["id"]
        secs.append((sid.replace("-", " ").title(), text_of(sec), f"#{sid}"))
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        secs.insert(0, ("Overview", meta["content"], "#hero"))
    return [s for s in secs if s[1].strip()]


def repo_card(rp: dict, md_count: int) -> Doc:
    lines = [
        f"GitHub repository: {rp['name']}",
        f"Description: {rp.get('description') or 'No description'}",
        f"Main language: {rp.get('language') or 'n/a'}",
        f"Topics: {', '.join(rp.get('topics') or []) or 'none'}",
        f"Stars: {rp.get('stargazers_count', 0)}",
        f"Created: {rp.get('created_at', '')[:10]} · Last updated: {rp.get('pushed_at', '')[:10]}",
        f"Homepage: {rp.get('homepage') or 'none'}",
        f"URL: {rp['html_url']}",
        f"Markdown docs indexed: {md_count}",
    ]
    return Doc(f"gh:{rp['full_name']}:__card__", "repo", rp["name"], f"{rp['name']} (GitHub repo)",
               rp["html_url"], text="\n".join(lines))


def repos_index(repos: list[dict]) -> Doc:
    lines = [f"Deepak's public GitHub repositories ({len(repos)} total, newest first):"]
    for rp in repos:
        lines.append(f"- {rp['name']}: {rp.get('description') or 'no description'} "
                     f"[{rp.get('language') or 'n/a'}] {rp['html_url']}")
    return Doc("gh:__index__", "repo", None, "GitHub repositories overview",
               f"https://github.com/{GITHUB_USER}?tab=repositories", text="\n".join(lines))


# ----------------------------------------------------------------- collection
def collect_github(gh: httpx.Client, skip_portfolio: bool) -> list[Doc]:
    docs: list[Doc] = []
    repos = list_repos(gh)
    log(f"• {len(repos)} public repos for {GITHUB_USER}")
    for rp in repos:
        is_portfolio = rp["name"].lower() == PORTFOLIO_REPO.lower()
        if is_portfolio and skip_portfolio:
            continue
        try:
            paths = repo_tree(gh, rp)
        except httpx.HTTPError as e:
            log(f"  ! {rp['name']}: tree failed ({e}); skipping")
            continue
        mds = [p for p in paths if p.lower().endswith((".md", ".mdx")) and not SKIP_MD.search(p)]
        for p in mds:
            try:
                txt = raw_bytes(gh, rp, p).decode("utf-8", "ignore")
            except httpx.HTTPError as e:
                log(f"  ! {rp['name']}/{p}: {e}")
                continue
            title = rp["name"] if p.lower() in ("readme.md", "readme.mdx") else f"{rp['name']} / {p}"
            docs.append(Doc(f"gh:{rp['full_name']}:{p}", "markdown", rp["name"], title,
                            blob_url(rp, p), text=txt))
        docs.append(repo_card(rp, len(mds)))
        if is_portfolio:
            docs += portfolio_docs_remote(gh, rp, paths)
        log(f"  - {rp['name']}: {len(mds)} md")
    docs.append(repos_index(repos))
    return docs


def portfolio_docs_from(files: dict[str, bytes], key_prefix: str) -> list[Doc]:
    """files: {relative_path: bytes} for index.html and PDFs. URLs are site-relative so
    the widget opens them on your own site (e.g. '#projects', 'Documents/x.pdf')."""
    docs: list[Doc] = []
    for path, data in files.items():
        if path.lower() == "index.html":
            docs.append(Doc(f"{key_prefix}:index.html", "portfolio", PORTFOLIO_REPO, "Portfolio website",
                            "#hero", sections=portfolio_sections(data.decode("utf-8", "ignore"))))
        elif path.lower().endswith(".pdf"):
            try:
                txt, meta_title = pdf_text(data)
            except Exception as e:                         # broken / scanned pdf
                log(f"  ! pdf {path}: {e}")
                continue
            if len(txt) < MIN_CHARS:
                log(f"  ! pdf {path}: no extractable text (scanned?) — skipped")
                continue
            docs.append(Doc(f"{key_prefix}:{path}", "pdf", PORTFOLIO_REPO, pdf_title(path, txt, meta_title),
                            quote(path), text=txt))
    return docs


def portfolio_docs_remote(gh: httpx.Client, rp: dict, paths: list[str]) -> list[Doc]:
    want = [p for p in paths if p.lower() == "index.html" or p.lower().endswith(".pdf")]
    files = {}
    for p in want:
        try:
            files[p] = raw_bytes(gh, rp, p)
        except httpx.HTTPError as e:
            log(f"  ! {p}: {e}")
    return portfolio_docs_from(files, f"gh:{rp['full_name']}")


def portfolio_docs_local(root: Path) -> list[Doc]:
    files = {}
    for f in root.rglob("*"):
        rel = f.relative_to(root).as_posix()
        if f.is_file() and (rel.lower() == "index.html" or rel.lower().endswith(".pdf")):
            files[rel] = f.read_bytes()
    docs = portfolio_docs_from(files, f"gh:{GITHUB_USER}/{PORTFOLIO_REPO}")
    for f in root.rglob("*.md"):
        rel = f.relative_to(root).as_posix()
        docs.append(Doc(f"gh:{GITHUB_USER}/{PORTFOLIO_REPO}:{rel}", "markdown", PORTFOLIO_REPO,
                        PORTFOLIO_REPO if rel.lower() == "readme.md" else f"{PORTFOLIO_REPO} / {rel}",
                        f"https://github.com/{GITHUB_USER}/{PORTFOLIO_REPO}/blob/main/{quote(rel)}",
                        text=f.read_text("utf-8", "ignore")))
    return docs


# ----------------------------------------------------------------- embeddings
class DailyQuotaExhausted(Exception):
    pass


def embed_batch(cl: httpx.Client, chunks: list[Chunk]) -> list[list[float]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents"
    body = {"requests": [{
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": c.content}]},
        "taskType": "RETRIEVAL_DOCUMENT",
        "title": c.title[:200],
        "outputDimensionality": EMBED_DIM,
    } for c in chunks]}
    for attempt in range(5):
        r = cl.post(url, json=body, headers={"x-goog-api-key": GEMINI_API_KEY})
        if r.status_code == 429:
            msg = r.text
            if "PerDay" in msg or "per day" in msg.lower():
                raise DailyQuotaExhausted(msg[:300])
            wait = 30 * (attempt + 1)
            log(f"    429 rate-limited, waiting {wait}s")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        vecs = [e["values"] for e in r.json()["embeddings"]]
        if any(len(v) != EMBED_DIM for v in vecs):
            raise RuntimeError(f"embedding dim mismatch (expected {EMBED_DIM})")
        return vecs
    raise RuntimeError("embedding failed after retries")


# ----------------------------------------------------------------- supabase
class Supa:
    def __init__(self):
        h = {"apikey": SUPABASE_KEY, "Content-Type": "application/json"}
        if SUPABASE_KEY.startswith("eyJ"):                 # legacy JWT service_role key
            h["Authorization"] = f"Bearer {SUPABASE_KEY}"
        self.c = httpx.Client(base_url=f"{SUPABASE_URL}/rest/v1", headers=h, timeout=60)

    def existing_ids(self) -> set[str]:
        ids, off = set(), 0
        while True:
            r = self.c.get("/chunks", params={"select": "id", "order": "id", "limit": 1000, "offset": off})
            r.raise_for_status()
            rows = r.json()
            ids |= {x["id"] for x in rows}
            if len(rows) < 1000:
                return ids
            off += 1000

    def upsert(self, rows: list[dict]):
        r = self.c.post("/chunks", content=json.dumps(rows),
                        headers={"Prefer": "resolution=merge-duplicates,return=minimal"})
        if r.status_code >= 300:
            raise RuntimeError(f"upsert failed {r.status_code}: {r.text[:400]}")

    def delete(self, ids: list[str]):
        for i in range(0, len(ids), 100):
            part = ids[i:i + 100]
            r = self.c.delete("/chunks", params={"id": f"in.({','.join(part)})"})
            if r.status_code >= 300:
                raise RuntimeError(f"delete failed {r.status_code}: {r.text[:400]}")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build chunks only; no API/DB calls")
    ap.add_argument("--local-portfolio", type=Path, help="read portfolio files from a local folder")
    ap.add_argument("--no-github", action="store_true", help="skip GitHub (testing)")
    ap.add_argument("--dump", type=Path, help="write chunks to a JSON file for inspection")
    args = ap.parse_args()

    docs: list[Doc] = []
    if not args.no_github:
        with gh_client() as gh:
            docs += collect_github(gh, skip_portfolio=bool(args.local_portfolio))
    if args.local_portfolio:
        docs += portfolio_docs_local(args.local_portfolio)

    chunks: dict[str, Chunk] = {}
    for d in docs:
        for c in chunk_doc(d):
            chunks[c.id] = c                               # dedupe identical chunks
    log(f"• {len(docs)} documents → {len(chunks)} chunks")

    if args.dump:
        args.dump.write_text(json.dumps([c.__dict__ for c in chunks.values()], indent=1, ensure_ascii=False))
        log(f"• wrote {args.dump}")
    if args.dry_run:
        sizes = sorted(len(c.content) for c in chunks.values())
        if sizes:
            log(f"• chunk chars: min {sizes[0]}, median {sizes[len(sizes)//2]}, max {sizes[-1]}")
        return

    missing = [k for k, v in {"GEMINI_API_KEY": GEMINI_API_KEY, "SUPABASE_URL": SUPABASE_URL,
                              "SUPABASE_SECRET_KEY": SUPABASE_KEY}.items() if not v]
    if missing:
        sys.exit(f"Missing env: {', '.join(missing)}")
    if not chunks:
        sys.exit("No chunks built — refusing to sync (would wipe the index).")

    db = Supa()
    have = db.existing_ids()
    new = [c for cid, c in chunks.items() if cid not in have]
    stale = sorted(have - set(chunks))
    log(f"• in DB: {len(have)} · new/changed: {len(new)} · stale: {len(stale)}")

    todo = new[:MAX_EMBED_RUN]
    if len(new) > len(todo):
        log(f"• capping this run at {len(todo)} embeddings; the rest go in the next run")

    done = 0
    with httpx.Client(timeout=60) as cl:
        for i in range(0, len(todo), EMBED_BATCH):
            batch = todo[i:i + EMBED_BATCH]
            try:
                vecs = embed_batch(cl, batch)
            except DailyQuotaExhausted as e:
                log(f"• daily embedding quota reached — stopping, will resume next run ({e})")
                break
            db.upsert([c.row(v) for c, v in zip(batch, vecs)])
            done += len(batch)
            log(f"  embedded {done}/{len(todo)}")
            if i + EMBED_BATCH < len(todo):
                time.sleep(EMBED_PAUSE_S)

    # Only delete stale rows once everything current is in, so the bot never
    # has a gap where an edited file has neither its old nor new chunks.
    if done == len(new) and stale:
        db.delete(stale)
        log(f"• deleted {len(stale)} stale chunks")
    log("✓ done")


if __name__ == "__main__":
    main()
