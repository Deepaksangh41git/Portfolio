-- =====================================================================
-- Deep · portfolio RAG — Supabase schema
-- Run once in Supabase Dashboard → SQL Editor → New query → Run.
-- Safe to re-run.
-- =====================================================================

create extension if not exists vector with schema extensions;

-- One row per chunk. `id` is a content hash, so an unchanged chunk keeps
-- its id and is never re-embedded; a changed chunk gets a new id.
create table if not exists public.chunks (
  id          text primary key,              -- sha256(source_key + content)
  source_key  text not null,                 -- e.g. gh:Owner/Repo:README.md
  source_type text not null,                 -- markdown | pdf | portfolio | repo
  repo        text,
  title       text not null,
  heading     text,
  url         text not null,
  content     text not null,
  embedding   extensions.vector(768) not null,
  fts         tsvector generated always as (
                to_tsvector('english',
                  coalesce(title, '') || ' ' || coalesce(heading, '') || ' ' || content)
              ) stored,
  updated_at  timestamptz not null default now()
);

create index if not exists chunks_embedding_idx
  on public.chunks using hnsw (embedding extensions.vector_cosine_ops);
create index if not exists chunks_fts_idx  on public.chunks using gin (fts);
create index if not exists chunks_source_idx on public.chunks (source_key);

-- Lock the table down: RLS on with no policies = nobody but service_role
-- (used only by the GitHub Action) can read or write rows directly.
alter table public.chunks enable row level security;
revoke all on table public.chunks from anon, authenticated;
grant select, insert, update, delete on table public.chunks to service_role;

-- ---------------------------------------------------------------------
-- Hybrid search: keyword (full-text) + semantic (vector), merged with
-- Reciprocal Rank Fusion. Pass query_embedding = null to fall back to
-- keyword-only search (used when the embedding API is rate-limited).
-- SECURITY DEFINER lets the public (publishable/anon) key call this
-- function without being able to read the table itself.
-- ---------------------------------------------------------------------
create or replace function public.match_chunks(
  query_text       text,
  query_embedding  extensions.vector(768) default null,
  match_count      int   default 6,
  full_text_weight float default 1,
  semantic_weight  float default 1,
  rrf_k            int   default 50
)
returns table (id text, title text, heading text, url text, content text, score float)
language sql
stable
security definer
set search_path = public, extensions
as $$
  with full_text as (
    select c.id,
           row_number() over (
             order by ts_rank_cd(c.fts, websearch_to_tsquery('english', query_text)) desc
           ) as rank_ix
    from public.chunks c
    where c.fts @@ websearch_to_tsquery('english', query_text)
    order by rank_ix
    limit least(match_count, 20) * 2
  ),
  semantic as (
    select c.id,
           row_number() over (order by c.embedding <=> query_embedding) as rank_ix
    from public.chunks c
    where query_embedding is not null
    order by rank_ix
    limit least(match_count, 20) * 2
  )
  select c.id, c.title, c.heading, c.url, c.content,
         (coalesce(1.0 / (rrf_k + ft.rank_ix), 0.0) * full_text_weight +
          coalesce(1.0 / (rrf_k + s.rank_ix),  0.0) * semantic_weight)::float as score
  from full_text ft
  full outer join semantic s on ft.id = s.id
  join public.chunks c on c.id = coalesce(ft.id, s.id)
  order by score desc
  limit least(match_count, 20);
$$;

revoke execute on function public.match_chunks(text, extensions.vector, int, float, float, int) from public;
grant  execute on function public.match_chunks(text, extensions.vector, int, float, float, int)
  to anon, authenticated, service_role;
