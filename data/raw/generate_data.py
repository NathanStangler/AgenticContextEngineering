import json
import random
import os
import itertools
import re
from pathlib import Path

random.seed(42)

NUM_SAMPLES = 8000
OUT_PATH = Path(__file__).resolve().parents[1] / "processed" / "querycontext_dataset.json"

# ── Parameter pools ────────────────────────────────────────────────────────────
LANGS = ["Python", "TypeScript", "Go", "Rust", "JavaScript", "Java", "Ruby", "C++"]
WEB_FW = ["FastAPI", "Django", "Flask", "Express", "Next.js", "NestJS", "Gin", "Spring Boot", "Rails", "Fastify"]
TEST_FW = ["pytest", "Jest", "Vitest", "Go test", "JUnit", "RSpec", "Mocha"]
ORM = ["SQLAlchemy", "Prisma", "GORM", "TypeORM", "ActiveRecord", "Hibernate"]
DB = ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "DynamoDB", "CockroachDB"]
CLOUD = ["AWS", "GCP", "Azure", "Vercel", "Railway", "Fly.io"]
TOOLS = ["Docker", "Kubernetes", "Terraform", "GitHub Actions", "CircleCI"]
GIT_OPS = ["rebase", "cherry-pick", "bisect", "stash", "reflog", "worktree"]

HTTP = {
    "400": ("Bad Request", "the request body is malformed or missing required fields", "400 is always the caller's fault"),
    "401": ("Unauthorized", "auth credentials are missing or invalid", "401 means retry with a valid token"),
    "403": ("Forbidden", "the client is authenticated but lacks permission", "403 won't go away with a new token — you need different permissions"),
    "404": ("Not Found", "the resource doesn't exist at that path", "404 could mean a wrong ID or the route isn't registered"),
    "409": ("Conflict", "the operation conflicts with existing state, like a duplicate key", "409 usually means you need to check before writing"),
    "422": ("Unprocessable Entity", "the request is syntactically valid but semantically wrong", "422 means your validation schema is rejecting the payload"),
    "429": ("Too Many Requests", "the rate limit has been exceeded", "429 means back off and retry after the Retry-After header"),
    "500": ("Internal Server Error", "an unhandled exception hit the server", "500 is always server-side — check your logs"),
    "502": ("Bad Gateway", "an upstream service returned garbage", "502 usually points to a proxy or load-balancer misconfiguration"),
    "503": ("Service Unavailable", "the server is overloaded or in maintenance", "503 is transient — add retry logic with exponential backoff"),
}

PY_ERRORS = ["AttributeError", "TypeError", "KeyError", "ValueError", "ImportError", "RuntimeError", "IndexError", "RecursionError"]
JS_ERRORS = ["TypeError", "ReferenceError", "Cannot read properties of undefined", "Promise rejection unhandled", "SyntaxError", "RangeError"]
GO_ERRORS = ["nil pointer dereference", "index out of range", "deadlock detected", "interface conversion panic", "context deadline exceeded"]
GENERIC_ERRORS = PY_ERRORS + JS_ERRORS + GO_ERRORS

CONCEPTS = [
    "dependency injection", "event sourcing", "CQRS", "circuit breaker", "rate limiting",
    "connection pooling", "optimistic locking", "eventual consistency", "idempotency",
    "two-phase commit", "saga pattern", "outbox pattern", "cache invalidation",
]
PERF_ISSUES = [
    "slow query", "N+1 query", "missing index", "memory leak", "goroutine leak",
    "event loop blocking", "cold start latency", "unbounded cache growth",
    "lock contention", "serialization bottleneck",
]
REFACTOR_TARGETS = [
    "this 200-line function", "the auth middleware", "the retry logic", "the config loader",
    "the database layer", "the error handling", "the pagination code", "the webhook handler",
]
FILE_TYPES = [".env", "Dockerfile", "docker-compose.yml", "pyproject.toml", "package.json",
              "tsconfig.json", "go.mod", ".github/workflows/ci.yml", "Makefile"]
ENDPOINTS = ["/users", "/auth/login", "/orders/{id}", "/health", "/metrics", "/webhooks", "/search", "/upload"]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
SERVICES = [
    "billing API", "auth gateway", "search service", "webhook consumer",
    "notifications worker", "analytics pipeline", "checkout service",
    "admin backend", "reporting job", "mobile API",
]
ENVIRONMENTS = [
    "production", "staging", "a canary deployment", "the EU region",
    "the background worker tier", "the primary cluster", "a preview environment",
]
CHANGE_EVENTS = [
    "today's deploy", "last night's migration", "a config rollout",
    "the latest hotfix", "a traffic spike", "a dependency bump",
    "a failover test",
]
GOALS = [
    "on-call simplicity", "the lowest-risk fix", "long-term maintainability",
    "fewer pager alerts", "faster incident recovery", "a boring production default",
]
STAKEHOLDERS = [
    "the platform team", "the on-call engineer", "the backend squad",
    "the incident commander", "the SRE rotation", "the teammate owning this service",
]

def p(pool): return random.choice(pool)
def http(): return random.choice(list(HTTP.keys()))


# ── Template definitions ───────────────────────────────────────────────────────
# Each template is a list of (user, assistant) turn pairs.
# Strings may reference local variables via .format(**ctx).

def make_ctx():
    code1, code2 = random.sample(list(HTTP.keys()), 2)
    lang = p(LANGS)
    fw = p(WEB_FW)
    db = p(DB)
    tool = p(TOOLS)
    git = p(GIT_OPS)
    orm = p(ORM)
    err = p(GENERIC_ERRORS)
    ep = p(ENDPOINTS)
    concept = p(CONCEPTS)
    perf = p(PERF_ISSUES)
    refactor = p(REFACTOR_TARGETS)
    test_fw = p(TEST_FW)
    cloud = p(CLOUD)
    log = p(LOG_LEVELS)
    ft = p(FILE_TYPES)
    service = p(SERVICES)
    environment = p(ENVIRONMENTS)
    change_event = p(CHANGE_EVENTS)
    goal = p(GOALS)
    stakeholder = p(STAKEHOLDERS)
    return dict(
        lang=lang, fw=fw, db=db, tool=tool, git=git, orm=orm, err=err,
        ep=ep, concept=concept, perf=perf, refactor=refactor, test_fw=test_fw,
        cloud=cloud, log=log, ft=ft, service=service, environment=environment,
        change_event=change_event, goal=goal, stakeholder=stakeholder,
        code1=code1, name1=HTTP[code1][0], desc1=HTTP[code1][1], tip1=HTTP[code1][2],
        code2=code2, name2=HTTP[code2][0], desc2=HTTP[code2][1], tip2=HTTP[code2][2],
    )

TEMPLATES = [
    # 0 — HTTP error basics
    lambda c: [
        ("I'm debugging a backend API. What does HTTP {code1} mean again?".format(**c),
         "It means {name1} — {desc1}.".format(**c)),
        ("What's the difference between {code1} and {code2}?".format(**c),
         "{code1} is client-side: {desc1}. {code2} is different: {desc2}. Rule of thumb — {tip2}.".format(**c)),
        ("In a real production system, which one should I log more aggressively?".format(**c),
         "{code2} every time. {code1} errors are usually caller mistakes; {code2} is your system failing. Alert on sustained {code2} spikes.".format(**c)),
    ],

    # 1 — Framework 404
    lambda c: [
        ("My {fw} app returns {code1} on {ep} but the route is definitely registered.".format(**c),
         "Double-check route ordering — in {fw}, a more specific route above it might be swallowing the request. Also verify any prefix middleware isn't stripping part of the path.".format(**c)),
        ("I checked that. Could it be a trailing slash issue?",
         "Yes, very common. {fw} treats `/foo` and `/foo/` differently by default. Either normalize at the reverse-proxy level or configure the framework to redirect trailing slashes.".format(**c)),
        ("What's the cleanest place to add that normalization?".format(**c),
         "At the ingress/nginx layer so every service benefits automatically. Add `rewrite ^(.*)/$ $1 permanent;` — one place, zero per-service code."),
    ],

    # 2 — Language error debugging
    lambda c: [
        ("I'm seeing `{err}` in my {lang} service and I can't figure out why.".format(**c),
         "In {lang}, `{err}` usually means {tip}. Can you share the stack trace or the line that throws?".format(
             tip="a value you assumed was set is actually None/nil/undefined — trace back to where it's populated" if "nil" in c['err'] or "undefined" in c['err'] or "Attribute" in c['err'] else "the type contract somewhere in the call chain is violated", **c)),
        ("Here's the trace: line 42 in `process_event`, line 18 in `handle_request`. The value comes from a DB query.",
         "Classic async race — the query is returning before the row is committed, or you're passing the result without awaiting it. Add a null-check at the boundary and log the raw DB response to confirm."),
        ("Should I add a fallback or fix the root cause?",
         "Fix the root cause. Fallbacks here would hide the real bug. Once it's fixed, add an assertion test so it can't regress silently."),
    ],

    # 3 — Git operation
    lambda c: [
        ("What's the difference between `git {git}` and `git merge`?".format(**c),
         "`git {git}` rewrites history — cleaner linear log but dangerous on shared branches. `git merge` preserves the full branch topology. Use `{git}` locally before pushing; use merge for shared branches.".format(**c)),
        ("If I already pushed the branch, is `git {git}` still safe?".format(**c),
         "Only if you're the sole author and you force-push. Anyone else who pulled will have divergent history. Safer: merge, or ask teammates to re-clone the branch."),
        ("When would you definitely avoid it?",
         "On `main`, `release/*`, or any branch another CI job is reading. Rewriting those breaks pipelines and blame history."),
    ],

    # 4 — Database query perf
    lambda c: [
        ("My {db} query is timing out in production but fast locally. Likely cause?".format(**c),
         "Most common: {perf}. Production has orders-of-magnitude more rows so missing indexes or unoptimized joins become critical. Run `EXPLAIN ANALYZE` on prod (read-only replica) to see the actual query plan.".format(**c)),
        ("EXPLAIN shows a sequential scan on the `events` table — 2M rows.",
         "Add a composite index on the columns in your WHERE and ORDER BY clauses. For a `(user_id, created_at)` filter + sort, that's `CREATE INDEX CONCURRENTLY idx_events_user_created ON events(user_id, created_at DESC);`."),
        ("CONCURRENTLY — does that lock the table?",
         "No, that's the point. It builds the index without holding a write lock, so production traffic continues. Takes longer but is safe for live tables."),
    ],

    # 5 — Refactoring
    lambda c: [
        ("I want to refactor {refactor}. Where do I even start?".format(**c),
         "Start by writing characterization tests — tests that lock in the current behavior without understanding it. Then you can refactor safely because the tests will catch regressions."),
        ("What if there are no existing tests at all?",
         "Add integration tests at the boundary (HTTP, queue, etc.) first. They're coarser but protect you. Unit tests inside the refactor come after you understand the shape."),
        ("Any heuristics for breaking up a big function?".format(**c),
         "Look for comment blocks — they're usually implicit function boundaries. Each block that could have a name is a candidate for extraction. Aim for functions that fit on one screen."),
    ],

    # 6 — Docker issue
    lambda c: [
        ("My Docker container starts fine locally but crashes immediately in {cloud}.".format(**c),
         "Likely causes: env vars missing in {cloud} that exist in your local `.env`; a platform architecture mismatch (M1/M2 image vs amd64); or a missing secret/volume mount. Check the container logs in {cloud} first.".format(**c)),
        ("Logs say `exec format error`.",
         "Architecture mismatch. Your image was built for arm64 (Apple Silicon) but {cloud} is running amd64. Rebuild with `docker buildx build --platform linux/amd64 -t your-image .`".format(**c)),
        ("Do I need to change the Dockerfile?",
         "Usually not. The `--platform` flag at build time is enough. If you push to a registry, use a multi-platform manifest so both architectures are covered: `--platform linux/amd64,linux/arm64`."),
    ],

    # 7 — Testing philosophy
    lambda c: [
        ("Should I mock the database in my {test_fw} tests?".format(**c),
         "For unit tests, yes — mock the repo layer so tests are fast and isolated. For integration tests, use a real {db} (e.g. via Docker or testcontainers) so you catch query and schema issues mocks hide.".format(**c)),
        ("We got burned when mocked tests passed but the production migration failed.",
         "Exactly the failure mode mocks cause. Add a separate integration test suite that runs against a real DB in CI. Keep unit tests for business logic, integration tests for anything touching persistence."),
        ("How do you keep integration tests fast enough for CI?".format(**c),
         "Spin up {db} once per suite (not per test), use transactions that roll back after each test, and parallelize suites. With that setup 200 integration tests typically finish under 30 seconds.".format(**c)),
    ],

    # 8 — Logging/observability
    lambda c: [
        ("What's the right log level for a failed external API call — {log} or something else?".format(**c),
         "Depends on whether it's expected. Transient failures (network blip, rate limit) → WARNING. Unexpected or repeated failures → ERROR. Never use {log} for errors you actually need to act on.".format(**c)),
        ("When should I use structured logging vs plain text?",
         "Always structured (JSON) in production — it's machine-parseable by Datadog, CloudWatch, Loki, etc. Plain text is fine for local dev. Never mix formats in the same service."),
        ("What fields should every log line include?".format(**c),
         "Minimum: `timestamp`, `level`, `service`, `trace_id`, `message`. Add `user_id` or `request_id` where available. Avoid logging PII unless it's masked."),
    ],

    # 9 — Performance
    lambda c: [
        ("My {lang} service latency spikes every few minutes. Where do I look first?".format(**c),
         "Periodic spikes usually point to GC pressure, scheduled jobs, or {db} connection pool exhaustion. Add histograms for GC pause time and pool wait time — whichever correlates with the spike is your culprit.".format(**c)),
        ("GC pause metrics show 200ms pauses. That matches the spike.",
         "You're allocating too much per request. Profile with pprof/{profiler} to find the hot allocation path, then reduce by reusing buffers (sync.Pool in Go, object pools in Java, etc.).".format(
             profiler="py-spy" if "Python" in "" else "pprof")),
        ("Is there a quick win before I do the full optimization?",
         "Yes — tune GC aggressiveness first. In Go, `GOGC=200` halves GC frequency at the cost of more memory. In the JVM, switch to G1GC with `-XX:MaxGCPauseMillis=50`. Measure before and after."),
    ],

    # 10 — CI/CD
    lambda c: [
        ("My {tool} pipeline fails on the deploy step but succeeds locally.".format(**c),
         "Classic causes: credentials not injected (secret missing in pipeline env), image pull policy difference, or a `latest` tag resolving to a different image. Add `--verbose` to the deploy command and check the exact error."),
        ("It says `unauthorized: authentication required`.",
         "The pipeline runner doesn't have the registry credentials. Add a registry login step before the deploy: `docker login -u $REGISTRY_USER -p $REGISTRY_TOKEN $REGISTRY_HOST`. Store the token as a pipeline secret, never hardcoded."),
        ("Should I use a deploy key or a service account token?".format(**c),
         "Service account with minimal permissions — read/pull on the registry, deploy on the target env only. Deploy keys are tied to a human account and rotate with them. Service accounts are auditable and purpose-scoped."),
    ],

    # 11 — Config/env
    lambda c: [
        ("What's the best way to manage secrets in a {lang} service?".format(**c),
         "12-factor: env vars injected at runtime, never in source. In {cloud} use the native secrets manager (AWS Secrets Manager, GCP Secret Manager). Locally, `.env` file that's gitignored + `python-dotenv` / `dotenv` package.".format(**c)),
        ("We currently commit a `.env.example` with fake values. Is that OK?",
         "Yes, `.env.example` with placeholder values is standard — it documents required config without exposing secrets. Never commit the real `.env`. Add `*.env` to `.gitignore` and verify with `git check-ignore -v .env`."),
        ("How do we rotate a secret without downtime?".format(**c),
         "Deploy the new secret alongside the old one, update the app to accept both (dual-read), roll the deploy, then remove the old secret. This is the blue/green secrets pattern — zero downtime."),
    ],

    # 12 — ORM/query
    lambda c: [
        ("Should I use raw SQL or {orm} for this complex join across 4 tables?".format(**c),
         "For complex joins, raw SQL is usually clearer and easier to optimize. {orm} excels at CRUD on single entities. The hybrid: use {orm} for simple queries, `session.execute(text(...))` or repository raw SQL for analytics.".format(**c)),
        ("The {orm} generated query has 12 JOINs. EXPLAIN shows it's doing a hash join on 800k rows.".format(**c),
         "Denormalize that query path. Either a materialized view, a read model updated by an event, or a dedicated summary table. 12-join queries don't scale — rethink the data shape for the read side."),
        ("Is {orm} still worth using if we're doing raw SQL for the hard stuff?".format(**c),
         "Yes. {orm} gives you migrations, schema introspection, connection pooling, and safe parameter binding. Use it as infrastructure, not as a query builder for complex queries.".format(**c)),
    ],

    # 13 — Architecture concept
    lambda c: [
        ("Can you explain {concept} in plain terms?".format(**c),
         "{concept} is a pattern where {explanation}. The core idea is to separate concerns so each part can evolve independently.".format(
             explanation={
                 "dependency injection": "dependencies are provided from outside rather than created internally — the caller controls what implementation is used",
                 "event sourcing": "you store every state change as an immutable event, then derive current state by replaying the event log",
                 "CQRS": "reads and writes use separate models — writes go through a command that validates and persists, reads go through an optimized query projection",
                 "circuit breaker": "after N failures the circuit 'opens' and fast-fails requests for a cooldown period, preventing cascade failures",
                 "idempotency": "the same operation can be applied multiple times without changing the outcome after the first application",
                 "eventual consistency": "replicas may temporarily diverge but are guaranteed to converge to the same state given no new writes",
                 "outbox pattern": "you write the event to an outbox table in the same DB transaction as your state change, then a separate process publishes it — guaranteeing at-least-once delivery",
             }.get(c['concept'], "you solve a specific distributed-systems problem by decoupling producers from consumers"), **c)),
        ("When would you actually use {concept} in a real service?".format(**c),
         "When you hit the pain it solves. Don't add it up front — it adds complexity. You'll know you need it when {signal}.".format(
             signal="your mutable state is hard to audit or replay" if "sourcing" in c['concept'] else
                    "downstream failures are cascading into your service" if "circuit" in c['concept'] else
                    "messages get lost during crashes because you publish before committing" if "outbox" in c['concept'] else
                    "read and write scalability requirements diverge" if "CQRS" in c['concept'] else
                    "duplicate requests from retries are corrupting state" if "idempotent" in c['concept'] else
                    "you keep passing 8-argument constructors and tests become hard to isolate")),
        ("What's the most common mistake when implementing it?".format(**c),
         "Over-engineering it before you need it. Start simple, measure the pain, then introduce {concept} surgically. Most teams that adopt it too early spend months fighting accidental complexity.".format(**c)),
    ],

    # 14 — Code review
    lambda c: [
        ("Can you review this function? It works but feels off:\n```{lang}\ndef process(items):\n    result = []\n    for i in range(len(items)):\n        if items[i] is not None:\n            result.append(items[i] * 2)\n    return result\n```".format(**c),
         "Three things: (1) `range(len(items))` → iterate directly `for item in items`; (2) the `None` check → use a comprehension; (3) no type hints. Cleaner: `def process(items: list) -> list: return [x * 2 for x in items if x is not None]`"),
        ("What about performance — is the list comprehension actually faster?",
         "Yes, list comprehensions are faster than `append` in a loop in CPython because they avoid per-iteration attribute lookup on `result.append`. For large inputs also consider a generator if the caller doesn't need the full list at once."),
        ("Should I add a docstring?".format(**c),
         "Only if the function name and type hints don't already tell the story. `process` is vague enough that a one-liner would help: `\"\"\"Double all non-None items.\"\"\"` — but the real fix is renaming to `double_non_null_items`."),
    ],

    # 15 — Security
    lambda c: [
        ("Is it safe to log the full request body in {fw}?".format(**c),
         "No. Request bodies often contain passwords, tokens, PII, or payment data. Log a sanitized version: drop known-sensitive fields (`password`, `token`, `ssn`, `card_number`) and truncate large payloads. Never log Authorization headers."),
        ("We're doing it today and need to fix it fast. What's the quickest safe approach?",
         "Add a middleware that strips sensitive keys from a copy of the body before logging. Use a blocklist: `['password', 'token', 'secret', 'authorization', 'cookie']`. Ship that, then audit existing logs for exposure."),
        ("How do we handle existing logs that already have sensitive data?".format(**c),
         "Treat it as an incident: identify the window, check who has log access, notify your security team, rotate any exposed credentials immediately, and document the timeline. Don't just delete the logs without a retention policy review."),
    ],

    # 16 — Async/concurrency
    lambda c: [
        ("When should I use async/await vs threads in {lang}?".format(**c),
         "Async for I/O-bound work (HTTP calls, DB queries, file reads) — low overhead, high concurrency. Threads for CPU-bound or blocking code that can't be made async. In Python specifically, the GIL means threads don't parallelize CPU work — use `multiprocessing` there."),
        ("My async service is still slow. I'm awaiting every call.",
         "Awaiting sequentially negates the benefit. Run independent calls concurrently with `asyncio.gather()` (Python) or `Promise.all()` (JS). If calls A and B don't depend on each other, fire them together."),
        ("Any gotchas with `asyncio.gather` I should know?".format(**c),
         "Yes: if one task raises, `gather` cancels the others by default. Pass `return_exceptions=True` if you want all results regardless. Also watch for shared mutable state — async doesn't give you a free pass from race conditions."),
    ],

    # 17 — Dependency management
    lambda c: [
        ("Should I pin exact versions in my {lang} dependencies or use ranges?".format(**c),
         "Pin exact versions in application lock files (poetry.lock, package-lock.json, go.sum) — reproducible builds. Use ranges only in library manifests so consumers aren't boxed in. Never loose ranges in app code."),
        ("What's the risk of not updating pinned deps?",
         "CVEs accumulate silently. Outdated deps are the most common supply-chain attack vector. Run `dependabot` or `renovate` to get automated PRs — review and merge them weekly rather than doing a big bang update every 6 months."),
        ("How do I safely update a dep that has a major version bump?".format(**c),
         "Read the migration guide, update in a separate branch, run the full test suite including integration tests, then do a canary deploy before rolling out fully. Major bumps often have subtle behavior changes tests don't catch — shadow traffic in staging first if you can."),
    ],

    # 18 — File/IO
    lambda c: [
        ("Best way to read a large file in {lang} without loading it all into memory?".format(**c),
         "Stream it line-by-line or in chunks. In Python: `for line in open('file'):`. In Go: `bufio.Scanner`. In Node: `fs.createReadStream` piped through a line reader. Never `readFile` or `File.read()` on multi-GB files."),
        ("I need to process each line and write results to another file concurrently.",
         "Producer-consumer pattern: one goroutine/thread reads and enqueues lines, a pool of workers processes them, another writer drains the output channel. In Python use `concurrent.futures.ThreadPoolExecutor` with a queue. Keep the pipeline bounded to avoid memory blowup."),
        ("How do I handle partial writes if the process crashes mid-file?".format(**c),
         "Write to a temp file, then atomically rename it to the final path. `os.rename` is atomic on POSIX — either you get the old file or the new one, never a partial. Add a `.tmp` suffix and clean up stale tmps on startup."),
    ],

    # 19 — Caching
    lambda c: [
        ("Should I cache {db} query results in {fw}? The query runs every request.".format(**c),
         "If the data is read-heavy and can tolerate slight staleness — yes. Use {db} or Redis with a TTL matching your staleness tolerance. Cache at the service layer, not the DB layer, so the cache is query-aware.".format(**c)),
        ("How do I invalidate the cache when data changes?",
         "Two strategies: TTL-based (simpler, eventually consistent) or event-driven invalidation (precise, more complex). For most CRUD apps, a short TTL (10–60s) is enough. For strong consistency, publish an invalidation event on every write."),
        ("We're seeing cache stampede under high traffic. Fix?".format(**c),
         "Use probabilistic early expiration (PER) or a lock-based approach: when an item expires, one request recomputes while others wait on a short lock. In Redis, use `SET key value EX ttl NX` to implement a simple mutex around the recompute."),
    ],

    # 20 — Kubernetes/infra
    lambda c: [
        ("My pod keeps OOMKilling. I set the memory limit to 512Mi but the service needs more.",
         "Either raise the limit (if the usage is legitimate) or fix the leak. First: `kubectl top pod` to see actual usage over time. If it grows unbounded it's a leak; if it stabilizes at 600Mi, just raise the limit to 768Mi with headroom."),
        ("How do I find the memory leak without a profiler?".format(**c),
         "Add a `/debug/pprof/heap` endpoint (Go) or `memory_profiler` (Python) and sample heap snapshots every 5 minutes. Compare snapshots — the growing allocation type is your leak. Also check for unbounded in-memory caches or event listener accumulation."),
        ("Should I set requests == limits for memory?".format(**c),
         "For memory yes — keeps the pod in the Guaranteed QoS class so the kubelet won't evict it under node pressure. For CPU, set requests conservatively and leave limits loose (or unset) to avoid throttling on a bursty workload."),
    ],

    # ── Non-tech domains ──────────────────────────────────────────────────────

    # 21 — Medicine / symptoms
    lambda c: [
        ("What's the difference between a virus and a bacteria?",
         "Bacteria are single-celled living organisms that replicate on their own and can be killed with antibiotics. Viruses are non-living particles that hijack host cells to reproduce — antibiotics don't work on them."),
        ("So why do doctors still prescribe antibiotics for colds?",
         "They shouldn't, and most don't anymore. Overprescribing antibiotics for viral infections drives antibiotic resistance — one of the biggest public health threats. Colds are viral; antibiotics do nothing except harm your gut microbiome."),
        ("How does antibiotic resistance actually develop?",
         "Random mutations in bacteria occasionally produce resistance. When you take antibiotics, susceptible bacteria die but resistant ones survive and replicate. The more often antibiotics are used unnecessarily, the faster resistance spreads through the population."),
    ],

    # 22 — Finance / investing
    lambda c: [
        ("What's the difference between a stock and a bond?",
         "A stock is ownership in a company — you share in profits and losses. A bond is a loan to a company or government — you earn fixed interest and get your principal back at maturity. Stocks are higher risk/reward; bonds are lower risk/reward."),
        ("Which should I hold if I'm 30 years from retirement?",
         "With 30 years of runway, a higher equity allocation (e.g. 80–90% stocks) makes sense. Time smooths out market volatility. As you approach retirement, shift toward bonds to protect the capital you've built."),
        ("Does the exact fund I pick matter if I'm doing index investing?",
         "Expense ratio matters most. A 0.03% expense ratio (e.g. Vanguard VTSAX) vs a 1% ratio costs you ~30% of your ending portfolio over 30 years due to compounding. For broad-market index funds, pick the lowest-cost option available to you."),
    ],

    # 23 — History
    lambda c: [
        ("Why did the Roman Empire fall?",
         "No single cause — it was a convergence: military overextension, economic inflation from debasing the currency, political instability (50+ emperors in 50 years at one point), pressure from Germanic tribes, and a shrinking tax base from depopulation and plague."),
        ("Did the Eastern Empire also fall at the same time?",
         "No — the Western Empire fell in 476 CE, but the Eastern Empire (Byzantine) survived another thousand years until 1453 CE when Constantinople fell to the Ottoman Turks. Byzantine Rome kept Greek language, culture, and the Eastern Orthodox Church alive."),
        ("What's the most underrated reason historians point to?",
         "Lead poisoning is a fringe theory but infectious disease is seriously underrated. The Antonine Plague (165 CE) and Plague of Cyprian (249 CE) killed millions and hollowed out the military and tax base at critical moments. Demographic collapse precedes political collapse."),
    ],

    # 24 — Psychology / behavior
    lambda c: [
        ("What's the difference between intrinsic and extrinsic motivation?",
         "Intrinsic motivation comes from internal satisfaction — curiosity, mastery, meaning. Extrinsic comes from external rewards — money, grades, praise. Both work, but research (self-determination theory) shows intrinsic motivation leads to more sustained engagement and creativity."),
        ("Can external rewards kill intrinsic motivation?",
         "Yes — this is called the overjustification effect. If you pay someone to do something they already enjoy, they may stop doing it once the payment stops. They've reattributed their motivation from 'I like this' to 'I do this for money'."),
        ("How do you build intrinsic motivation in a team?",
         "Three levers from self-determination theory: autonomy (let people choose how they work), mastery (give challenging but achievable goals), and purpose (connect work to something meaningful). Micromanagement destroys all three simultaneously."),
    ],

    # 25 — Legal concepts
    lambda c: [
        ("What's the difference between civil and criminal law?",
         "Criminal law is the state prosecuting an individual for an offense against society — the burden is 'beyond reasonable doubt' and punishment is prison or fines. Civil law is disputes between private parties — the burden is 'preponderance of evidence' (>50% likely) and remedies are usually monetary."),
        ("Can someone be tried in both for the same act?",
         "Yes. The O.J. Simpson case is the classic example — acquitted criminally (higher standard) but found liable civilly (lower standard). Double jeopardy only bars a second criminal trial for the same offense, not a civil case."),
        ("What does 'beyond reasonable doubt' actually mean in practice?",
         "There's no mathematical definition, but courts describe it as near-certainty — not absolute certainty (impossible) but much more than probable. Studies suggest jurors interpret it as roughly 90–95% confidence. The high bar protects against wrongful convictions."),
    ],

    # 26 — Physics concepts
    lambda c: [
        ("Can you explain why the sky is blue in simple terms?",
         "Sunlight contains all colors. When it hits the atmosphere, gas molecules scatter shorter wavelengths (blue, violet) much more than longer ones (red, orange). Your eye is more sensitive to blue than violet, so the sky looks blue rather than violet."),
        ("Why is the sunset red then?",
         "At sunset, sunlight travels through much more atmosphere to reach your eye. Most blue light scatters away long before it reaches you, leaving the longer-wavelength reds and oranges to dominate. Same physics, different path length."),
        ("Does this mean on the Moon the sky would look black even in daytime?",
         "Exactly. No atmosphere means no scattering, so the sky is black even with the Sun out. You'd see stars in broad daylight and harsh, unfiltered sunlight on one side with stark black shadow on the other."),
    ],

    # 27 — Economics
    lambda c: [
        ("What's the difference between inflation and deflation, and which is worse?",
         "Inflation is a general rise in prices (money loses purchasing power). Deflation is a fall in prices (money gains purchasing power). Mild inflation (~2%) is considered healthy. Deflation sounds good but is dangerous — it incentivizes consumers to delay spending, which spirals into recession."),
        ("Why does the Fed target 2% inflation specifically?",
         "It's a buffer. At exactly 0% inflation, any negative shock tips into deflation. 2% gives room to cut interest rates (which can't go far below 0%) as a stimulus tool. It's a judgment call balancing growth, employment, and purchasing power — not a law of nature."),
        ("What's quantitative easing in plain English?",
         "The Fed creates new money and uses it to buy government bonds (and sometimes other assets) from banks. This pushes money into the financial system, lowers long-term interest rates, and encourages lending and investment. Critics say it inflates asset prices and widens inequality."),
    ],

    # 28 — Biology
    lambda c: [
        ("How does a vaccine actually train the immune system?",
         "A vaccine introduces an antigen — a harmless piece of a pathogen (protein, weakened virus, mRNA instructions to make a protein) — that your immune system treats as a threat. It produces antibodies and, crucially, memory B and T cells that recognize that antigen in the future."),
        ("What's the difference between mRNA vaccines and traditional ones?",
         "Traditional vaccines use weakened/killed pathogens or protein subunits. mRNA vaccines give your cells instructions to produce a specific protein (e.g. the spike protein), triggering the immune response. mRNA is degraded quickly and never enters the cell nucleus or alters DNA."),
        ("Why do some vaccines need boosters?",
         "Antibody levels wane over time for some pathogens. Boosters restimulate the memory cells, raising antibody titers again. For fast-mutating viruses like influenza or COVID variants, boosters may also update the antigen to match the current circulating strain."),
    ],

    # 29 — Nutrition / health
    lambda c: [
        ("Is saturated fat actually bad for you? I keep reading conflicting things.",
         "The evidence is genuinely mixed. Early research (Ancel Keys) linked saturated fat to heart disease via LDL cholesterol. More recent meta-analyses find weak or no independent association when carbohydrates replace the fat. Context matters — replacing saturated fat with refined carbs may be worse than keeping it."),
        ("What about the difference between LDL and HDL?",
         "HDL ('good') carries cholesterol away from arteries back to the liver. LDL ('bad') deposits it in artery walls. But even LDL is nuanced — small dense LDL particles are more atherogenic than large buoyant ones. Total cholesterol or even LDL-C alone is a crude marker."),
        ("So what does the best current evidence say to actually eat?",
         "Minimize ultra-processed foods, refined sugars, and trans fats (strong consensus). Beyond that: mostly whole foods — vegetables, legumes, nuts, fish, quality protein. Mediterranean and DASH patterns have the most consistent positive evidence. Calories in/out still matters for weight."),
    ],

    # 30 — Philosophy
    lambda c: [
        ("What's the difference between deontological and consequentialist ethics?",
         "Deontological ethics (Kant) says actions are inherently right or wrong regardless of outcomes — lying is always wrong even if it saves lives. Consequentialism (Mill/Bentham) judges actions solely by their outcomes — the right action maximizes good consequences."),
        ("Which do most people actually follow in practice?",
         "Neither purely. Moral psychology research (Jonathan Haidt, Joshua Greene) suggests humans use both: fast intuitive responses (deontological-feeling) and slower deliberate reasoning (more consequentialist). Trolley-problem studies show context shifts people between the two."),
        ("Does philosophy actually help with real ethical decisions?",
         "It helps by surfacing hidden assumptions, clarifying which values are in conflict, and making trade-offs explicit. It rarely gives you a clean answer, but it stops you from being unknowingly inconsistent — which is most of what goes wrong in real ethical failures."),
    ],

    # 31 — Climate / environment
    lambda c: [
        ("What's the difference between weather and climate?",
         "Weather is the atmospheric state right now or over days — temperature, precipitation, wind. Climate is the statistical pattern over 30+ years. 'Climate is what you expect, weather is what you get.' A cold day doesn't refute warming; a decade of shifted averages does."),
        ("How do scientists know current warming is human-caused and not natural?",
         "Multiple independent lines: isotopic fingerprinting shows the CO2 increase is from fossil fuels (different carbon isotope ratio than volcanic CO2); the stratosphere is cooling while the troposphere warms (greenhouse effect signature, not solar); warming correlates tightly with CO2 emissions, not solar output."),
        ("What's the most effective individual action someone can take?",
         "Flight reduction and plant-rich diet have the largest individual footprints. But individual action is dwarfed by systemic change — supporting policy (carbon pricing, clean energy mandates) and voting have higher leverage than any lifestyle change. Both matter, but don't let 'personal responsibility' framing substitute for political action."),
    ],

    # 32 — Education / learning
    lambda c: [
        ("What's the most effective way to study and actually retain information?",
         "Spaced repetition and active recall beat passive re-reading by a large margin. Instead of highlighting and re-reading, close the book and try to recall the material. Then space out reviews — test yourself after 1 day, 3 days, 1 week, 1 month."),
        ("Why does re-reading feel productive even though it's not?",
         "Familiarity illusion — the text feels easier the second time, so your brain signals 'I know this'. But recognition (seeing and thinking 'I know that') is much weaker than recall (retrieving it from memory). The effort of retrieval is what builds the memory trace."),
        ("Is there a good tool for spaced repetition?",
         "Anki is the gold standard — free, open-source, algorithm-driven scheduling. For language learning, Duolingo uses a similar algorithm. The tool matters less than the habit: 20 minutes of daily reviews beats a 3-hour cram session the night before."),
    ],

    # 33 — Writing / communication
    lambda c: [
        ("How do I make my writing more concise without losing meaning?",
         "Cut these first: filler openers ('It is important to note that...'), redundant pairs ('each and every', 'first and foremost'), weak intensifiers ('very', 'really', 'quite'), and passive voice where active is clearer. Then read each sentence and ask: what's doing real work here?"),
        ("What about technical writing specifically?",
         "Lead with the conclusion, not the method. Readers want to know the answer before the argument. Use numbered lists for sequential steps, bullet lists for non-sequential items. Define acronyms on first use. Write for the person who will implement this, not the person who already knows it."),
        ("Any advice for writing emails that actually get responses?",
         "Subject line = specific action + deadline: 'Review: Q2 proposal — needed by Friday'. First sentence states what you need and why. Keep it under 150 words if possible. End with a single clear question or action, not multiple options. People respond to clarity, not length."),
    ],
]


TOKEN_BUDGETS = [256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048]

USER_PREFIXES = {
    0: [
        "",
        "Quick sanity check.",
        "Need a second set of eyes on this.",
        "I'm debugging this in {environment}.",
        "This popped up in the {service}.",
        "We noticed this after {change_event}.",
    ],
    1: [
        "",
        "I already ruled out the obvious stuff.",
        "I'm trying not to paper over it.",
        "{stakeholder} wants the practical next step.",
        "I'm looking for the lowest-risk follow-up.",
    ],
    2: [
        "",
        "From an on-call angle.",
        "This is for the team owning {service}.",
        "Goal here is {goal}.",
        "I'm trying to keep the fix boring.",
    ],
}

USER_SUFFIXES = {
    0: [
        "",
        "This is happening in {environment}.",
        "I'm trying to keep {service} stable.",
        "The issue started after {change_event}.",
    ],
    1: [
        "",
        "I want the answer that scales in production.",
        "We're trying to avoid a band-aid fix.",
        "This is blocking {stakeholder}.",
    ],
    2: [
        "",
        "I'd rather solve it once at the right layer.",
        "This needs to be supportable at 2 a.m.",
        "We're optimizing for {goal}.",
    ],
}

ASSISTANT_PREFIXES = {
    0: [
        "",
        "Short answer.",
        "My first read.",
        "Most likely explanation.",
        "The common pattern here.",
    ],
    1: [
        "",
        "Next thing I'd test.",
        "Given that context.",
        "That points to a common failure mode.",
        "Here's the practical next step.",
    ],
    2: [
        "",
        "Practical answer.",
        "How I'd handle it in production.",
        "Lowest-risk move.",
        "Operationally, this is the boring default.",
    ],
}

ASSISTANT_SUFFIXES = {
    0: [
        "",
        "That's where I'd start.",
        "I'd verify that before changing code.",
        "It's a common production gotcha.",
    ],
    1: [
        "",
        "That keeps the diagnosis honest.",
        "It also makes the fix easier to verify.",
        "That gives you a concrete next experiment.",
    ],
    2: [
        "",
        "That's the boring answer, and boring is good in production.",
        "That tends to age well operationally.",
        "It also keeps the system easier to support.",
    ],
}

START_VARIANTS = {
    "What does": ["What exactly does", "Can you remind me what", "How should I read"],
    "What's the difference between": ["How do you compare", "How is", "What separates"],
    "What's the best way to": ["How should I", "What's the safest way to", "What's a practical way to"],
    "What's the right": ["What counts as the right", "What's a sensible", "What's the pragmatic"],
    "What's the cleanest place to": ["Where should I", "What's the best place to", "What's the safest layer to"],
    "Should I": ["Would you", "Is it better to", "Do you recommend I"],
    "How do I": ["What's the best way to", "How should I", "What's a safe way to"],
    "How do we": ["What's the safest way to", "How should we", "What's the practical way to"],
    "When should I": ["In what cases should I", "When would you", "What's the right time to"],
    "When would you": ["In which cases would you", "What situations make you", "When do you usually"],
    "Can you explain": ["Can you break down", "Can you walk me through", "Explain"],
    "Can you review": ["Can you sanity-check", "Can you take a look at", "Would you review"],
    "Why do": ["What makes", "Why would", "Why do people"],
    "Why does": ["What makes", "Why would", "Why does it"],
    "Is it safe to": ["Is it a bad idea to", "Would you consider it safe to", "Is it reasonable to"],
}

INLINE_VARIANTS = {
    "Likely cause?": ["What would you suspect first?", "What's the usual culprit?", "What tends to cause that?"],
    "Could it be": ["Do you think it could be", "Would you look at", "Is there a chance it's"],
    "What if": ["How would you handle it if", "What should I do if", "If it turns out"],
    "quick win": ["low-effort win", "fastest useful improvement", "cheapest useful improvement"],
    "full optimization": ["deeper optimization pass", "full tuning project", "bigger optimization effort"],
    "most common mistake": ["failure mode people hit most often", "mistake that shows up the most", "easiest way to get this wrong"],
    "most underrated reason": ["reason people overlook the most", "most underappreciated reason", "cause people underrate"],
}


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def choose_fmt(options, ctx):
    return random.choice(options).format(**ctx)


def replace_once(text: str, variants: dict[str, list[str]]) -> str:
    matches = [source for source in variants if source in text]
    if not matches:
        return text
    source = random.choice(matches)
    return text.replace(source, random.choice(variants[source]), 1)


def stitch_text(*parts: str) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return ""

    if any("```" in part for part in cleaned):
        return "\n\n".join(cleaned)

    text = " ".join(cleaned)
    return re.sub(r"\s+", " ", text).strip()


def diversify_turn(text: str, role: str, role_turn_idx: int, ctx: dict) -> str:
    varied = text
    if "```" not in varied:
        if random.random() < 0.8:
            varied = replace_once(varied, START_VARIANTS)
        if random.random() < 0.55:
            varied = replace_once(varied, INLINE_VARIANTS)

    if role == "user":
        prefix = choose_fmt(USER_PREFIXES.get(role_turn_idx, [""]), ctx)
        suffix = choose_fmt(USER_SUFFIXES.get(role_turn_idx, [""]), ctx)
    else:
        prefix = choose_fmt(ASSISTANT_PREFIXES.get(role_turn_idx, [""]), ctx)
        suffix = choose_fmt(ASSISTANT_SUFFIXES.get(role_turn_idx, [""]), ctx)

    return stitch_text(prefix, varied, suffix)


def conversation_key(conversation: list) -> str:
    return json.dumps(conversation, sort_keys=True, ensure_ascii=False)


def last_user_query(conversation: list) -> str:
    for turn in reversed(conversation):
        if turn["role"] == "user" and turn["content"]:
            return turn["content"]
    return ""


def make_retrieval_candidates(conversation: list, k: int = 8) -> list:
    user_turns = [t["content"] for t in conversation if t["role"] == "user"]
    query = " ".join(user_turns[-2:]) if len(user_turns) >= 2 else (user_turns[-1] if user_turns else "")

    chunks = []
    for turn in conversation:
        if turn["role"] == "assistant":
            # split on sentence boundaries
            raw = turn["content"].replace(" — ", ". ").replace(": ", ". ")
            for sent in raw.split(". "):
                sent = sent.strip()
                if len(sent.split()) >= 5:
                    chunks.append(sent)

    if not chunks:
        return []

    scored = sorted([(c, _jaccard(query, c)) for c in chunks], key=lambda x: x[1], reverse=True)
    top = scored[:k]

    raw_scores = [s for _, s in top]
    lo, hi = min(raw_scores), max(raw_scores)
    norm = [(s - lo) / (hi - lo) if hi != lo else 0.0 for s in raw_scores]

    return [{"text": chunk, "importance": round(float(n), 4)} for (chunk, _), n in zip(top, norm)]


def render_template(tmpl_fn):
    ctx = make_ctx()
    turns = tmpl_fn(ctx)
    role_counts = {"user": 0, "assistant": 0}
    conversation = []

    for role, content in itertools.chain.from_iterable(
        [("user", u), ("assistant", a)] for u, a in turns
    ):
        role_turn_idx = role_counts[role]
        conversation.append({
            "role": role,
            "content": diversify_turn(content, role, role_turn_idx, ctx),
        })
        role_counts[role] += 1

    return conversation


def generate():
    dataset = []
    seen_conversations = set()
    seen_queries = set()
    attempts = 0

    while len(dataset) < NUM_SAMPLES:
        attempts += 1
        if attempts > NUM_SAMPLES * 200:
            raise RuntimeError("Unable to generate enough unique conversations; expand template variety.")

        tmpl_fn = TEMPLATES[attempts % len(TEMPLATES)]
        convo = render_template(tmpl_fn)
        convo_key = conversation_key(convo)
        query_key = last_user_query(convo)

        if convo_key in seen_conversations or query_key in seen_queries:
            continue

        seen_conversations.add(convo_key)
        seen_queries.add(query_key)
        dataset.append({
            "id": f"conv_{len(dataset)}",
            "conversation": convo,
            "retrieval_candidates": make_retrieval_candidates(convo),
            "token_budget": random.choice(TOKEN_BUDGETS),
        })
    return dataset


if __name__ == "__main__":
    os.makedirs(OUT_PATH.parent, exist_ok=True)
    dataset = generate()
    with OUT_PATH.open("w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} samples → {OUT_PATH}")
