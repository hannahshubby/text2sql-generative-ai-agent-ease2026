## A1–A4 Evolution Overview (Baseline → Failure-Closing Pipeline)

To clarify how DECAL is operationalised, we briefly summarise the **baseline-to-final** evolution across four agent configurations (A1–A4). Each step introduces control mechanisms to reduce silent failures and improve correctability.

### A1 — Single Text-to-SQL Baseline (Retrieval → Generation)
**Goal:** Establish a comparable baseline aligned with common Text-to-SQL practice.  
**Flow:**
1. **Query Retrieval:** Embed the user query and retrieve top-k schema/code/term chunks (e.g., FAISS).
2. **LLM SQL Generation:** Generate SQL from retrieved context.

**Typical weakness:** Limited stage-wise visibility—most failures are only observable in the final SQL, making localisation and systematic correction difficult.

---

### A2 — Vector + Ontology Multi-Agent (Role-Decomposed with Fixed Artifacts)
**Goal:** Reduce cascading failures and debug opacity by decomposing decisions into roles and fixing intermediate outputs.  
**Flow:**
1. **Candidate Generation:** Retrieve and propose candidates (schema/term/code) with scores.
2. **LLM-Assisted Finalisation:** Refine candidate selections using intent under validation gates.
3. **Column Finalisation:** Commit selected columns into a standardised artifact schema (with rationale/confidence).
4. **Table/Join Planning:** Plan joins via deterministic graph-based planning under bounded search.
5. **Constraint Planning:** Generate filters/ordering only when supported by evidence (intent + plan).
6. **Artifacts as Interfaces:** Each stage emits JSON artifacts that become the *only* input to the next stage.

**Net effect:** Failures become observable *where decisions are made*, enabling separation of semantic selection errors from structural planning errors.

---

### A3 — Hierarchical Ontology Agent (Structured Understanding → Rule-Based Commitment)
**Goal:** Make ambiguity explicit early and constrain grounding via hierarchical term organisation.  
**Flow:**
1. **Query Understanding (Structured Plan):** Convert the request into a JSON plan; keep unresolved items in `ambiguous_phrases` instead of auto-committing.
2. **Lexicon/TTL Grounding:** Ground `termMentions` and `codeMentions` from the structured plan using controlled vocabularies.
3. **Rule-Based Final Selection:** Commit physical names via explicit rules (explainable, reproducible).
4. **Table Linking + (Optional) Semantic Join Check:** Perform constrained linking; optionally reject meaningless joins via a verifier stage.

**Net effect:** Stronger explainability and earlier ambiguity control, but structural robustness depends heavily on admissibility checks and link constraints.

---

### A4 — Final Agent (DECAL-Implementing Failure-Closing Pipeline)
**Goal:** Close the observed failures by combining **catalog validation, evidence gates, decision contracts, link constraints, and an artifact loop** into a single pipeline.  
**Flow (high-level):**
1. **Query Understanding:** Produce a structured interpretation (bounded format; ambiguity externalised).
2. **Lexicon Grounding:** Retrieve/ground term + code candidates from controlled vocabularies.
3. **Final Selection:** Select candidates with explicit rules and/or bounded LLM assistance.
4. **Final Confirmation (Existence-Space Gate):** Accept only items that cross-validate against catalog indices; reject and record reasons otherwise.
5. **Table Linking Engine (Graph-Constrained):** Choose root/anchor and build join paths under bounded search and admissible links.
6. **Join Semantic Verifier:** Re-block meaningless joins that are structurally possible but semantically invalid.
7. **SQL Planner → SQL Synthesis:** Build a deterministic plan from confirmed bindings and synthesise SQL.
8. **Gold Reverser → Benchmark Compare (Artifact Loop):** Decompose admin-corrected SQL back into the same intermediate artifact schema and compute artifact-level discrepancies to localise failures and drive correction.

**Key property:** Every step emits a versioned, schema-fixed artifact; the system supports reverse-and-compare diagnosis rather than relying on end-SQL inspection only.
