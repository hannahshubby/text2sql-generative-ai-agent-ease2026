```md
# DECAL Principles and Guidelines

This section presents design norms to address the observed failures, formulated as the five DECAL principles (C, E, D, L, A) and two operational guidelines (V, R). Here, a **principle** is a general norm that is not tied to a specific task (e.g., Text-to-SQL), while a **guideline** is an operational/deployment discipline that broadens the applicability of the principles. Section 3.3 then describes how each principle is instantiated in our system modules and artifacts.

---

## 1 C — Catalog-Grounded

**C** states that the system must restrict what it can reference or generate to a pre-approved **catalog** (the “space of existence”). The catalog includes entities, attributes (fields), values (codes/constants), and the allowable relations among them. In other words, it is not about “being able to produce the correct answer,” but about **guaranteeing the ability not to reference what does not exist**. This structurally suppresses failures such as non-executable outputs, invalid schema references, and leakage of out-of-domain knowledge.

1. The system must not “add new items by inference” outside the catalog.
2. If producing an output requires an item not present in the catalog, the system must refrain from generating the output and instead leave it unresolved or delegate to a human.
3. Catalog updates must occur only through an approved governance/verification process—not through model inference.

---

## 2 E — Evidence-Gated

**E** requires that every conclusion (selection/condition/structure) be justified by explicit **evidence**. Evidence is restricted to auditable sources such as the input (user requirements), the catalog, approved logs/data, or pre-defined rule sets. This shifts evaluation from “plausibility” to **sufficiency of evidence**. The principle is especially effective at suppressing arbitrary condition insertion, negation-logic errors, and evidence-free expansion that triggers cascading failures.

1. Conditions generated without evidence (filters/sorting/time windows/priorities, etc.) are prohibited.
2. If an “assumption” is required, the system must not blend the assumed value into the output; it must be separated and surfaced explicitly, or converted into a hold/question.
3. Evidence must be stored with the output so that “why it was done” can be reconstructed after the fact.

---

## 3 D — Decision-Contracted

**D** requires that the authority and limits of automated decisions be explicitly defined as a **Decision Contract**. The contract specifies (i) when auto-finalisation is permitted, (ii) when the system must hold/delegate, and (iii) how the system must fail safely. The goal is not to eliminate all errors, but to reduce silent semantic drift and unauthorised finalisation by making it explicit **what decision was made and when**.

1. Automatic finalisation is prohibited by default when ambiguity (candidate conflicts) exists.
2. Even when auto-finalisation is permitted, the conditions (e.g., uniqueness, sufficient evidence, no conflict) must be explicitly stated and checked.
3. Every decision must be logged with the “decision rule/evidence/confidence level” so that it is auditable and reproducible.

---

## 4 L — Link-Constrained

**L** states that when composing or linking multiple components, the composition must follow allowable **link constraints**. These constraints may include a relationship graph, directionality, permitted paths, and cost/limit conditions. **L** controls structural failures such as combinatorial explosion, incorrect anchor selection, directionality errors, and meaningless joins/links. In complex enterprise systems, this is essential to maintain performance, accuracy, and stability simultaneously.

1. Linking must not be “link whenever possible,” but “link only when necessary and permitted.”
2. Path search for linking must include limits (e.g., hop limits, candidate caps, cost functions) and must not allow uncontrolled combinatorial explosion.
3. Structural feasibility does not guarantee semantic validity; where needed, a semantic validation step must be introduced.

---

## 5 A — Artifact-Looped

**A** requires the system to fix intermediate states as standardised **artifacts**, and to provide a diagnosis/correction loop by reconstructing ideal intermediate states from outcomes (or gold/operational criteria) and comparing them. This reduces debugging opacity (not knowing why something failed), lowers operational cost, and enables the system to evolve through cumulative, repeatable improvements.

1. Each stage output must be stored as an artifact with a fixed schema, traceable without overwriting.
2. When an output error occurs, the system must be able to decompose the error into stage-wise differences (diffs).
3. Fixes must not accumulate as ad-hoc patches; they must be applied through repeatable artifact-based correction procedures.

---

## 6 V — Guideline: Role-Bounded LLM Placement

**V** is an operational guideline: do not place the LLM as an “all-powerful decision maker.” Instead, deploy it in clearly defined roles with **role-bounding** so that model variability, hallucination amplification, and multi-step instability are constrained within an operationally manageable envelope.

1. LLM inputs/outputs should be fixed in structured formats as much as possible so that verification and gating are feasible.
2. LLM conclusions may be finalised only after passing the C/E/D/L gates.
3. Provide a deterministic fallback path so that the system does not collapse even when the LLM fails.

---

## 7 R — Guideline: Reproducibility and Trace Discipline

**R** is an operational guideline for trace/version/log discipline to make outcomes reproducible for identical inputs. It secures both evaluability and operational trust—particularly for enterprise requirements like “why is the same request different today than yesterday?”

1. Log the input, catalog version, model version, parameters, and artifact versions together.
2. Standardise per-step trace IDs and log schemas so that results are comparable.
3. Turn “result variability” into an observable metric in both experiments and operations, and define acceptable variability bounds.
```
