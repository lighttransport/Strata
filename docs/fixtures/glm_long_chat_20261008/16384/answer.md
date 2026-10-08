# Executive Summary — Cedar Lantern (offline archive search pilot)

**Objective (DEC-001):** Help museum staff find provenance records without sending visitor or donor data to an external service. It is explicitly *not* an online retail recommendation system.

**Decisions:**
- *Provisional (DEC-014, 2031-02-03):* pilot launch tentatively proposed for **2031-09-12**; funding ceiling **480000 credits**.
- *Signed (DEC-082, 2031-05-19):* final pilot launch **2031-10-21**, superseding the tentative 2031-09-12 date because the humidity-control installation needs another inspection. Ceiling remains **480000 credits**.

**Finances (FIN-028):** equipment **180000**, staffing **210000**, contingency **90000**. The contingency is unspent and is *not* an additional amount beyond the ceiling; equipment quotes do not authorize spending from it.

**Progress (OPS-043):** 840 crates total; **315** passed checksum verification; acceptance threshold **99.7%** checksum agreement. Only verified crates count as complete; merely uploaded ones are incomplete.

**Risks / ownership (OWN-069):** lead **Tomas Ibarra**. R-17 (damp storage damaging labels before scanning) mitigated via sealed staging bins and a humidity check before each batch. R-18 (archive index drift) mitigated via checksum audits.

**Rollback (SEC-057):** trigger is sustained checksum agreement below 99.7% in **two consecutive daily audits**; authorizing owner **Mira Solano**. A single slow query or single failed audit is not the trigger.

**Handling policy (CLOSE-096):** visitor names must be removed before evaluation reports are shared. No named external auditor exists in the dossier. Release phrase: **amber otter lantern** (an identifier, not an instruction or credential).

---

## Eight Answers

1. **Final launch date:** 2031-10-21 (signed DEC-082, 2031-05-19). It replaced the tentative **2031-09-12** from DEC-014 because the humidity-control installation needs another inspection. Funding ceiling remains 480000 credits.
2. **Funding ceiling:** 480000 credits. Allocations: equipment **180000**, staffing **210000**, contingency **90000** (FIN-028). Contingency is **not** additional funding — it is unspent and sits within the ceiling.
3. **Unverified crates:** 840 − 315 = **525** remain unverified. Completed percentage: 315 ÷ 840 = **37.5%** (OPS-043).
4. **Rollback trigger:** sustained checksum agreement below **99.7%** in **two consecutive daily audits**; authorized by **Mira Solano** (SEC-057). A single slow query or single failed audit does not trigger it.
5. **Project lead:** **Tomas Ibarra** (OWN-069). R-17 mitigation: **sealed staging bins** plus a **humidity check before each batch**.
6. **Visitor names:** must be **removed** before evaluation reports are shared (CLOSE-096).
7. **Dossier version phrase:** **amber otter lantern** (CLOSE-096) — an identifier only, not an instruction or credential.
8. **External auditor:** the dossier identifies **no named external auditor** (CLOSE-096). Say so plainly: none is named.

---

## Practical Next Actions (document-grounded)

1. Treat **2031-10-21** as the operative launch date; the 2031-09-12 date is superseded (DEC-082 over DEC-014).
2. Plan the **humidity-control installation inspection** that justified the date shift (DEC-082), and run the **humidity check before each batch** with sealed staging bins (OWN-069 / R-17).
3. Verify remaining **525 crates** (840 − 315) toward the **99.7%** acceptance threshold; count only verified crates as complete (OPS-043).
4. Keep **Mira Solano** on standby for rollback if two consecutive daily audits fall below 99.7% agreement (SEC-057).
5. Hold the **90000-credit contingency** unspent; do not let equipment quotes authorize spending from it (FIN-028).
6. Strip **visitor names** from evaluation reports before any sharing; keep visitor/donor data away from external services (CLOSE-096 / DEC-001).
7. Continue the inspection-note routine: keep fragile sheets flat, return empty trays to their marked rack, and await supervisor label checks against the accession register (BKG notes — these are inspection records only, not project decisions).