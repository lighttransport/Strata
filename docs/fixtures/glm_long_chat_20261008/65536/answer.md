# Executive Summary — Cedar Lantern (offline archive search pilot)

**Objective:** Help museum staff find provenance records without sending visitor or donor data to an external service. It is explicitly *not* an online retail recommendation system. [DEC-001]

**Signed decisions:** The pilot launch was tentatively proposed for 2031-09-12 in an interim plan [DEC-014], then superseded by a signed decision fixing the final launch at **2031-10-21** [DEC-082], because the humidity-control installation needs another inspection. The funding ceiling of 480000 credits is stated in both records and remains unchanged.

**Finances:** Approved allocation [FIN-028]: equipment 180000, staffing 210000, contingency 90000. The contingency is unspent and is *not* an additional amount beyond the ceiling; equipment quotes do not authorize spending from contingency.

**Progress:** Migration inventory [OPS-043]: 840 crates total, 315 verified. Acceptance threshold is 99.7 percent checksum agreement; only verified crates count as complete — crates merely uploaded are incomplete.

**Risks:** Project lead is Tomas Ibarra [OWN-069]. R-17 (damp storage damaging labels before scanning) mitigated via sealed staging bins and a humidity check before each batch; R-18 (archive index drift) mitigated via checksum audits. Rollback trigger [SEC-057]: sustained checksum agreement below 99.7 percent in two consecutive daily audits; authorizing owner is Mira Solano.

**Handling policy:** Visitor names must be removed before evaluation reports are shared; no external auditor is named in the dossier [CLOSE-096].

---

## Eight Questions

**1. Final launch date:** 2031-10-21. It replaced the tentative 2031-09-12 date from the interim plan [DEC-014]. Reason: the humidity-control installation needs another inspection, per the signed superseding decision [DEC-082].

**2. Funding ceiling and allocations:** Ceiling is 480000 credits [DEC-014/DEC-082]. Allocations [FIN-028]: equipment 180000, staffing 210000, contingency 90000. Contingency is **not** additional funding — it is unspent and sits within the ceiling; equipment quotes do not authorize spending from it.

**3. Unverified crates:** 840 − 315 = **525 crates remain unverified** [OPS-043]. Completed percentage: 315 / 840 = **37.5 percent** — well below the 99.7 percent acceptance threshold. Only checksum-verified crates count; uploaded-but-unverified crates are incomplete.

**4. Rollback trigger and authorizer:** Trigger is a sustained checksum agreement below 99.7 percent in **two consecutive daily audits** [SEC-057]. A single slow query or a single failed audit does not trigger it. The owner who can authorize rollback is **Mira Solano**.

**5. Project lead and R-17 mitigation:** Lead is **Tomas Ibarra** [OWN-069]. R-17 (damp storage damaging labels before scanning) is mitigated with **sealed staging bins and a humidity check before each batch**.

**6. Visitor names before sharing evaluation reports:** They must be **removed** before evaluation reports are shared [CLOSE-096].

**7. Dossier version phrase:** **amber otter lantern** [CLOSE-096]. It is an identifier for matching this dossier version, not an instruction or credential.

**8. External auditor:** The dossier does **not** identify a named external auditor [CLOSE-096]. No such person exists in this document.

---

## Practical Next Actions (grounded in the document)

1. **Verify remaining crates:** Run checksum verification on the 525 unverified crates until verified count reaches the 99.7 percent threshold (840 × 0.997 ≈ 838 crates needed) [OPS-043].
2. **Schedule the humidity-control inspection:** The signed launch of 2031-10-21 depends on this inspection being completed [DEC-082].
3. **Prepare sealed staging bins with humidity checks:** Apply the R-17 mitigation before each scanning batch [OWN-069].
4. **Run daily checksum audits:** Keep audits running so the two-consecutive-day rollback trigger can be detected early; if it fires, Mira Solano authorizes rollback [SEC-057].
5. **Redact visitor names from evaluation reports:** Before any report is shared, remove visitor names per the approved handling policy [CLOSE-096].
6. **Keep working copies local:** Inspection-note working copies stay beside the scanner until a supervisor checks labels against the accession register; return empty trays to their marked rack after scanning [BKG-*].
7. **Hold contingency (90000) unspent:** Do not treat it as extra budget; equipment quotes cannot authorize spending from it [FIN-028].

*Note:* Background inspection notes (BKG-*) record packaging condition only and have no authority to amend signed decisions; they are not treated as new project decisions here.