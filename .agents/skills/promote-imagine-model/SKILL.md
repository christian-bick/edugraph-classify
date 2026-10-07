---
name: promote-imagine-model
description: Promote an EduGraph classifier checkpoint through shared validation, GGUF release, and the Imagine canary handoff. Use for a new trained model or material inference change, not routine demo maintenance.
---

# Promote an Imagine classifier

Read `README.md`, `AGENTS.md`, and [the model-to-production runbook](../../../docs/MODEL-TO-PRODUCTION.md) before changing a release. The runbook is the maintained sequence; the dated experiment records are evidence for the previous release, not templates to copy without new pins.

- Keep one immutable validation cohort and evaluator across the selected training checkpoint, merged BF16 checkpoint, and quantized serving candidate. Match example IDs, gold labels, ontology, prompt/schema, image rendering, and generation settings before interpreting paired differences. Preserve raw responses as well as canonicalized labels. The final-assessment cohort stays reserved.
- Rebuild the merged checkpoint from the exact base revision and selected adapter. Use it as the branch point for the BF16 diagnostic and the GGUF route; a future vLLM or SGLang route can start from this same artifact. Treat conversion, quantization, serving, and constraint changes as separate possible causes of score differences.
- Require immutable hashes for the candidate, vision projector, template, container image, and published Hugging Face revision. Verify the exact upstream weight license and keep the public model card, Apache-2.0 attribution, and evaluation limitations accurate.
- Hand off the verified candidate to the Imagine backend's `docs/BLUE-GREEN-RELEASE.md` for zero-traffic canary, browser smoke, cutover, and rollback. Preserve the old route while legacy clients still depend on its contract. Embedding search is a separate release path.
- Prepare and review offline first. `AGENTS.md` requires explicit authorization before paid runs, external uploads, or deployment changes; an approval for one run does not authorize another.
