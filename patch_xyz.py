#!/usr/bin/env python3
"""
Patches xyz-llama so the xyz drafter and the multimodal projector can be loaded together.

usage: patch_xyz.py <src_dir> [apply_B: 0|1]

Every anchor must match EXACTLY ONCE, otherwise the build fails. That is on purpose:
if you bump REF and upstream moved the code, you want a loud failure, not a silently
unpatched binary.

server-common.cpp
  C1/C2 : REQUIRED. get_tokens()/set_token() asserted !has_mtmd, and has_mtmd is true for EVERY
          slot once a projector is loaded. The assertions now check for actual media.

server-context.cpp
  0-guard : lets the projector load when a drafter is configured (env-gated).
  A       : get_n_draft_max() returns 0 for a slot whose context holds media -> plain decode.
  D1      : the mtmd callback does not mirror media chunks into the drafter.
  B1/B2   : (optional, APPLY_B=1) text-only fast path in prepare_speculative_draft().

common/speculative.cpp
  D2      : the mirror loop skips a ubatch that cannot continue ctx_dft's KV (or its stash) contiguously
            (everything after a media chunk: the target uses M-RoPE-compressed positions).
  E       : stash_flush() does not try to decode stale rows that cannot continue ctx_dft's KV.
"""
import pathlib
import sys

src_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/src")
apply_b = (sys.argv[2] if len(sys.argv) > 2 else "0") == "1"


class Patcher:
    def __init__(self, rel: str):
        self.path = src_dir / rel
        self.text = self.path.read_text()

    def sub(self, old: str, new: str, name: str) -> None:
        n = self.text.count(old)
        if n != 1:
            sys.exit(f"PATCH FAILED [{name}] in {self.path.name}: expected exactly 1 match, found {n}")
        self.text = self.text.replace(old, new)
        print(f"patch [{name}]: applied")

    def save(self) -> None:
        self.path.write_text(self.text)


# ============================ server-common.cpp ==============================
common = Patcher("tools/server/server-common.cpp")

common.sub(
    "const llama_tokens & server_tokens::get_tokens() const {\n"
    "    GGML_ASSERT(!has_mtmd);",
    "const llama_tokens & server_tokens::get_tokens() const {\n"
    "    GGML_ASSERT(map_idx_to_media.empty()); // xyz fork: no media chunks, instead of 'no projector loaded'",
    "C1-get_tokens",
)
common.sub(
    "GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled",
    "GGML_ASSERT(map_idx_to_media.empty()); // xyz fork: only allow this if there is no media",
    "C2-set_token",
)
common.save()

# ============================ server-context.cpp ============================
ctx = Patcher("tools/server/server-context.cpp")

ctx.sub(
    "if (has_spec && !params_base.mmproj.path.empty()) {",
    'if (has_spec && !params_base.mmproj.path.empty() && !getenv("XYZ_ALLOW_MMPROJ_WITH_DRAFTER")) {',
    "0-guard",
)

# A: find_next_media_chunk() uses upper_bound(idx) (strictly AFTER idx), so a chunk at index 0 is
# invisible to it; tokens[0] == LLAMA_TOKEN_NULL covers that case.
ctx.sub(
    "// determine the max draft that fits the current slot state",
    "// xyz fork: a slot whose context holds media (image/audio) runs as plain decode, no drafting\n"
    "        if (prompt.tokens.has_mtmd &&\n"
    "            (prompt.tokens.find_next_media_chunk(0).first != nullptr ||\n"
    "             (!prompt.tokens.empty() && prompt.tokens[0] == LLAMA_TOKEN_NULL))) {\n"
    "            return 0;\n"
    "        }\n"
    "\n"
    "        // determine the max draft that fits the current slot state",
    "A-no-draft-with-media",
)

# D1: never mirror a media chunk into the drafter (its rows carry mtmd embeddings at M-RoPE positions).
ctx.sub(
    "static auto cb = [](llama_batch batch, void * user_data) {\n"
    "                    common_speculative * spec = static_cast<common_speculative *>(user_data);\n"
    "                    if (!common_speculative_process(spec, batch)) {\n"
    "                        return 1;\n"
    "                    }\n"
    "                    return 0;\n"
    "                };",
    "// xyz fork (D1): never mirror a media chunk into the drafter\n"
    "                static auto cb = [](llama_batch batch, void * user_data) {\n"
    "                    (void) batch;\n"
    "                    (void) user_data;\n"
    "                    return 0;\n"
    "                };",
    "D1-cb-no-op",
)

if apply_b:
    ctx.sub(
        "if (slot.prompt.tokens.has_mtmd) {\n"
        "            slot.spec_prompt = slot.prompt.tokens.get_text_tokens();",
        "if (false) {\n"
        "            slot.spec_prompt = slot.prompt.tokens.get_text_tokens();",
        "B1-spec-prompt",
    )
    ctx.sub(
        "if (spec_early_verify_ok && !slot.prompt.tokens.has_mtmd) {",
        "if (spec_early_verify_ok) {",
        "B2-early-verify",
    )
else:
    print("patch [B]: skipped (APPLY_B=0)")

ctx.save()

# ============================ common/speculative.cpp ========================
spec = Patcher("common/speculative.cpp")

# D2: continuity guard, placed AFTER stash_trim()/n_stash/stashed_pos_max are known and BEFORE the
# stash is flushed/emitted. It mirrors the emission below exactly, including the bridge's third
# precondition (pending_pos > ctx_dft's KV end): after a draft() seed pass the boundary row is
# already in the KV, the bridge does not fire, and the first emitted row is pos[beg], not pending_pos.
# A single-token batch emits no row but moves the deferred boundary to pos[end], which must be
# bridgeable later, so pos[end] is checked in that case (plain decode of a media slot is exactly that).
D2_NEW = r"""// xyz fork (D2): skip the mirror for a ubatch that cannot continue ctx_dft's KV contiguously (the
            // batches after a media chunk keep the target's M-RoPE-compressed positions: KV ends at 57 while the
            // batch starts at 73, ctx_dft's memory module rejects that -> HTTP 500). What this call would emit
            // first mirrors the code below exactly: the stashed rows, else the deferred boundary row (all three
            // bridge preconditions), else this ubatch's first row, else (single-token batch) the position the new
            // deferred boundary will have. ctx_dft's state is left untouched; the ROLLBACK block above re-syncs
            // it when a continuous batch for this sequence arrives again.
            {
                const llama_pos dft_mem_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                bool rows_increasing = true;
                for (int32_t k = beg + 1; k <= end; ++k) {
                    if (batch_in.pos[k] <= batch_in.pos[k - 1]) {
                        rows_increasing = false;
                        break;
                    }
                }
                // the first row this call needs ctx_dft (or its stash) to continue with
                llama_pos next_needed;
                if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg] &&
                        pending_pos > std::max(dft_mem_max, stashed_pos_max)) {
                    next_needed = pending_pos;
                } else if (end > beg) {
                    next_needed = batch_in.pos[beg];
                } else {
                    next_needed = batch_in.pos[end];
                }
                // the last row ctx_dft holds, or has queued in its stash (stashed_pos_max is -1 when the stash is empty).
                // Checking only the stash's FRONT is not enough: rows appended behind a stash that ends before a
                // media chunk would leave a hole inside the stash, and flushing it fails ("positions are not continuous").
                const llama_pos last_pos = std::max(dft_mem_max, stashed_pos_max);
                const bool stash_ok = n_stash == 0 || dft_mem_max < 0 || stash_rows[seq_id].front().pos == dft_mem_max + 1;
                const bool next_ok  = last_pos < 0 || next_needed == last_pos + 1;
                if (!rows_increasing || !stash_ok || !next_ok) {
                    static int n_logged = 0;
                    if (n_logged++ < 16) {
                        fprintf(stderr, "xyz-mm: drafter mirror skipped (seq %d): next row %d, ctx_dft/stash ends at %d (KV %d, stash %d rows), batch pos %d..%d\n",
                                (int) seq_id, (int) next_needed, (int) last_pos, (int) dft_mem_max, (int) n_stash,
                                (int) batch_in.pos[beg], (int) batch_in.pos[end]);
                    }
                    continue;
                }
            }

            // unless they don't fit: a prompt cut at a user message"""

spec.sub(
    "// unless they don't fit: a prompt cut at a user message",
    D2_NEW,
    "D2-continuity-guard",
)

# E: stash_flush(): do not attempt a decode that the memory module would reject (stale stash rows).
spec.sub(
    "    void stash_flush(llama_seq_id seq_id) const {\n"
    "        const int32_t n = stash_trim(seq_id);\n"
    "        if (n <= 0) {\n"
    "            return;\n"
    "        }\n",
    "    void stash_flush(llama_seq_id seq_id) const {\n"
    "        const int32_t n = stash_trim(seq_id);\n"
    "        if (n <= 0) {\n"
    "            return;\n"
    "        }\n"
    "        // xyz fork (E): stale rows that do not continue ctx_dft's KV would only fail to decode; keep them\n"
    "        // (the ROLLBACK logic drops them when a continuous batch arrives), exactly as a failed flush does.\n"
    "        {\n"
    "            const llama_pos mem_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);\n"
    "            if (mem_max >= 0 && stash_rows[seq_id].front().pos != mem_max + 1) {\n"
    "                return;\n"
    "            }\n"
    "        }\n",
    "E-stash-flush-stale-guard",
)
spec.save()
print("done")