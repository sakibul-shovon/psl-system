"""
Pearson Specter Litt — Document Intelligence UI

Run with:
    streamlit run ui/app.py

Requires the FastAPI backend running on http://localhost:8000
"""

import time
import httpx
import streamlit as st

API = "http://localhost:8000"


def api(method: str, path: str, **kwargs):
    """Make an API call. Returns (data, error)."""
    try:
        r = httpx.request(method, f"{API}{path}", timeout=120, **kwargs)
        if r.status_code >= 400:
            return None, r.json().get("detail", r.text)
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PSL Document Intelligence",
    page_icon="⚖️",
    layout="wide",
)

st.sidebar.title("⚖️ PSL Intelligence")
st.sidebar.caption("Pearson Specter Litt")
page = st.sidebar.radio(
    "Navigate",
    ["Upload", "Query", "Draft", "Feedback", "Metrics"],
    index=0,
)

# ── Shared state ───────────────────────────────────────────────────────────────

if "last_draft" not in st.session_state:
    st.session_state.last_draft = None
if "last_document_id" not in st.session_state:
    st.session_state.last_document_id = None


def fetch_documents() -> list[dict]:
    """
    Get all ingested documents from the API. Falls back to session_state if
    the API is unreachable (e.g., backend not running yet).

    Returns list of dicts with keys: document_id, title, document_type, page_count.
    """
    data, err = api("GET", "/documents")
    if data and data.get("documents"):
        return data["documents"]
    # Fallback to whatever was uploaded in this browser session
    return st.session_state.get("uploaded_docs", [])


def document_picker(label: str = "Document") -> str:
    """
    Render a selectbox of all known documents. Returns the chosen document_id,
    or empty string if none exist yet.
    """
    docs = fetch_documents()
    if not docs:
        st.info("No documents found. Upload one first on the Upload page.")
        return ""
    labels = [f"{d['title']}  ({d.get('document_type', '?')})" for d in docs]
    ids = [d["document_id"] for d in docs]
    # Default to last-used doc if it still exists, otherwise first in list
    last = st.session_state.get("last_document_id", "")
    default_idx = ids.index(last) if last in ids else 0
    choice = st.selectbox(label, labels, index=default_idx)
    return ids[labels.index(choice)]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

if page == "Upload":
    st.title("Upload Document")
    st.caption("Accepts PDF, JPG, PNG, TIFF. OCR runs automatically on scanned pages.")

    uploaded = st.file_uploader(
        "Choose a legal document",
        type=["pdf", "jpg", "jpeg", "png", "tiff", "tif"],
    )

    if uploaded and st.button("Upload & Process", type="primary"):
        with st.spinner("Uploading..."):
            data, err = api(
                "POST", "/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            )

        if err:
            st.error(f"Upload failed: {err}")
        else:
            job_id = data["job_id"]
            document_id = data["document_id"]
            st.success(f"Uploaded! Job ID: `{job_id}`")

            # Save for later pages
            docs = st.session_state.get("uploaded_docs", [])
            docs.append({"document_id": document_id, "title": uploaded.name})
            st.session_state.uploaded_docs = docs
            st.session_state.last_document_id = document_id

            # Poll job status
            progress_bar = st.progress(0, text="Starting pipeline...")
            status_box = st.empty()

            for _ in range(60):
                time.sleep(2)
                job, jerr = api("GET", f"/job/{job_id}")
                if jerr or not job:
                    break

                stage = job.get("stage", "processing")
                progress = job.get("progress", 0)
                status = job.get("status")

                progress_bar.progress(
                    min(progress / 100, 1.0),
                    text=f"Stage: {stage} ({progress}%)",
                )
                status_box.info(f"Status: **{status}** | Stage: {stage}")

                if status == "done":
                    progress_bar.progress(1.0, text="Complete!")
                    result = job.get("result", {})
                    st.success("Processing complete!")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Chunks extracted", result.get("chunk_count", "?"))
                    col2.metric("Document type", result.get("document_type", "?"))
                    col3.metric("Pages", result.get("page_count", "?"))
                    st.info(f"**Document ID:** `{document_id}`  \nUse this in Query and Draft pages.")
                    break
                elif status == "failed":
                    st.error(f"Pipeline failed: {job.get('error', 'unknown error')}")
                    break
            else:
                st.warning("Timed out waiting for pipeline. Check server logs.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — QUERY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Query":
    st.title("Query Evidence")
    st.caption("Hybrid retrieval: BM25 + dense vectors + cross-encoder rerank → top-5 evidence")

    doc_id = document_picker("Document")
    query = st.text_input("Query", placeholder="What are the termination and compensation terms?")

    if st.button("Search", type="primary") and doc_id and query:
        with st.spinner("Retrieving evidence..."):
            data, err = api("POST", "/query", json={"document_id": doc_id, "query": query})

        if err:
            st.error(f"Query failed: {err}")
        elif data and data.get("status") == "INSUFFICIENT_EVIDENCE":
            st.warning(f"Insufficient evidence: {data.get('diagnostic')}")
        elif data:
            st.success(f"Retrieved via **{data.get('retrieval_method', '?')}**")
            for ev in data.get("evidence", []):
                with st.expander(
                    f"**[{ev['evidence_id']}]** {ev.get('breadcrumb', '')} — score: {ev.get('rerank_score', 0):.3f}",
                    expanded=True,
                ):
                    st.caption(f"Source: {ev.get('source_title', '')} | Confidence: {ev.get('confidence_tier', '')}")
                    st.write(ev.get("content", ""))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DRAFT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Draft":
    st.title("Generate Draft")
    st.caption("Retrieve evidence → inject learned patterns → Gemini generates → NLI grounding check → judge score")

    doc_id = document_picker("Document")
    query = st.text_input(
        "Query",
        placeholder="Summarize the compensation and termination terms",
    )
    draft_type = st.selectbox("Draft type", ["case_fact_summary"])

    if st.button("Generate Draft", type="primary") and doc_id and query:
        with st.spinner("Generating (this takes 10–20 seconds)..."):
            data, err = api(
                "POST", "/draft",
                json={"document_id": doc_id, "query": query, "draft_type": draft_type},
            )

        if err:
            st.error(f"Generation failed: {err}")
        elif data and data.get("status") == "INSUFFICIENT_EVIDENCE":
            st.warning(f"Insufficient evidence: {data.get('diagnostic')}")
        elif data and data.get("status") == "INSUFFICIENT_GROUNDING":
            st.error(
                f"Draft refused — grounding score {data.get('grounding_score', 0):.2f} "
                f"below threshold. {data.get('diagnostic', '')}"
            )
        elif data:
            st.session_state.last_draft = data

            # ── Header metrics ──
            col1, col2, col3, col4 = st.columns(4)
            gs = data.get("grounding_score", 0)
            gs_color = "🟢" if gs >= 0.75 else "🟡" if gs >= 0.50 else "🔴"
            col1.metric("Grounding", f"{gs_color} {gs:.2f}")
            col2.metric("Patterns applied", data.get("patterns_applied", 0))
            col3.metric("Adherence", f"{data.get('adherence_score', 1.0):.2f}")

            js = data.get("judge_scores", {})
            col4.metric("Judge overall", f"{js.get('overall', '?')} / 5")

            st.divider()

            # ── Judge scorecard ──
            with st.expander("Judge Scorecard (Groq 70B independent evaluation)"):
                jcol1, jcol2, jcol3, jcol4 = st.columns(4)
                jcol1.metric("Groundedness", f"{js.get('groundedness', '?')} / 5")
                jcol2.metric("Completeness", f"{js.get('completeness', '?')} / 5")
                jcol3.metric("Structure", f"{js.get('structure', '?')} / 5")
                jcol4.metric("Overall", f"{js.get('overall', '?')} / 5")
                if js.get("reasoning"):
                    st.caption(f"Reasoning: {js['reasoning']}")

            # ── Draft sections ──
            st.subheader(data.get("title", "Draft"))
            for section in data.get("sections", []):
                st.markdown(f"#### {section.get('title', '')}")
                st.write(section.get("content", ""))
                cited = section.get("citedEvidence", [])
                if cited:
                    st.caption(f"Evidence cited: {', '.join(cited)}")
                st.divider()

            # ── Warnings ──
            warnings = data.get("warnings", [])
            if warnings:
                with st.expander(f"{len(warnings)} grounding warning(s)"):
                    for w in warnings:
                        st.warning(f"**{w['type']}**: {w['sentence'][:120]}")

            # ── Adherence detail ──
            adherence = data.get("adherence_detail", [])
            if adherence:
                with st.expander("Pattern adherence detail"):
                    for a in adherence:
                        icon = "✅" if a["result"] == "FOLLOWED" else "❌"
                        st.write(f"{icon} **{a['result']}** — {a['description']}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Feedback":
    st.title("Submit Operator Edits")
    st.caption("Edit the draft sections below. Submit to teach the system your style preferences.")

    draft = st.session_state.get("last_draft")
    if not draft:
        st.info("Generate a draft first on the Draft page.")
    else:
        st.success(f"Editing draft: `{draft['draft_id']}`")
        st.caption(f"Document: `{draft['document_id']}`")

        sections = draft.get("sections", [])
        edits = []

        for section in sections:
            st.markdown(f"#### {section.get('title', '')}")
            col_orig, col_edit = st.columns(2)

            with col_orig:
                st.caption("Original (generated)")
                st.text_area(
                    f"original_{section['sectionId']}",
                    value=section.get("content", ""),
                    height=200,
                    disabled=True,
                    label_visibility="collapsed",
                )

            with col_edit:
                st.caption("Your edit")
                edited = st.text_area(
                    f"edit_{section['sectionId']}",
                    value=section.get("content", ""),
                    height=200,
                    label_visibility="collapsed",
                )

            edits.append({
                "section_id": section["sectionId"],
                "section_title": section.get("title", ""),
                "original_text": section.get("content", ""),
                "edited_text": edited,
            })
            st.divider()

        if st.button("Submit Edits", type="primary"):
            changed = [e for e in edits if e["original_text"] != e["edited_text"]]
            if not changed:
                st.warning("No changes detected. Edit at least one section before submitting.")
            else:
                with st.spinner(f"Submitting {len(changed)} edit(s)..."):
                    data, err = api(
                        "POST", "/feedback",
                        json={"draft_id": draft["draft_id"], "edits": changed},
                    )

                if err:
                    st.error(f"Feedback failed: {err}")
                else:
                    st.success(
                        f"Accepted {data.get('edits_stored', 0)} edit(s). "
                        f"Pattern extraction running in background."
                    )
                    st.info("Check the Metrics page in ~10 seconds to see newly learned patterns.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — METRICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Metrics":
    st.title("System Metrics")

    col_refresh = st.columns([1, 4])[0]
    if col_refresh.button("Refresh"):
        st.rerun()

    # ── Counts ────────────────────────────────────────────────────────────────
    metrics, err = api("GET", "/metrics")
    if err:
        st.error(f"Could not load metrics: {err}")
    else:
        counts = metrics.get("counts", {})
        quality = metrics.get("quality", {})

        st.subheader("System Counts")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Documents", counts.get("documents", 0))
        c2.metric("Chunks", counts.get("chunks", 0))
        c3.metric("Drafts", counts.get("drafts", 0))
        c4.metric("Edits submitted", counts.get("edits_submitted", 0))
        c5.metric("Active patterns", counts.get("patterns_active", 0))

        st.subheader("Quality Scores")
        q1, q2 = st.columns(2)
        q1.metric(
            "Avg grounding score",
            f"{quality.get('avg_grounding_score', 0):.2f}" if quality.get("avg_grounding_score") else "—",
        )
        q2.metric(
            "Avg judge overall",
            f"{quality.get('avg_judge_overall', 0):.1f} / 5" if quality.get("avg_judge_overall") else "—",
        )

    st.divider()

    # ── Improvement report ────────────────────────────────────────────────────
    st.subheader("Improvement Report")
    report, err2 = api("GET", "/evaluation/improvement-report")
    if err2:
        st.error(f"Could not load improvement report: {err2}")
    elif report:
        st.caption(report.get("message", ""))

        if report.get("has_data"):
            b = report["before_patterns"]
            a = report["after_patterns"]
            delta = report["delta"]

            col_b, col_a, col_d = st.columns(3)
            with col_b:
                st.markdown("**Before patterns**")
                st.metric("Drafts", b["draft_count"])
                st.metric("Avg grounding", f"{b['avg_grounding_score']:.2f}")
                st.metric("Judge overall", f"{b['avg_judge_scores']['overall']:.1f}")
            with col_a:
                st.markdown("**After patterns**")
                st.metric("Drafts", a["draft_count"])
                st.metric("Avg grounding", f"{a['avg_grounding_score']:.2f}")
                st.metric("Judge overall", f"{a['avg_judge_scores']['overall']:.1f}")
            with col_d:
                st.markdown("**Delta (after − before)**")
                st.metric("Grounding", f"{delta['grounding_score']:+.3f}")
                st.metric("Overall", f"{delta['overall_judge_score']:+.2f}")
        else:
            st.info(report.get("message"))

    st.divider()

    # ── Active patterns ───────────────────────────────────────────────────────
    st.subheader("Learned Patterns")
    patterns_data, perr = api("GET", "/patterns")
    if perr:
        st.error(f"Could not load patterns: {perr}")
    elif patterns_data:
        plist = patterns_data.get("patterns", [])
        if not plist:
            st.info("No patterns learned yet. Submit edits on the Feedback page.")
        else:
            for p in plist:
                with st.expander(
                    f"**[{p['rule_type'].upper()}]** {p['description'][:80]}  "
                    f"(conf: {p['confidence']:.2f})",
                ):
                    col_l, col_r = st.columns(2)
                    col_l.markdown("**Before (wrong)**")
                    col_l.code(p.get("few_shot_before", ""), language=None)
                    col_r.markdown("**After (correct)**")
                    col_r.code(p.get("few_shot_after", ""), language=None)
                    st.caption(
                        f"Frequency: {p['frequency']} | "
                        f"Created: {p['created_at'][:10]}"
                    )
