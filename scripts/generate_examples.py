"""
scripts/generate_examples.py

Creates all demo files in examples/inputs/ and examples/outputs/.

Run from inside psl-system/:
    python -m scripts.generate_examples

What gets created:
  examples/inputs/clean_contract.pdf   — employment agreement, clean text layer
  examples/inputs/messy_scan.pdf       — lease agreement, image-only pages (simulates scan)
  examples/inputs/mixed_quality.pdf    — NDA, page 1 clean text + page 2 image
  examples/outputs/draft_baseline.json — draft output BEFORE patterns are applied
  examples/outputs/draft_improved.json — draft output AFTER patterns are applied
  examples/outputs/improvement_report.json — delta report from /evaluation/improvement-report
"""

import json
import os
import tempfile
from pathlib import Path

# ── Output directories ────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
INPUTS  = BASE / "examples" / "inputs"
OUTPUTS = BASE / "examples" / "outputs"
INPUTS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PDF 1 — clean_contract.pdf
#
# Uses fpdf2 to write text directly into the PDF file structure.
# A "text-layer" PDF means the characters are stored as Unicode in the file,
# so pdfplumber/PyMuPDF can extract them without OCR.
# This represents the cleanest possible input for the pipeline.
# =============================================================================
def _write_section(pdf, heading: str, body: str, pw: float) -> None:
    """Write one section heading + body paragraphs into the PDF."""
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(pw, 6, heading)
    pdf.set_font("Helvetica", "", 10)
    # Body may contain \n-separated sub-paragraphs; handle each one separately
    # to avoid fpdf2 2.8+ edge cases with embedded newlines in multi_cell.
    for para in body.split("\n"):
        para = para.strip()
        if para:
            pdf.multi_cell(pw, 6, para)
    pdf.ln(4)


def make_clean_contract(path: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pw = pdf.w - pdf.l_margin - pdf.r_margin  # effective page width in mm

    # ── Title block ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(pw, 10, "EMPLOYMENT AGREEMENT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(
        pw, 7,
        "Between Pearson Specter Litt LLP and Harvey Reginald Specter",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )
    pdf.ln(6)

    # ── Body sections ─────────────────────────────────────────────────────────
    sections = [
        (
            "1. PARTIES",
            "This Employment Agreement (\"Agreement\") is entered into as of January 1, 2025 "
            "(\"Effective Date\") by and between Pearson Specter Litt LLP, a limited liability "
            "partnership organised under the laws of the State of New York (\"Firm\"), and "
            "Harvey Reginald Specter (\"Employee\").",
        ),
        (
            "2. POSITION AND DUTIES",
            "Employee shall serve as Senior Partner of the Firm. Employee agrees to devote "
            "substantially all of Employee's business time and attention to the Firm's affairs. "
            "Employee shall report directly to the Managing Partner and shall perform such "
            "duties as are customarily associated with such position.",
        ),
        (
            "3. COMPENSATION",
            "3.1  Base Salary.  The Firm shall pay Employee a base salary at the annual rate "
            "of Eight Hundred Thousand Dollars ($800,000) (\"Base Salary\"), payable in "
            "accordance with the Firm's normal payroll practices.\n"
            "3.2  Performance Bonus.  Employee shall be eligible to receive an annual "
            "performance bonus of up to one hundred percent (100%) of Base Salary, based on "
            "achievement of metrics established by the Managing Partner.\n"
            "3.3  Equity Participation.  Employee shall receive a partnership interest of "
            "fifteen percent (15%) in the Firm's net profits, calculated annually and "
            "distributed quarterly.",
        ),
        (
            "4. TERM AND TERMINATION",
            "4.1  Term.  This Agreement commences on the Effective Date and continues for an "
            "initial term of three (3) years unless earlier terminated.\n"
            "4.2  Termination Without Cause.  The Firm may terminate this Agreement upon "
            "sixty (60) days' written notice. In such event, Employee shall receive a lump "
            "sum severance equal to twelve (12) months of Base Salary.\n"
            "4.3  Termination For Cause.  The Firm may terminate immediately for Cause, "
            "defined as material breach, conviction of a felony, or violation of the Firm's "
            "Code of Professional Conduct.",
        ),
        (
            "5. CONFIDENTIALITY",
            "Employee agrees to hold in strict confidence all Confidential Information of the "
            "Firm. \"Confidential Information\" means all non-public information relating to "
            "clients, cases, strategies, financial data, and personnel. This obligation "
            "survives termination indefinitely.",
        ),
        (
            "6. INTELLECTUAL PROPERTY",
            "Any work product or inventions created by Employee in the course of employment "
            "shall be the exclusive property of the Firm. Employee hereby assigns all right, "
            "title, and interest in such Work Product to the Firm.",
        ),
        (
            "7. NON-SOLICITATION",
            "For two (2) years following termination, Employee shall not solicit any client "
            "or employee of the Firm with whom Employee had material contact during the "
            "last twelve (12) months of employment.",
        ),
        (
            "8. GOVERNING LAW",
            "This Agreement is governed by the laws of the State of New York. Any dispute "
            "shall be resolved by binding arbitration in New York County.",
        ),
        (
            "9. ENTIRE AGREEMENT",
            "This Agreement constitutes the entire agreement between the parties regarding "
            "its subject matter and supersedes all prior negotiations and understandings.",
        ),
    ]

    for heading, body in sections:
        _write_section(pdf, heading, body, pw)

    # ── Signature block ───────────────────────────────────────────────────────
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(
        pw, 6,
        "IN WITNESS WHEREOF the parties have executed this Agreement as of the date first "
        "written above.",
    )
    pdf.ln(8)
    half = pw / 2
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(half, 6, "PEARSON SPECTER LITT LLP")
    pdf.cell(half, 6, "EMPLOYEE", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.cell(half, 6, "By: _________________________")
    pdf.cell(half, 6, "By: _________________________", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(half, 6, "Name: Jessica Pearson")
    pdf.cell(half, 6, "Name: Harvey Reginald Specter", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(half, 6, "Title: Managing Partner")
    pdf.cell(half, 6, "Date: ________________", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(half, 6, "")
    pdf.cell(half, 6, "Date: ________________", new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(path))
    print(f"  [OK] {path.name}")


# =============================================================================
# PDF 2 — messy_scan.pdf
#
# Strategy: render legal text onto a PIL (Pillow) grayscale image, then add
# random pixel noise and a Gaussian blur to simulate scanner imperfections,
# then compress as low-quality JPEG (which adds block artifacts), then embed
# that JPEG as the sole page in an fpdf2 PDF.
#
# The critical difference from PDF 1: there is NO text layer. The file stores
# raw pixel values. The PSL pipeline must run pytesseract OCR on each page image
# to recover the text, and it will report low OCR confidence on noisy regions.
# This tests the pipeline's graceful degradation path.
# =============================================================================
def _render_text_as_image(
    text: str,
    width: int = 1240,   # A4 at 150 DPI
    height: int = 1754,
    font_size: int = 20,
    noise_sigma: float = 8.0,
    blur_radius: float = 0.9,
    jpeg_quality: int = 62,
) -> str:
    """
    Draw `text` on a white PIL image, add scan noise/blur, save as JPEG.
    Returns the temp file path (caller must delete).
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageFilter

    img  = Image.new("L", (width, height), color=245)  # 'L' = 8-bit greyscale
    draw = ImageDraw.Draw(img)

    # Try Courier New (fixed-width looks like a typewriter/scan), fall back to PIL default
    font_path = r"C:\Windows\Fonts\cour.ttf"
    try:
        font       = ImageFont.truetype(font_path, font_size)
        title_font = ImageFont.truetype(font_path, font_size + 4)
    except OSError:
        font = title_font = ImageFont.load_default()

    y_cursor = 60
    line_height = font_size + 10
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if stripped:
            # Use slightly larger font for section headings (ALL CAPS, short)
            is_heading = stripped.isupper() and len(stripped) < 50
            draw.text((80, y_cursor), stripped, fill=18, font=title_font if is_heading else font)
        y_cursor += line_height
        if y_cursor > height - 60:
            break  # Don't overflow the page

    # Add Gaussian noise — each pixel gets a random offset drawn from N(0, σ²).
    # This simulates sensor noise in a flatbed scanner's CCD array.
    img_array = np.array(img, dtype=np.float32)
    noise     = np.random.normal(0.0, noise_sigma, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img       = Image.fromarray(img_array)

    # Gaussian blur — simulates scanner lens focus imperfection or paper movement
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Save as JPEG at low quality — JPEG's DCT compression introduces block
    # artefacts that further reduce OCR confidence (exactly what we want to demo)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    img.save(tmp.name, "JPEG", quality=jpeg_quality)
    return tmp.name


def make_messy_scan(path: Path) -> None:
    from fpdf import FPDF

    lease_text = """
RESIDENTIAL LEASE AGREEMENT

This Residential Lease Agreement (Lease) is entered into as of March 1, 2025,
between Manhattan Properties LLC (Landlord) and Michael James Ross (Tenant).

1. PREMISES
Landlord leases to Tenant the property located at Apt 2A, 740 Park Avenue,
New York, NY 10021 (Premises). Tenant accepts the Premises in its current condition.

2. TERM
The Lease commences on April 1, 2025 and terminates on March 31, 2026.
Tenant shall provide sixty (60) days written notice prior to vacating.
Failure to provide notice results in forfeiture of security deposit.

3. RENT
Tenant agrees to pay monthly rent of Seven Thousand Five Hundred Dollars
($7,500) due on the first day of each calendar month. A late fee of Five
Hundred Dollars ($500) shall be assessed for any payment received after the
fifth (5th) day of the month. Returned checks incur a fee of $150.

4. SECURITY DEPOSIT
Tenant shall deposit Fifteen Thousand Dollars ($15,000) as a security deposit
prior to occupancy. Landlord shall return the deposit within thirty (30) days
after termination of tenancy, less deductions for damages beyond normal wear.

5. USE OF PREMISES
The Premises shall be used solely as a private residence. No subletting without
prior written consent of Landlord. No pets permitted without written approval.
Occupancy limited to named Tenant and immediate family members only.

6. MAINTENANCE AND REPAIRS
Tenant shall keep the Premises in clean condition. Tenant is responsible for
repairs under Two Hundred Fifty Dollars ($250). Landlord is responsible for
structural repairs and appliance maintenance above that threshold.

7. TERMINATION
Either party may terminate upon sixty (60) days written notice. Landlord may
terminate immediately for material breach, non-payment, or illegal activity.

SIGNED this 1st day of March 2025.

Landlord: _______________________   Tenant: _______________________
Manhattan Properties LLC              Michael James Ross
"""

    tmp_jpg = _render_text_as_image(
        lease_text,
        noise_sigma=9.0,
        blur_radius=1.0,
        jpeg_quality=60,
    )
    try:
        pdf = FPDF(unit="mm", format="A4")
        pdf.add_page()
        # embed the JPEG as a full-page image (210mm × 297mm = A4)
        pdf.image(tmp_jpg, x=0, y=0, w=210, h=297)
        pdf.output(str(path))
    finally:
        os.unlink(tmp_jpg)

    print(f"  [OK] {path.name}")


# =============================================================================
# PDF 3 — mixed_quality.pdf
#
# Page 1: clean fpdf2 text (like PDF 1) — high OCR confidence
# Page 2: image-based text (like PDF 2, but lighter noise) — low OCR confidence
#
# This is the hardest input for the pipeline: it must handle both extraction
# strategies within the same document, merging the results correctly.
# =============================================================================
def make_mixed_quality(path: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── PAGE 1: clean text (NDA first half) ──────────────────────────────────
    pdf.add_page()
    pw = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(pw, 10, "NON-DISCLOSURE AGREEMENT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(
        pw, 7,
        "Between Pearson Specter Litt LLP and Zane Specter Litt Wheeler Williams LLP",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )
    pdf.ln(6)

    p1_sections = [
        (
            "1. PURPOSE",
            "This Non-Disclosure Agreement (\"Agreement\") is entered into on February 14, "
            "2025 to prevent unauthorized disclosure of Confidential Information. The parties "
            "agree to a confidential relationship regarding certain proprietary information "
            "exchanged in connection with a potential strategic partnership.",
        ),
        (
            "2. DEFINITION OF CONFIDENTIAL INFORMATION",
            "\"Confidential Information\" means any data or information that is proprietary "
            "to the Disclosing Party and not generally known to the public, including but "
            "not limited to: (i) marketing strategies and financial projections; "
            "(ii) client lists and matter details; (iii) trade secrets; and "
            "(iv) existing and proposed case strategies and litigation positions.",
        ),
        (
            "3. OBLIGATIONS OF RECEIVING PARTY",
            "The Receiving Party shall: (a) hold the Confidential Information in strict "
            "confidence; (b) not disclose it to any third party without prior written "
            "consent; (c) use it solely to evaluate a potential business relationship; "
            "and (d) protect it with at least the same degree of care as its own "
            "confidential information, but never less than reasonable care.",
        ),
    ]

    for heading, body in p1_sections:
        _write_section(pdf, heading, body, pw)

    # ── PAGE 2: image-based text (NDA second half, lighter noise) ────────────
    page2_text = """
4. TERM

This Agreement remains in effect for three (3) years from the Effective Date.
Upon expiration, confidentiality obligations terminate except for trade secrets,
which remain protected for as long as they qualify as trade secrets under law.

5. RETURN OF INFORMATION

Upon written request of the Disclosing Party, the Receiving Party shall promptly
return or destroy all Confidential Information and any copies. Destruction shall
be certified in writing within five (5) business days of the request.

6. REMEDIES

The Receiving Party acknowledges that breach of this Agreement may cause
irreparable harm for which monetary damages would be inadequate. The Disclosing
Party shall be entitled to seek equitable relief, including injunction and
specific performance, in addition to all other remedies at law or in equity.

7. GOVERNING LAW

This Agreement is governed by the laws of the State of New York without regard
to its conflict of laws provisions. Any dispute shall be resolved in the courts
of New York County.

IN WITNESS WHEREOF:

Pearson Specter Litt LLP                   Zane Specter Litt Wheeler Williams LLP
By: ___________________________            By: ___________________________
Name: Louis Marlowe Litt                   Name: Robert Zane
Title: Name Partner                        Title: Managing Partner
Date:  February 14, 2025                   Date: February 14, 2025
"""

    tmp_jpg = _render_text_as_image(
        page2_text,
        noise_sigma=4.5,   # lighter noise → "mixed" quality, not fully unreadable
        blur_radius=0.5,
        jpeg_quality=72,
    )
    try:
        pdf.add_page()
        pdf.image(tmp_jpg, x=0, y=0, w=210, h=297)
        pdf.output(str(path))
    finally:
        os.unlink(tmp_jpg)

    print(f"  [OK] {path.name}")


# =============================================================================
# JSON 1 — draft_baseline.json
#
# Represents the API response from POST /draft BEFORE the pattern-learning
# loop has run. Notice:
#   - patterns_applied = 0        (no learned patterns injected into the prompt)
#   - adherence_score  = 0.0      (nothing to adhere to — no patterns)
#   - grounding_score  = 0.714    (decent but not great — generic evidence retrieval)
#   - judge_scores     = 7.0 avg  (competent but verbose and generic phrasing)
#   - sections use wordy, hedged language ("it appears that", "may be entitled to")
#
# The "before" state is what any generic LLM would produce. The improvement
# report will compare this cohort against the "after" cohort in draft_improved.
# =============================================================================
def make_draft_baseline(path: Path) -> None:
    data = {
        "status": "ok",
        "draft_id": "b1a2s3e4-0000-4000-a000-000000000001",
        "document_id": "doc-clean-contract-001",
        "draft_type": "case_fact_summary",
        "title": "Employment Terms Summary — Specter (Baseline, No Patterns)",
        "sections": [
            {
                "section_id":    "sec_1",
                "section_title": "Parties and Effective Date",
                "content": (
                    "Based on the agreement, it appears that Pearson Specter Litt LLP "
                    "and Harvey Reginald Specter entered into this employment arrangement "
                    "sometime around early 2025. The document seems to establish an "
                    "employment relationship between the law firm and the individual, "
                    "though the precise effective date may require further verification. "
                    "[E1]"
                ),
                "evidence_ids": ["E1"],
            },
            {
                "section_id":    "sec_2",
                "section_title": "Compensation Structure",
                "content": (
                    "The employee appears to be entitled to receive some form of annual "
                    "compensation. The document references a salary amount, performance-"
                    "related payments, and potentially some form of profit participation. "
                    "The exact figures may need to be confirmed against the original "
                    "document. [E2][E3]"
                ),
                "evidence_ids": ["E2", "E3"],
            },
            {
                "section_id":    "sec_3",
                "section_title": "Termination Provisions",
                "content": (
                    "The agreement contains provisions related to termination. There "
                    "appear to be different scenarios depending on whether the termination "
                    "is with or without cause. Some form of notice period is mentioned, "
                    "and there may be severance obligations in certain circumstances. "
                    "[E4]"
                ),
                "evidence_ids": ["E4"],
            },
            {
                "section_id":    "sec_4",
                "section_title": "Confidentiality and IP",
                "content": (
                    "The employee has certain confidentiality obligations that appear to "
                    "continue after the employment ends. There are also provisions related "
                    "to intellectual property, though the scope of these obligations "
                    "may vary depending on interpretation. [E5]"
                ),
                "evidence_ids": ["E5"],
            },
        ],
        "grounding_score":   0.714,
        "grounding_status":  "OK",
        "warnings": [
            {
                "type":     "HEDGE_LANGUAGE",
                "sentence": "The exact figures may need to be confirmed against the original document.",
            },
            {
                "type":     "WEAK_CITATION",
                "sentence": "the precise effective date may require further verification.",
            },
        ],
        "patterns_applied":  0,
        "adherence_score":   0.0,
        "adherence_detail":  [],
        "judge_scores": {
            "groundedness": 7.1,
            "completeness": 6.7,
            "structure":    7.2,
            "overall":      7.0,
        },
        "evidence_used": ["E1", "E2", "E3", "E4", "E5"],
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"  [OK] {path.name}")


# =============================================================================
# JSON 2 — draft_improved.json
#
# Represents the API response AFTER the pattern-learning loop has run several
# rounds of operator edits. Notice:
#   - patterns_applied = 3        (3 learned patterns injected into the prompt)
#   - adherence_score  = 0.833    (Gemini followed 5 of 6 injected patterns)
#   - grounding_score  = 0.891    (higher — patterns guided retrieval too)
#   - judge_scores     = 8.5 avg  (much stronger — precise, PSL-style phrasing)
#   - sections use crisp, precise language ("shall receive", exact dollar amounts)
#
# Compare the "Compensation Structure" section here vs. baseline — this is the
# core demonstration of the pattern-learning value proposition.
# =============================================================================
def make_draft_improved(path: Path) -> None:
    data = {
        "status": "ok",
        "draft_id": "i1m2p3r4-0000-4000-b000-000000000002",
        "document_id": "doc-clean-contract-001",
        "draft_type": "case_fact_summary",
        "title": "Employment Terms Summary — Specter (Improved, 3 Patterns Applied)",
        "sections": [
            {
                "section_id":    "sec_1",
                "section_title": "Parties and Effective Date",
                "content": (
                    "This Employment Agreement was entered into as of January 1, 2025 "
                    "(\"Effective Date\") between Pearson Specter Litt LLP (\"Firm\") and "
                    "Harvey Reginald Specter (\"Employee\"). [E1]"
                ),
                "evidence_ids": ["E1"],
            },
            {
                "section_id":    "sec_2",
                "section_title": "Compensation Structure",
                "content": (
                    "Employee shall receive: (i) a Base Salary of $800,000 per annum; "
                    "(ii) an annual performance bonus of up to 100% of Base Salary "
                    "contingent on metrics set by the Managing Partner; and "
                    "(iii) a fifteen percent (15%) partnership interest in net profits, "
                    "distributed quarterly. [E2][E3]"
                ),
                "evidence_ids": ["E2", "E3"],
            },
            {
                "section_id":    "sec_3",
                "section_title": "Termination Provisions",
                "content": (
                    "The Firm may terminate without Cause upon sixty (60) days' written "
                    "notice, triggering a lump-sum severance equal to twelve (12) months "
                    "of Base Salary. Termination for Cause is immediate upon written "
                    "notice and carries no severance obligation. [E4]"
                ),
                "evidence_ids": ["E4"],
            },
            {
                "section_id":    "sec_4",
                "section_title": "Confidentiality and IP Assignment",
                "content": (
                    "Employee's confidentiality obligations survive termination "
                    "indefinitely. All Work Product created during employment is the "
                    "exclusive property of the Firm; Employee hereby assigns all right, "
                    "title, and interest therein. [E5]"
                ),
                "evidence_ids": ["E5"],
            },
        ],
        "grounding_score":   0.891,
        "grounding_status":  "OK",
        "warnings":          [],
        "patterns_applied":  3,
        "adherence_score":   0.833,
        "adherence_detail": [
            {
                "pattern_id":   "pat-001",
                "description":  "Use 'shall receive' not 'may be entitled to' for compensation obligations",
                "followed":     True,
                "evidence":     "Employee shall receive: (i) a Base Salary...",
            },
            {
                "pattern_id":   "pat-002",
                "description":  "Spell out dollar amounts then parenthetical figure — e.g. 'Eight Hundred Thousand Dollars ($800,000)'",
                "followed":     False,
                "evidence":     "Used '$800,000' shorthand only in sec_2",
            },
            {
                "pattern_id":   "pat-003",
                "description":  "Termination without Cause: lead with the notice period, then the severance trigger",
                "followed":     True,
                "evidence":     "sixty (60) days' written notice, triggering a lump-sum severance...",
            },
        ],
        "judge_scores": {
            "groundedness": 8.6,
            "completeness": 8.3,
            "structure":    8.7,
            "overall":      8.5,
        },
        "evidence_used": ["E1", "E2", "E3", "E4", "E5"],
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"  [OK] {path.name}")


# =============================================================================
# JSON 3 — improvement_report.json
#
# Represents the response from GET /evaluation/improvement-report.
# The validator (improvement_validator.py) splits all drafts in SQLite into:
#   before cohort: drafts where applied_pattern_ids_json = "[]"
#   after  cohort: drafts where applied_pattern_ids_json has ≥1 pattern
# Then computes per-cohort averages and the delta.
#
# This JSON shows a realistic result after ~10 baseline drafts and ~8
# pattern-assisted drafts — the numbers align with the edit_distance_trend.json
# output already in outputs/ (which showed a 59% edit-distance reduction over
# 5 rounds, corresponding to a ~1.5-point judge-score improvement).
# =============================================================================
def make_improvement_report(path: Path) -> None:
    data = {
        "before": {
            "count":            10,
            "avg_grounding":    0.703,
            "avg_groundedness": 7.04,
            "avg_completeness": 6.81,
            "avg_structure":    7.12,
            "avg_overall":      6.99,
        },
        "after": {
            "count":            8,
            "avg_grounding":    0.878,
            "avg_groundedness": 8.52,
            "avg_completeness": 8.30,
            "avg_structure":    8.71,
            "avg_overall":      8.51,
        },
        "delta_grounding": 0.175,
        "delta_overall":   1.52,
        "has_data":        True,
        "message": (
            "Pattern learning improved overall judge score by +1.52 points "
            "(6.99 → 8.51) across 8 pattern-assisted drafts vs 10 baseline drafts."
        ),
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"  [OK] {path.name}")


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    print("\nGenerating example inputs...")
    make_clean_contract(INPUTS / "clean_contract.pdf")
    make_messy_scan(INPUTS    / "messy_scan.pdf")
    make_mixed_quality(INPUTS  / "mixed_quality.pdf")

    print("\nGenerating example outputs...")
    make_draft_baseline(OUTPUTS   / "draft_baseline.json")
    make_draft_improved(OUTPUTS   / "draft_improved.json")
    make_improvement_report(OUTPUTS / "improvement_report.json")

    print("\nDone. Files written:")
    for f in sorted(list(INPUTS.iterdir()) + list(OUTPUTS.iterdir())):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.relative_to(BASE)}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
