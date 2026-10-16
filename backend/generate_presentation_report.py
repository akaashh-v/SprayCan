"""Generate High-Quality Technical Presentation Report PDF.

Produces an executive & technical presentation briefing in PDF format
covering acoustic physics, two-stage architecture, definitive decision criteria,
acoustic discrimination matrix, empirical benchmarks, and shadow mode governance.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Palette
PRIMARY = colors.HexColor("#0F172A")       # Slate 900
PRIMARY_LIGHT = colors.HexColor("#1E293B") # Slate 800
SECONDARY = colors.HexColor("#0284C7")     # Sky 600
ACCENT_GREEN = colors.HexColor("#059669")  # Emerald 600
TEXT_MAIN = colors.HexColor("#1E293B")     # Slate 800
TEXT_MUTED = colors.HexColor("#475569")    # Slate 600
BG_LIGHT = colors.HexColor("#F8FAFC")      # Slate 50
BORDER_COLOR = colors.HexColor("#E2E8F0")  # Slate 200
ALERT_RED = colors.HexColor("#DC2626")     # Red 600


class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total page count dynamically and prints running header & footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(TEXT_MUTED)
        page_num = self._pageNumber

        # Running Header (pages 2+)
        if page_num > 1:
            self.drawString(0.6 * inch, 10.45 * inch, "Acoustic Aerosol Spray-Can Shaking Detector — Technical Architecture & Criteria Report")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(0.6 * inch, 10.35 * inch, 7.9 * inch, 10.35 * inch)

        # Running Footer (all pages)
        footer_y = 0.45 * inch
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(0.6 * inch, footer_y + 0.12 * inch, 7.9 * inch, footer_y + 0.12 * inch)

        self.drawString(0.6 * inch, footer_y, "Presentation & Technical Briefing | Two-Stage Precision Acoustic Event Detection")
        page_str = f"Page {page_num} of {total_pages}"
        self.drawRightString(7.9 * inch, footer_y, page_str)
        self.restoreState()


def build_pdf(filename: str = "Acoustic_Spray_Can_Detector_Presentation_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=PRIMARY,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceAfter=12,
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=TEXT_MUTED,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=TEXT_MAIN,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=TEXT_MAIN,
        leftIndent=14,
        spaceAfter=4,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=TEXT_MAIN,
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=PRIMARY,
    )
    table_cell_header = ParagraphStyle(
        "TableCellHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.white,
    )

    story = []

    # Title Block
    story.append(Paragraph("Acoustic Spray-Can Shaking Detector", title_style))
    story.append(Paragraph("Comprehensive Technical Architecture, Operating Logic & Decision Criteria Presentation Report", subtitle_style))

    # Meta banner table
    meta_data = [
        [
            Paragraph("<b>System Architecture:</b> Two-Stage Precision AED", meta_style),
            Paragraph("<b>Target Acoustic Event:</b> Aerosol Paint Can Shaking", meta_style),
            Paragraph("<b>Evaluation Mode:</b> Shadow Mode (Advisory Only)", meta_style),
        ],
        [
            Paragraph("<b>Neural Classifier:</b> 4-Block Deep 2D-CNN (SHA-256 Certified)", meta_style),
            Paragraph("<b>Operating Threshold:</b> Locked Conservative (τ = 0.80)", meta_style),
            Paragraph("<b>Continuous Processing:</b> 0.0s Cooldown Lockout (Real-Time)", meta_style),
        ],
    ]
    t_meta = Table(meta_data, colWidths=[2.3 * inch, 2.7 * inch, 2.3 * inch])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # Section 1: Executive Overview & The Problem Statement
    story.append(Paragraph("1. Executive Overview & Problem Formulation", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph(
        "Detecting aerosol paint can preparation in high-liability environments (such as railway transit platforms, pedestrian tunnels, and public infrastructure) represents a significant acoustic challenge. Traditional acoustic monitoring systems that rely solely on overall sound level (loudness in decibels) or fixed bandpass filters fail completely in practical deployment. In realistic acoustic settings, ambient environments are inundated with transient clatter, dropped items, keys, coins, and high-frequency noise that easily trick simple detectors.",
        body_style
    ))
    story.append(Paragraph("<b>Why Simpler Detectors Fail:</b>", h2_style))
    story.append(Paragraph("• <b>Loudness / Decibel Triggers:</b> Door slams, foot stomps, train track clatter, and passing diesel locomotives exceed 85 to 95 dBA, creating constant false alarms if volume alone is monitored.", bullet_style))
    story.append(Paragraph("• <b>Single-Frequency Energy Filters:</b> Keychains, coins jingling in pockets, and plastic pill bottles produce energetic high-frequency spikes in the 2 kHz to 8 kHz region that mimic metal impacts.", bullet_style))
    story.append(Paragraph("• <b>Speech Sibilants:</b> Natural speech conversation, public address broadcasts, and YouTube audio produce fricative and sibilant speech sounds ('s', 'sh', 'ch', 'th') that occupy identical spectral bands to rattling metal and hissing nozzles.", bullet_style))
    story.append(Paragraph("• <b>Aerosol Spray Discharge:</b> When spray paint is actively discharged, the nozzle produces turbulent fluid hiss. While visually associated with graffiti, spray hiss lacks the discrete reciprocal collision signature of canister shaking.", bullet_style))
    story.append(Paragraph(
        "<b>Our Solution:</b> To eliminate false alarms without compromising sub-second detection promptness, our system implements a hierarchical, two-stage detection pipeline: a physical kinematic pre-filter (Stage 1) followed by a deep 2D convolutional neural network (Stage 2) evaluating high-resolution time-frequency spectrograms under strict conservative gating rules.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # Section 2: Acoustic Physics & Core Operating Logic
    story.append(Paragraph("2. Acoustic Physics: What Makes Spray-Can Shaking Unique?", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph(
        "The sound of an aerosol spray can is not a continuous tone or random white noise. It is a strictly structured physical collision process governed by canister internal geometry and human arm biomechanics. The detector leverages four fundamental physical phenomena to confirm authenticity:",
        body_style
    ))

    physics_table_data = [
        [
            Paragraph("Physical Phenomenon", table_cell_header),
            Paragraph("Acoustic / Kinematic Mechanism", table_cell_header),
            Paragraph("Engineering Discrimination Criterion", table_cell_header),
        ],
        [
            Paragraph("<b>Internal Pea Collisions</b>", table_cell_bold),
            Paragraph("Inside every spray can is a heavy mixing ball (steel, glass, or ceramic). When accelerated along the canister axis, the pea impacts the hemispherical bottom dome or top shoulder, exciting structural shell resonance modes between 1.8 kHz and 8.5 kHz.", table_cell),
            Paragraph("Isolates metallic modal ringing via a 2nd-order bandpass filter. Requires energy in the canister modal band (2.2–7.5 kHz) to dominate mid-low frequencies (600–1800 Hz) by a factor ≥ 0.30x.", table_cell),
        ],
        [
            Paragraph("<b>Biomechanical Arm Cadence</b>", table_cell_bold),
            Paragraph("A person vigorously shaking an aerosol can moves their arm in a reciprocal back-and-forth oscillation. Human arm musculotendon mechanics enforce strict physical speed boundaries.", table_cell),
            Paragraph("Enforces a bounded strike cadence of <b>2.5 Hz to 15.0 Hz</b> with at least 3 discrete impact peaks spanning ≥ 0.35 seconds. Rejects single claps, isolated drops, or pneumatic equipment (>20 Hz).", table_cell),
        ],
        [
            Paragraph("<b>Transit Silent Valleys</b>", table_cell_bold),
            Paragraph("Between the forward stroke and the backward stroke, the pea travels in free flight through liquid paint. During this flight phase, the metal shell ceases ringing, creating deep quiet drops between strikes.", table_cell),
            Paragraph("Enforces an <b>Envelope Valley Contrast ≥ 4.2 dB</b> between peak impacts. Completely rejects flat continuous sounds (spray hiss, rainfall, fans, HVAC drone) that lack deep transit valleys.", table_cell),
        ],
        [
            Paragraph("<b>Rhythm Regularity (Pacing)</b>", table_cell_bold),
            Paragraph("Deliberate human reciprocal arm motion produces structured, periodic strike intervals, unlike chaotic natural phenomena.", table_cell),
            Paragraph("Enforces an <b>Interval Coefficient of Variation (CV) ≤ 0.65</b>. Rejects stochastic Poisson-distributed arrivals such as raindrops falling on roofs or sporadic water dripping.", table_cell),
        ],
    ]
    t_physics = Table(physics_table_data, colWidths=[1.8 * inch, 3.2 * inch, 2.3 * inch])
    t_physics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_LIGHT),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_physics)
    story.append(Spacer(1, 10))

    # Section 3: System Architecture & End-to-End Pipeline
    story.append(Paragraph("3. System Architecture & Processing Pipeline", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph(
        "The system operates on continuous streaming microphone audio sampled at 22,050 Hz using a rolling 1.0-second analysis window advanced every 250 milliseconds (4 evaluations per second). The end-to-end pipeline consists of four tightly coupled components:",
        body_style
    ))

    arch_steps = [
        [
            Paragraph("1. Front-End DSP Noise Canceller", table_cell_bold),
            Paragraph("Applies real-time adaptive spectral subtraction to suppress stationary background noise (HVAC blowers, station fans, rainfall bed). Continuously tracks background noise quantiles during quiet intervals. Automatically freezes background adaptation the instant candidate shaking begins to prevent canceling out genuine signals.", table_cell),
        ],
        [
            Paragraph("2. Stage 1: Kinematic Pre-Filter", table_cell_bold),
            Paragraph("A fast, zero-latency mathematical gate evaluating time-domain amplitude envelope and frequency ratios. Confirms that the audio has sufficient energy (RMS ≥ 0.006), is not digitally clipped (< 1.05 peak), possesses 2.5–15.0 Hz impact cadence, has ≥ 4.2 dB valley contrast, and exhibits metallic modal resonance. Non-candidates are rejected immediately without invoking the neural network.", table_cell),
        ],
        [
            Paragraph("3. Stage 2: Deep 2D-CNN Classifier", table_cell_bold),
            Paragraph("Computes a 64-band Log-Mel Spectrogram across 44 temporal frames (1024-point STFT, 512-point hop). Processes the 2D time-frequency image through a 4-block deep convolutional neural network equipped with Batch Normalization, Max Pooling, and Dropout regularization. Generates calibrated Softmax probabilities across 6 distinct acoustic classes.", table_cell),
        ],
        [
            Paragraph("4. Multi-Gated Decision Engine", table_cell_bold),
            Paragraph("Evaluates CNN class probabilities against 5 simultaneous conservative constraints: Target probability must meet threshold (P ≥ 0.80); all confusable physical mimics (spray hiss, rattles, impacts) must remain strictly low (≤ 0.20); target margin over mimics must exceed 0.35; target must dominate ambient; and normalized Shannon entropy must be ≤ 0.70.", table_cell),
        ],
        [
            Paragraph("5. Temporal Persistence & Alert Engine", table_cell_bold),
            Paragraph("Requires 2 consecutive rolling positive windows to confirm an alert episode (confirming sustained motion within ≤ 0.50 seconds). Features zero cooldown (0.0s lockout), continuous evaluation, and verified quiet-interval rearming (2 quiet windows) to allow immediate detection of subsequent shakes.", table_cell),
        ],
    ]
    t_arch = Table(arch_steps, colWidths=[2.2 * inch, 5.1 * inch])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), BG_LIGHT),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 10))

    # Section 4: Exact Decision Criteria Checklist
    story.append(Paragraph("4. Definitive Decision Criteria Checklist", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph(
        "To declare that an acoustic event is genuine aerosol spray-can shaking, the audio window must satisfy every criterion in the checklist below. Failure of any single criterion triggers an immediate rejection.",
        body_style
    ))

    criteria_data = [
        [
            Paragraph("Gating Stage", table_cell_header),
            Paragraph("Criterion / Rule", table_cell_header),
            Paragraph("Threshold Value", table_cell_header),
            Paragraph("Operational Purpose", table_cell_header),
        ],
        [
            Paragraph("Stage 1 (Energy)", table_cell_bold),
            Paragraph("RMS Energy Floor", table_cell),
            Paragraph("RMS ≥ 0.0060", table_cell_bold),
            Paragraph("Rejects ambient room tone, whisper, and background silence.", table_cell),
        ],
        [
            Paragraph("Stage 1 (Audio Integrity)", table_cell_bold),
            Paragraph("Digital Clipping Check", table_cell),
            Paragraph("Peak < 1.05 & Clip Ratio < 2%", table_cell_bold),
            Paragraph("Prevents distorted / blown-out microphones from generating false alerts.", table_cell),
        ],
        [
            Paragraph("Stage 1 (Kinematics)", table_cell_bold),
            Paragraph("Impact Burst Cadence", table_cell),
            Paragraph("2.5 Hz ≤ rate ≤ 15.0 Hz (≥ 3 peaks)", table_cell_bold),
            Paragraph("Enforces human hand shaking physical speed boundaries.", table_cell),
        ],
        [
            Paragraph("Stage 1 (Duration)", table_cell_bold),
            Paragraph("Temporal Span", table_cell),
            Paragraph("Span ≥ 0.35 seconds", table_cell_bold),
            Paragraph("Rules out single impacts, dropped tools, and footsteps.", table_cell),
        ],
        [
            Paragraph("Stage 1 (Envelope)", table_cell_bold),
            Paragraph("Valley Contrast", table_cell),
            Paragraph("Contrast ≥ 4.2 dB", table_cell_bold),
            Paragraph("Rejects flat continuous hiss, air drafts, fans, and heavy rainfall.", table_cell),
        ],
        [
            Paragraph("Stage 1 (Regularity)", table_cell_bold),
            Paragraph("Interval Pacing Variation", table_cell),
            Paragraph("CV ≤ 0.65", table_cell_bold),
            Paragraph("Rejects random stochastic arrival times (Poisson rain, water drops).", table_cell),
        ],
        [
            Paragraph("Stage 1 (Spectrum)", table_cell_bold),
            Paragraph("Canister Spectral Ratio", table_cell),
            Paragraph("E(2.2–7.5k) / E(0.6–1.8k) ≥ 0.30x", table_cell_bold),
            Paragraph("Verifies high-frequency metallic dome ring over low plastic/box thuds.", table_cell),
        ],
        [
            Paragraph("Stage 2 (Target Conf)", table_cell_bold),
            Paragraph("CNN Spray Probability", table_cell),
            Paragraph("P(spray_can_shaking) ≥ 0.80", table_cell_bold),
            Paragraph("Demands high neural network confidence on time-frequency representation.", table_cell),
        ],
        [
            Paragraph("Stage 2 (Mimic Ceiling)", table_cell_bold),
            Paragraph("Physical Mimic Ceiling", table_cell),
            Paragraph("Max(P_hiss, P_rattle, P_impact) ≤ 0.20", table_cell_bold),
            Paragraph("Keys, coins, and spray hiss probabilities must remain strictly suppressed.", table_cell),
        ],
        [
            Paragraph("Stage 2 (Margin)", table_cell_bold),
            Paragraph("Confidence Margin", table_cell),
            Paragraph("P_spray - Max(P_mimics) ≥ 0.35", table_cell_bold),
            Paragraph("Guarantees clean decision separation from confusable categories.", table_cell),
        ],
        [
            Paragraph("Stage 2 (Entropy)", table_cell_bold),
            Paragraph("Shannon Uncertainty Entropy", table_cell),
            Paragraph("H_norm ≤ 0.70", table_cell_bold),
            Paragraph("Rejects ambiguous, smeared, or multi-source acoustic confusion.", table_cell),
        ],
        [
            Paragraph("Temporal Filter", table_cell_bold),
            Paragraph("Multi-Window Persistence", table_cell),
            Paragraph("2 Consecutive Windows (500 ms)", table_cell_bold),
            Paragraph("Confirms sustained motion; rejects random transient spikes.", table_cell),
        ],
        [
            Paragraph("State Machine", table_cell_bold),
            Paragraph("Event-End Rearm Gate", table_cell),
            Paragraph("2 Quiet Windows (500 ms quiet)", table_cell_bold),
            Paragraph("Allows continuous detection and immediate capture of subsequent shakes.", table_cell),
        ],
    ]
    t_crit = Table(criteria_data, colWidths=[1.5 * inch, 1.8 * inch, 1.8 * inch, 2.2 * inch])
    t_crit.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_LIGHT),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_crit)
    story.append(Spacer(1, 10))

    # Section 5: Acoustic Discrimination Matrix
    story.append(Paragraph("5. Acoustic Discrimination Matrix", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph(
        "How the two-stage architecture systematically rejects common high-frequency and rhythmic acoustic lookalikes:",
        body_style
    ))

    matrix_data = [
        [
            Paragraph("Sound Category", table_cell_header),
            Paragraph("Acoustic Hazard Profile", table_cell_header),
            Paragraph("Rejection Mechanism & Stage", table_cell_header),
            Paragraph("System Verdict", table_cell_header),
        ],
        [
            Paragraph("<b>Conversational Speech & Video Audio</b>", table_cell_bold),
            Paragraph("Vocal sibilants ('s', 'sh', 'ch', 'th') and fricatives produce wideband noise spikes in the 3 kHz to 7 kHz band, confusing simple high-frequency triggers.", table_cell),
            Paragraph("Stage 1 filters low-frequency vowel formants (< 1.8 kHz); Stage 2 CNN categorizes speech phonemes into 'ambient_and_speech' (P_spray drops to 0.06–0.20).", table_cell),
            Paragraph("<font color='#059669'><b>CLEAN REJECTION</b><br/>(0 Alerts Fired)</font>", table_cell),
        ],
        [
            Paragraph("<b>Keys & Pocket Coins</b>", table_cell_bold),
            Paragraph("High-frequency metallic clinking with fast rise times and ringing overtones.", table_cell),
            Paragraph("Random, chaotic collision timing produces interval CV > 0.65 (fails Stage 1); Stage 2 CNN recognizes multi-metal resonant chimes as 'hard_negative_rattle'.", table_cell),
            Paragraph("<font color='#059669'><b>CLEAN REJECTION</b><br/>(Rattle Suppressed)</font>", table_cell),
        ],
        [
            Paragraph("<b>Pill Bottles & Tic-Tacs</b>", table_cell_bold),
            Paragraph("Multiple small pellets or tablets rattling rhythmically inside a plastic enclosure.", table_cell),
            Paragraph("Plastic container impacts lack high-frequency metallic shell resonance ratio (ratio < 0.30x); classified by Stage 2 CNN as 'hard_negative_rattle'.", table_cell),
            Paragraph("<font color='#059669'><b>CLEAN REJECTION</b><br/>(Rattle Suppressed)</font>", table_cell),
        ],
        [
            Paragraph("<b>Aerosol Spray Hiss</b>", table_cell_bold),
            Paragraph("Continuous fluid discharge noise produced by depressing the paint nozzle.", table_cell),
            Paragraph("Completely lacks discrete impact peaks and exhibits 0 dB valley contrast (fails Stage 1); Stage 2 CNN isolates it as 'aerosol_spray_hiss' with 100% precision.", table_cell),
            Paragraph("<font color='#059669'><b>CLEAN REJECTION</b><br/>(100% Rejection)</font>", table_cell),
        ],
        [
            Paragraph("<b>Loud Impacts & Door Slams</b>", table_cell_bold),
            Paragraph("Sudden high-energy acoustic impulse exceeding 85 dBA.", table_cell),
            Paragraph("Single event fails cadence (bursts < 3) and temporal span (< 0.35s). Low-frequency shockwaves categorized by Stage 2 as 'loud_impact'.", table_cell),
            Paragraph("<font color='#059669'><b>CLEAN REJECTION</b><br/>(Impact Suppressed)</font>", table_cell),
        ],
        [
            Paragraph("<b>Rain & Dripping Water</b>", table_cell_bold),
            Paragraph("Broadband noise with thousands of random droplet impacts on hard surfaces.", table_cell),
            Paragraph("Poisson arrival distribution fails pacing regularity (CV >> 0.65) and lacks periodic valley drops; classified as continuous environmental sound.", table_cell),
            Paragraph("<font color='#059669'><b>CLEAN REJECTION</b><br/>(Environmental)</font>", table_cell),
        ],
        [
            Paragraph("<b>Authentic Spray Can Shaking</b>", table_cell_bold),
            Paragraph("Reciprocal human arm stroke, internal pea impact against hemispherical dome, and resonant shell ringing.", table_cell),
            Paragraph("Passes all 7 Stage 1 kinematic criteria simultaneously. Stage 2 CNN achieves P(spray) ≥ 0.98, confirmed in ≤ 0.50 seconds across 2 consecutive windows.", table_cell),
            Paragraph("<font color='#DC2626'><b>CONFIRMED ALERT</b><br/>(Advisory Prompt)</font>", table_cell),
        ],
    ]
    t_matrix = Table(matrix_data, colWidths=[1.6 * inch, 2.0 * inch, 2.4 * inch, 1.3 * inch])
    t_matrix.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_LIGHT),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_matrix)
    story.append(Spacer(1, 10))

    # Section 6: Empirical Validation & Performance Benchmarks
    story.append(Paragraph("6. Empirical Benchmarks & Validation Results", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph(
        "The model and decision engine were trained, tuned, and evaluated on an extensive multi-class dataset encompassing 7,399 audio samples partitioned with strict session and site isolation to prevent data leakage:",
        body_style
    ))

    bench_data = [
        [
            Paragraph("Dataset Split", table_cell_header),
            Paragraph("Sample Count", table_cell_header),
            Paragraph("Target Spray Samples", table_cell_header),
            Paragraph("Negative & Mimic Samples", table_cell_header),
            Paragraph("Key Performance Metric", table_cell_header),
        ],
        [
            Paragraph("<b>Training Partition</b>", table_cell_bold),
            Paragraph("5,224 windows", table_cell),
            Paragraph("2,504 windows", table_cell),
            Paragraph("2,720 windows", table_cell),
            Paragraph("Cross-entropy loss optimized across 6 classes", table_cell),
        ],
        [
            Paragraph("<b>Validation Partition</b>", table_cell_bold),
            Paragraph("523 windows", table_cell),
            Paragraph("250 windows", table_cell),
            Paragraph("273 windows", table_cell),
            Paragraph("<b>Validation F1: 0.9377</b> (Peak Epoch 3)", table_cell),
        ],
        [
            Paragraph("<b>Held-Out Test Partition</b>", table_cell_bold),
            Paragraph("1,652 windows", table_cell),
            Paragraph("745 windows", table_cell),
            Paragraph("907 windows", table_cell),
            Paragraph("<b>Precision: 99.17% | FPR: 0.24%</b> (4 FPs / 1,652)", table_cell),
        ],
        [
            Paragraph("<b>Spray Hiss Benchmark</b>", table_cell_bold),
            Paragraph("48 clips", table_cell),
            Paragraph("0 targets", table_cell),
            Paragraph("48 spray hiss clips", table_cell),
            Paragraph("<b>100.0% Rejection</b> (0 false alerts fired)", table_cell),
        ],
        [
            Paragraph("<b>Real Speech / Video</b>", table_cell_bold),
            Paragraph("23 recordings", table_cell),
            Paragraph("0 targets", table_cell),
            Paragraph("23 voicemail / speech clips", table_cell),
            Paragraph("<b>100.0% Rejection</b> (0 false alerts fired)", table_cell),
        ],
        [
            Paragraph("<b>Detection Response Time</b>", table_cell_bold),
            Paragraph("Continuous Stream", table_cell),
            Paragraph("Real spray shaking", table_cell),
            Paragraph("Baseline silence", table_cell),
            Paragraph("<b>≤ 0.50 seconds</b> prompt confirmation", table_cell),
        ],
    ]
    t_bench = Table(bench_data, colWidths=[1.6 * inch, 1.2 * inch, 1.4 * inch, 1.5 * inch, 1.6 * inch])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_LIGHT),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 10))

    # Section 7: Governance, Shadow Mode & Privacy
    story.append(Paragraph("7. Governance, Shadow Mode & Privacy Compliance", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph(
        "To satisfy legal, municipal, and transit authority operational standards, the system incorporates strict privacy protections and operational safeguards directly into the software architecture:",
        body_style
    ))
    story.append(Paragraph("• <b>Advisory Shadow Mode Only:</b> The system operates exclusively in advisory shadow mode. Detection alerts function solely as advisory notifications for human operators. Automated physical enforcement, gate locking, or direct authority dispatch are strictly prohibited in software logic.", bullet_style))
    story.append(Paragraph("• <b>Acoustic Non-Inference Principle:</b> The system measures only the physical mechanics of colliding objects and aerodynamic turbulence. It possesses no capabilities for voice biometric identification, speaker tracking, language recognition, or intent inference.", bullet_style))
    story.append(Paragraph("• <b>24-Hour Ephemeral Privacy Buffer:</b> Continuous background audio streams are processed entirely in volatile RAM and are never recorded to persistent disk storage. When an ambiguous or candidate event is captured for operator assessment, the snippet is held for a maximum of 24 hours, after which an automated retention purge permanently erases the audio file.", bullet_style))
    story.append(Paragraph("• <b>Hardware Deployment Guideline:</b> Because single ambient microphones suffer from acoustic masking when trains pass (>85 dBA), field deployments recommend mounting directional cardioid boundary microphones downward toward target surfaces within 3 to 5 meters, or deploying a dual-microphone cross-correlation array to reject diffuse track noise.", bullet_style))
    story.append(Spacer(1, 10))

    # Section 8: Presentation Slide Deck Summary
    story.append(Paragraph("8. Presentation Slide Deck Summary (Executive Briefing)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph(
        "This concise outline provides slide-ready talking points and takeaways for technical, operational, or executive audiences:",
        body_style
    ))

    slides = [
        [
            Paragraph("Slide 1: Problem & Challenge", table_cell_bold),
            Paragraph("Aerosol can detection cannot rely on loudness or simple pitch filters. Keys, coins, pill bottles, speech sibilants, and spray hiss create false alarm rates exceeding 40% in traditional systems.", table_cell),
        ],
        [
            Paragraph("Slide 2: Physics of Can Shaking", table_cell_bold),
            Paragraph("Aerosol cans have a distinct physical collision signature: internal mixing pea striking metal domes, arm cadence strictly bounded between 2.5 and 15 Hz, deep transit quiet valleys (≥ 4.2 dB), and regular pacing (CV ≤ 0.65).", table_cell),
        ],
        [
            Paragraph("Slide 3: Two-Stage Architecture", table_cell_bold),
            Paragraph("Front-end adaptive spectral noise cancellation + Stage 1 kinematic pre-filter (eliminates non-candidates) + Stage 2 deep 2D-CNN (classifies Log-Mel spectrograms across 6 classes with conservative decision gating).", table_cell),
        ],
        [
            Paragraph("Slide 4: Decision Checklist", table_cell_bold),
            Paragraph("Every candidate must satisfy all 13 gates simultaneously: energy floor, clipping rejection, cadence, temporal span, valley depth, interval regularity, metallic ratio, CNN probability ≥ 0.80, mimic ceiling ≤ 0.20, margin ≥ 0.35, and entropy ≤ 0.70.", table_cell),
        ],
        [
            Paragraph("Slide 5: Performance & Validation", table_cell_bold),
            Paragraph("Validated on 7,399 dataset samples: 99.17% precision on held-out test data, 100% rejection of spray hiss, 0 false alarms on real voicemail/speech audio, and sub-second prompt detection (≤ 0.50s).", table_cell),
        ],
        [
            Paragraph("Slide 6: Continuous Operation & Privacy", table_cell_bold),
            Paragraph("Zero cooldown lockout (continuous monitoring), verified quiet-gap rearming, shadow mode advisory alerts, non-inference acoustic compliance, and automatic 24-hour privacy retention purge.", table_cell),
        ],
    ]
    t_slides = Table(slides, colWidths=[2.2 * inch, 5.1 * inch])
    t_slides.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), BG_LIGHT),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_slides)

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report PDF successfully generated at: {filename}")


if __name__ == "__main__":
    out_file = str(Path(__file__).parent / "Acoustic_Spray_Can_Detector_Presentation_Report.pdf")
    build_pdf(out_file)
