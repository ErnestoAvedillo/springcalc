"""Manufacturing spec sheet for suppliers: dimensions with tolerances and a
material/wire callout a supplier needs to quote and manufacture a spring.
Deliberately leaves out the fatigue/safety analysis and load/travel curves
that `SpringPDFReport` carries for internal (colleague-facing) use -- a
supplier manufactures to the dimensional spec, not to the customer's
internal safety margin.
"""
from datetime import date
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

from ..pymodels.wire_characteristics import get_standard_wire_diameters
from .geometry_export import spring_geometry_profile
from .pdf_report import _TABLE_STYLE, _fmt, SpringPDFReport

_SPEC_LABELS = {
    "material": "Material grade",
    "wire_diameter": "Wire diameter (d)",
    "diameter_tolerance_mm": "Wire diameter tolerance (± mm)",
    "RMa_min_MPa": "Tensile strength min, Rm (MPa)",
    "RMa_max_MPa": "Tensile strength max, Rm (MPa)",
    "mean_diameter": "Mean coil diameter (D)",
    "free_length": "Free length (L0)",
    "nr_coils": "Total coils",
    "solid_length": "Solid length",
    "spring_index": "Spring index (C)",
    "type_of_end": "End type",
    "type_conforming": "Forming",
    "shot_peening": "Shot peening",
    "coating": "Coating",
}


class SupplierSpecSheet:
    """Generates a manufacturing spec-sheet PDF for a spring, meant to hand
    to a supplier for quoting/manufacturing.

    Usage::

        sheet = SupplierSpecSheet(design, title="Spring XYZ-123", part_number="XYZ-123")
        sheet.build("spring_spec.pdf")
    """

    def __init__(self, design, title: Optional[str] = None, part_number: Optional[str] = None):
        self.design = design
        self.spring = getattr(design, "spring", design)
        self.title = title or "Spring Manufacturing Specification"
        self.part_number = part_number
        self.styles = getSampleStyleSheet()
        self._section_style = ParagraphStyle(
            "SectionTitle", parent=self.styles["Heading2"], spaceBefore=12, spaceAfter=6,
        )

    def _build_header(self):
        elements = [Paragraph(self.title, self.styles["Title"])]
        if self.part_number:
            elements.append(Paragraph(f"Part number: {self.part_number}", self.styles["Normal"]))
        elements.append(Paragraph(f"Generated on {date.today().isoformat()}", self.styles["Normal"]))
        elements.append(Spacer(1, 8 * mm))
        return elements

    def _spec_data(self) -> dict:
        data = dict(self.spring.get_spring_data())
        data["type_of_end"] = getattr(self.spring, "type_of_end", None)
        data["type_conforming"] = getattr(self.spring, "type_conforming", None)
        data["diameter_tolerance_mm"] = getattr(self.spring, "diameter_tolerance", None)
        data["RMa_min_MPa"] = getattr(self.spring, "RMa_min", None)
        data["RMa_max_MPa"] = getattr(self.spring, "RMa_max", None)
        return data

    def _build_data_table(self):
        data = self._spec_data()
        rows = [["Property", "Value"]]
        for key, label in _SPEC_LABELS.items():
            rows.append([label, _fmt(data.get(key))])
        table = Table(rows, colWidths=[80 * mm, 80 * mm], hAlign="LEFT")
        table.setStyle(_TABLE_STYLE)
        return [
            Paragraph("Specification", self._section_style),
            table,
            Spacer(1, 6 * mm),
        ]

    def _build_profile_table(self):
        """Coil-diameter/pitch-vs-position table, sampled directly from the
        spring's own f_mean_diameter/f_pitch functions (see
        `spring_geometry_profile`) rather than assuming a linear taper --
        so this covers a constant, a linear taper, or any other profile the
        underlying CompressionSpringGeneral was built with. Omitted for a
        plain (non-variable) CompressionSpring, whose single mean_diameter
        row in the Specification table already says everything."""
        profile = spring_geometry_profile(self.spring)
        if not profile:
            return []
        rows = [["Position from fixed end (mm)", "Mean diameter (mm)", "Pitch (mm)"]]
        for point in profile:
            rows.append([
                _fmt(point["position_mm"], 1),
                _fmt(point["mean_diameter_mm"], 2),
                _fmt(point["pitch_mm"], 2),
            ])
        table = Table(rows, colWidths=[55 * mm] * 3, hAlign="LEFT")
        table.setStyle(_TABLE_STYLE)
        return [
            Paragraph("Diameter / pitch profile", self._section_style),
            table,
            Spacer(1, 6 * mm),
        ]

    def _build_standard_size_note(self):
        wire_diameter_mm = float(self.spring.wire_diameter.to("mm").magnitude)
        standard_diameters = get_standard_wire_diameters()
        is_standard = any(abs(d - wire_diameter_mm) < 1e-6 for d in standard_diameters)
        note = (
            "Wire diameter is a standard size."
            if is_standard
            else "Wire diameter is NOT a standard size; confirm availability with the supplier "
                 "or select a standard size before ordering."
        )
        return [Paragraph(note, self.styles["Normal"]), Spacer(1, 4 * mm)]

    def _build_3d_section(self):
        elements = [Paragraph("Geometry", self._section_style)]
        try:
            image_b64 = self.spring.get_3d_plot(show=False)
        except Exception as exc:
            elements.append(Paragraph(f"3D view could not be generated ({exc})", self.styles["Normal"]))
            return elements
        elements.append(SpringPDFReport._b64_to_image(image_b64, width=120 * mm))
        elements.append(Spacer(1, 6 * mm))
        return elements

    def build(self, output_path: str) -> str:
        """Render the spec sheet and write it to `output_path`. Returns the path."""
        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        )
        story = []
        story += self._build_header()
        story += self._build_data_table()
        story += self._build_profile_table()
        story += self._build_standard_size_note()
        story += self._build_3d_section()
        doc.build(story)
        return output_path
