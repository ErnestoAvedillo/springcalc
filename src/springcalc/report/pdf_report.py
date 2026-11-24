"""PDF report generation for compression springs.

Builds a printable summary of a spring's characteristics: the geometry and
material data, the load/travel/diameter curves, and the Goodman fatigue
diagram with its safety-factor analysis. All graphs are produced by the
spring's own ``get_*_graph``/``create_goodman_diagram`` methods (base64 PNGs
from matplotlib) and embedded directly into the PDF.
"""
import base64
import io
from datetime import date
from typing import Optional

import numpy as np
from pint import Quantity
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..lineal.compresion import CompressionSpring

_SPRING_DATA_LABELS = {
    "material": "Material",
    "young_modulus": "Young's modulus (E)",
    "shear_modulus": "Shear modulus (G)",
    "elastic_limit_factor": "Elastic limit factor",
    "poisson_coef": "Poisson coefficient",
    "RMa_file": "RMa reference table",
    "wire_diameter": "Wire diameter (d)",
    "mean_diameter": "Mean diameter (D)",
    "free_length": "Free length (L0)",
    "nr_coils": "Total coils",
    "nr_active_coils": "Active coils",
    "spring_constant": "Spring constant (k)",
    "spring_index": "Spring index (C)",
    "wahl_factor": "Wahl factor",
    "wahl_factor_category": "Wahl factor rating",
    "solid_length": "Solid length",
    "wire_length": "Wire length",
    "number_cycles": "Design cycles",
    "shot_peening": "Shot peening",
}

_TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
])


def _format_number(value: float, decimals: int = 3) -> str:
    """Format a float without scientific notation, dropping trailing zeros."""
    if value == int(value):
        return str(int(value))
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _fmt(value, decimals: int = 3) -> str:
    """Format a spring data value (Quantity, number, str, bool, None) for display."""
    if value is None:
        return "-"
    if isinstance(value, Quantity):
        unit = format(value.units, "~P")
        magnitude = _format_number(float(value.magnitude), decimals)
        return f"{magnitude} {unit}".strip()
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return _format_number(float(value), decimals)
    return str(value)


class SpringPDFReport:
    """Generates a PDF report presenting a compression spring's data and graphs.

    Usage::

        report = SpringPDFReport(spring, title="Spring XYZ-123")
        report.build("spring_report.pdf")

    `spring` may also be an inverse-design result that carries the spring
    as `.spring` (e.g. `ConicalCurveInverseDesign`). If that result has a
    target curve (`target_displacement`/`target_load`), the load graphs
    show those required loads in red and the report adds a table comparing
    them with the loads the spring actually delivers.
    """

    def __init__(self, spring: CompressionSpring, title: Optional[str] = None):
        design = spring
        self.spring = getattr(design, "spring", design)
        self.title = title or "Spring Calculation Report"
        target_displacement = getattr(design, "target_displacement", None)
        target_load = getattr(design, "target_load", None)
        if target_displacement is not None and target_load is not None:
            self.required_curve = (np.asarray(target_displacement, dtype=float),
                                   np.asarray(target_load, dtype=float))
        else:
            self.required_curve = None
        simulated_displacement = getattr(design, "simulated_displacement", None)
        simulated_load = getattr(design, "simulated_load", None)
        if simulated_displacement is not None and simulated_load is not None:
            self._simulated_curve = (np.asarray(simulated_displacement, dtype=float),
                                     np.asarray(simulated_load, dtype=float))
        else:
            self._simulated_curve = None
        self.styles = getSampleStyleSheet()
        self._section_style = ParagraphStyle(
            "SectionTitle", parent=self.styles["Heading2"], spaceBefore=12, spaceAfter=6,
        )
        self._caption_style = ParagraphStyle(
            "Caption", parent=self.styles["Normal"], alignment=TA_CENTER,
            textColor=colors.grey, spaceAfter=8,
        )

    @staticmethod
    def _b64_to_image(image_b64: str, width=160 * mm) -> RLImage:
        raw = base64.b64decode(image_b64)
        img = RLImage(io.BytesIO(raw))
        aspect = img.imageHeight / float(img.imageWidth)
        img.drawWidth = width
        img.drawHeight = width * aspect
        return img

    def _styled_table(self, rows, col_widths=(80 * mm, 80 * mm)):
        table = Table(rows, colWidths=list(col_widths), hAlign="LEFT")
        table.setStyle(_TABLE_STYLE)
        return table

    def _build_header(self):
        return [
            Paragraph(self.title, self.styles["Title"]),
            Paragraph(f"Generated on {date.today().isoformat()}", self.styles["Normal"]),
            Spacer(1, 8 * mm),
        ]

    def _build_data_table(self):
        data = self.spring.get_spring_data()
        rows = [["Property", "Value"]]
        for key, label in _SPRING_DATA_LABELS.items():
            rows.append([label, _fmt(data.get(key))])
        return [
            Paragraph("Spring data", self._section_style),
            self._styled_table(rows),
            Spacer(1, 6 * mm),
        ]

    def _build_positions_table(self):
        positions = self.spring.get_data_positions()
        if not positions:
            return []
        rows = [["Position (mm)", "Travel (mm)", "Load (N)", "Stress (MPa)", "Outer Ø (mm)"]]
        for pc in positions:
            rows.append([
                _fmt(pc.position, 2),
                _fmt(pc.travel, 2),
                _fmt(pc.load, 1),
                _fmt(pc.stress, 1),
                _fmt(pc.outer_diameter, 2),
            ])
        table = self._styled_table(rows, col_widths=[32 * mm] * 5)
        table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        return [
            Paragraph("Load positions", self._section_style),
            table,
            Spacer(1, 6 * mm),
        ]

    def _calculated_loads_at(self, travel_mm: np.ndarray) -> Optional[np.ndarray]:
        """Loads the spring delivers at `travel_mm`, interpolated on its
        simulated curve (from the design result if available, otherwise
        from the spring's own progressive simulation)."""
        curve = self._simulated_curve
        if curve is None and hasattr(self.spring, "_get_load_curve"):
            curve = self.spring._get_load_curve()
        if curve is None:
            return None
        return np.interp(travel_mm, curve[0], curve[1])

    def _build_required_vs_calculated_table(self):
        if self.required_curve is None:
            return []
        travel_mm, required_n = self.required_curve
        calculated_n = self._calculated_loads_at(travel_mm)
        if calculated_n is None:
            return []
        rows = [["Travel (mm)", "Required load (N)", "Calculated load (N)", "Deviation (N)", "Deviation (%)"]]
        for travel, required, calculated in zip(travel_mm, required_n, calculated_n):
            deviation = calculated - required
            deviation_pct = _fmt(100 * deviation / required, 1) if required else "-"
            rows.append([
                _fmt(travel, 2),
                _fmt(required, 1),
                _fmt(calculated, 1),
                _fmt(deviation, 1),
                deviation_pct,
            ])
        rmse = float(np.sqrt(np.mean((calculated_n - required_n) ** 2)))
        table = self._styled_table(rows, col_widths=[32 * mm] * 5)
        table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        return [
            Paragraph("Required vs. calculated load", self._section_style),
            Paragraph(f"RMS deviation: {_fmt(rmse, 2)} N", self.styles["Normal"]),
            Spacer(1, 3 * mm),
            table,
            Spacer(1, 6 * mm),
        ]

    def _build_3d_section(self):
        elements = [Paragraph("3D isometric view", self._section_style)]
        try:
            image_b64 = self.spring.get_3d_plot(show=False)
        except Exception as exc:
            elements.append(Paragraph(
                f"3D view could not be generated ({exc})", self.styles["Normal"]))
            return elements
        elements.append(self._b64_to_image(image_b64, width=120 * mm))
        elements.append(Spacer(1, 6 * mm))
        return elements

    def _build_graphs(self):
        elements = [Paragraph("Load and geometry curves", self._section_style)]
        graph_calls = [
            ("Load vs. position", self.spring.get_forces_vs_position_graph),
            ("Load vs. travel", self.spring.get_forces_vs_travel_graph),
            ("Outer diameter vs. position", self.spring.get_diameter_vs_position_graph),
        ]
        load_graphs = {self.spring.get_forces_vs_position_graph, self.spring.get_forces_vs_travel_graph}
        for label, method in graph_calls:
            try:
                if self.required_curve is not None and method in load_graphs:
                    image_b64 = method(show=False, required_curve=self.required_curve)
                else:
                    image_b64 = method(show=False)
            except Exception as exc:
                elements.append(Paragraph(
                    f"{label}: could not be generated ({exc})", self.styles["Normal"]))
                continue
            elements.append(self._b64_to_image(image_b64))
            elements.append(Paragraph(label, self._caption_style))
        return elements

    def _build_goodman_section(self):
        elements = [Paragraph("Fatigue analysis (Goodman diagram)", self._section_style)]
        result = self.spring.create_goodman_diagram(show=False)
        if not result or "error" in result:
            reason = result.get("error") if result else "no load positions available"
            message = (
                "Goodman diagram could not be generated. "
                "Add at least one load position to the spring before generating the report."
            )
            if reason:
                message += f" Details: {reason}"
            elements.append(Paragraph(message, self.styles["Normal"]))
            return elements

        elements.append(self._b64_to_image(result["image"], width=140 * mm))
        elements.append(Spacer(1, 4 * mm))

        analysis = result["analysis"]
        stresses = result["stresses"]
        rows = [
            ["Parameter", "Value"],
            ["Maximum stress", _fmt(stresses["stress_max"], 1) + " MPa"],
            ["Minimum stress", _fmt(stresses["stress_min"], 1) + " MPa"],
            ["Maximum load", _fmt(stresses["load_max"], 1) + " N"],
            ["Minimum load", _fmt(stresses["load_min"], 1) + " N"],
            ["Corrected fatigue limit (Se)", _fmt(analysis["strengths"]["Se_MPa"], 1) + " MPa"],
            ["Corrected fatigue strength (Sf)", _fmt(analysis["strengths"]["Sf_MPa"], 1) + " MPa"],
            ["Safety factor", _fmt(analysis["safety_factor"], 2)],
        ]
        elements.append(self._styled_table(rows))
        return elements

    def build(self, output_path: str) -> str:
        """Render the report and write it to ``output_path``. Returns the path."""
        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        )
        story = []
        story += self._build_header()
        story += self._build_data_table()
        story += self._build_3d_section()
        story.append(PageBreak())
        story += self._build_graphs()
        story.append(PageBreak())
        required_table = self._build_required_vs_calculated_table()
        if required_table:
            story += required_table
            story.append(PageBreak())
        story += self._build_positions_table()
        story += self._build_goodman_section()
        doc.build(story)
        return output_path
