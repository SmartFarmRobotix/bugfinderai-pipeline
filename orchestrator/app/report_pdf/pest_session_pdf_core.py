# Copyright 2026 Smart Farm Robotix VCC
#
# This file is derived from the OpenAgri Reporting service
# (https://github.com/agstack/OpenAgri-ReportingService), which is licensed under the
# European Union Public Licence v. 1.2 (EUPL-1.2). It is distributed under the EUPL-1.2,
# not under the Apache License of the rest of this repository. The licence text is in
# LICENSE-EUPL-1.2 beside this file.

"""
Pest-session PDF builder: a copy of the OpenAgri Reporting service's builder, kept identical so that the report rendered
on the device matches the report rendered in the cloud. The OpenAgri logo and the fonts are bundled under
``assets/`` and ``fonts/``.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

from fpdf import FPDF
from fpdf.enums import XPos, YPos

logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(_THIS_DIR, "fonts")
_ASSETS_DIR = os.path.join(_THIS_DIR, "assets")


def add_fonts(pdf: "FPDF", fonts_dir: str = _FONTS_DIR) -> None:
    pdf.add_font("FreeSerif", "", os.path.join(fonts_dir, "FreeSerif.ttf"), uni=True)
    pdf.add_font("FreeSerif", "B", os.path.join(fonts_dir, "FreeSerifBold.ttf"), uni=True)


class EX(FPDF):
    """FPDF subclass carrying the report header/footer shared by every PDF
    report this service builds. `logo_path`/`eu_ack_image_path` default to
    the bundled local assets next to this module; overridable so a mirrored
    copy (e.g. the edge-local orchestrator core) can point at its own copy
    of the same two files without editing this class."""

    def __init__(
        self,
        *args: Any,
        logo_path: str = os.path.join(_ASSETS_DIR, "openagri_logo.png"),
        eu_ack_image_path: str = os.path.join(_ASSETS_DIR, "eu.png"),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._logo_path = logo_path
        self._eu_ack_image_path = eu_ack_image_path

    def header(self):
        self.image(
            self._logo_path,
            w=40.0,
            keep_aspect_ratio=True,
            x=160,
        )

    def footer(self):
        self.set_y(-15)
        self.set_font("FreeSerif", "", 7)
        acknowledgement_text = """
            \tOpenAgri has received funding from the EU's Horizon Europe research and  innovation programme under Grant Agreement no. 101134083. This output reflects
            only the author's view and the European Commission cannot be held responsible for any use that may be made of the information contained therein.
        """
        self.image(self._eu_ack_image_path, x=self.x, y=self.y, w=20)
        self.set_x(self.get_x() + 10)
        self.multi_cell(200, 2, acknowledgement_text, border=0, align="J")


ImageSource = Callable[[str], Optional[Tuple[bytes, str]]]


class PdfRenderError(RuntimeError):
    """The PDF could not be written to `pdf_output_path`."""


@dataclass
class PdfRenderResult:
    images_embedded: bool


def _format_window_datetime(raw: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 payload timestamp, tolerating 'Z'; None if absent
    or unparseable (callers fall back to the raw string — a malformed
    timestamp must never break rendering)."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _tz_designator(dt: datetime) -> str:
    offset = dt.utcoffset()
    if offset is None or offset.total_seconds() == 0:
        return "UTC"
    total_min = int(offset.total_seconds() // 60)
    sign = "+" if total_min >= 0 else "-"
    total_min = abs(total_min)
    return f"UTC{sign}{total_min // 60:02d}:{total_min % 60:02d}"


def format_session_window(time_start: Optional[str], time_end: Optional[str]) -> str:
    """Human-readable session window: '13 Feb 2026, 07:23-07:57 UTC' for a
    same-day window, full date on both sides otherwise. Each side falls
    back to its own raw string INDEPENDENTLY when it does not parse — one
    malformed boundary never drags the other back to machine format."""
    start = _format_window_datetime(time_start)
    end = _format_window_datetime(time_end)
    if start is not None and end is not None:
        tz = _tz_designator(start)
        if start.date() == end.date() and _tz_designator(end) == tz:
            return (
                f"{start.strftime('%d %b %Y')}, "
                f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')} {tz}"
            )
        return (
            f"{start.strftime('%d %b %Y %H:%M')} {tz} - "
            f"{end.strftime('%d %b %Y %H:%M')} {_tz_designator(end)}"
        )
    start_text = (
        f"{start.strftime('%d %b %Y %H:%M')} {_tz_designator(start)}"
        if start is not None else (time_start or "n/a")
    )
    end_text = (
        f"{end.strftime('%d %b %Y %H:%M')} {_tz_designator(end)}"
        if end is not None else (time_end or "n/a")
    )
    return f"{start_text} -> {end_text}"


def format_window_string(raw: Optional[str]) -> str:
    """Format a pre-joined window STRING (the composers' shared
    '{start} → {end}' shape — orchestrator compute_weather_window and
    PDM's risk core both emit it, '->' tolerated). Unrecognized shapes
    pass through verbatim; a single parseable timestamp is formatted
    alone."""
    if not raw:
        return ""
    text = str(raw)
    for sep in ("→", "->"):
        if sep in text:
            left, _, right = text.partition(sep)
            return format_session_window(left.strip(), right.strip())
    single = _format_window_datetime(text.strip())
    if single is not None:
        return f"{single.strftime('%d %b %Y %H:%M')} {_tz_designator(single)}"
    return text


_KEY_VALUE_LABEL_WIDTH = 60


def _label_column_width(pdf: EX, labels: Any) -> float:
    """Width of a label column that holds the longest of `labels` on ONE
    line in the label font, with a gap before the value column. A label
    wider than its cell overruns into the value column and extracts as a
    split line, so rows whose labels are sentence prefixes (the verdict
    rows) size their column from the text, never from a constant."""
    pdf.set_font("FreeSerif", "B", 10)
    widest = max(pdf.get_string_width(label) for label in labels)
    return max(_KEY_VALUE_LABEL_WIDTH, widest + 6)


def _render_key_value(
    pdf: EX, label: str, value: str, label_width: float = _KEY_VALUE_LABEL_WIDTH
) -> None:
    pdf.set_font("FreeSerif", "B", 10)
    pdf.cell(label_width, 6, label)
    pdf.set_font("FreeSerif", "", 10)
    pdf.multi_cell(0, 6, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)


_WEATHER_VARIABLE_WORDS = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "relative humidity",
    "precipitation": "precipitation",
}


def _weather_variables_in_words(variables: Any) -> str:
    """The weather variables as plain words: the known data-source
    identifiers are mapped, an unknown one is printed as it came."""
    return ", ".join(
        _WEATHER_VARIABLE_WORDS.get(str(name), str(name)) for name in (variables or [])
    )


def _split_verdict_line(line: str) -> Optional[Tuple[str, str]]:
    """A composed verdict sentence 'Label: band' as (label incl. the colon
    and space, value), or None when the line is not of that shape. Splits
    at the first ': ' so the pinned sentence prefixes stay whole in the
    label cell whatever they are."""
    label, sep, value = line.partition(": ")
    if not sep or not value.strip():
        return None
    return label + sep, value.strip()


def _render_verdict_rows(pdf: EX, note: str) -> None:
    """The risk paths' fused verdicts (one composed sentence per line of
    `note`) as label/value rows: the label cell carries the sentence prefix
    unchanged, the value cell its band, so a layout-preserving extractor
    reads 'prefix band' on one line. A line that is not a 'Label: value'
    sentence renders as plain text."""
    lines = [line for line in str(note).split("\n") if line.strip()]
    rows = [_split_verdict_line(line) for line in lines]
    labels = [row[0] for row in rows if row]
    width = _label_column_width(pdf, labels) if labels else _KEY_VALUE_LABEL_WIDTH
    for line, row in zip(lines, rows):
        if row:
            _render_key_value(pdf, row[0], row[1], label_width=width)
        else:
            pdf.set_font("FreeSerif", "", 9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def _render_section_header(pdf: EX, title: str) -> None:
    pdf.set_font("FreeSerif", "B", 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(2)


def _render_forecast_outlook(pdf: EX, forecast_outlook: Dict[str, Any]) -> None:
    if not forecast_outlook:
        return
    if not (
        forecast_outlook.get("days")
        or forecast_outlook.get("recommendation")
        or forecast_outlook.get("outlook_summary")
    ):
        return

    days_source = forecast_outlook.get("days_source")
    if days_source == "history":
        _render_section_header(pdf, "Risk outlook: the five days after the session")
    else:
        _render_section_header(pdf, "Forecast risk outlook: the next 5 days")

    source = forecast_outlook.get("source", "")
    fetched_at = forecast_outlook.get("fetched_at", "")
    location = forecast_outlook.get("location") or {}
    if location:
        _render_key_value(
            pdf, "Location",
            f"{location.get('lat', '')}, {location.get('lon', '')}",
        )
    if source:
        _render_key_value(pdf, "Source", source)
    if fetched_at:
        _render_key_value(pdf, "Data fetched at", fetched_at)

    pdf.ln(1)

    days = forecast_outlook.get("days") or []
    if days:
        pdf.set_font("FreeSerif", "B", 9)
        pdf.set_fill_color(210, 225, 210)
        col_w = [24, 17, 17, 23, 21, 19, 35, 30]
        headers = [
            "Date", "Max °C", "Min °C", "Humidity %", "Precip mm", "Wind m/s",
            "Risk (weather rules)", "Risk (fuzzy path)",
        ]
        for h, w in zip(headers, col_w):
            pdf.cell(w, 6, h, border=1, fill=True)
        pdf.ln()

    risk_colors = {
        "High":     (255, 220, 220),
        "Moderate": (255, 245, 200),
        "Low":      (220, 245, 220),
    }
    for day in days:
        risk = day.get("risk_level", "Low")
        r, g, b = risk_colors.get(risk, (240, 240, 240))
        pdf.set_fill_color(r, g, b)
        pdf.set_font("FreeSerif", "", 8)
        wind = day.get("wind_speed_ms", "?")
        row_vals = [
            str(day.get("date", "")),
            f"{day.get('temp_max_C', '?')}",
            f"{day.get('temp_min_C', '?')}",
            f"{day.get('humidity_pct', '?')}",
            f"{day.get('precipitation_mm', '?')}",
            "n/a" if wind is None else f"{wind}",
            risk,
            str(day.get("fuzzy_level") or "n/a"),
        ]
        for val, w in zip(row_vals, col_w):
            pdf.cell(w, 5, val, border=1, fill=True)
        pdf.ln()

    pdf.ln(3)

    for day in days:
        drivers = day.get("drivers") or []
        suppressors = day.get("suppressors") or []
        if not drivers and not suppressors:
            continue
        pdf.set_font("FreeSerif", "B", 8)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 5, f"{day.get('date', '')}  ({day.get('risk_level', '')} risk):",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("FreeSerif", "", 8)
        for d in drivers:
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 4, f"+ {d}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for s in suppressors:
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 4, f"- {s} (suppressor)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(2)

    summary = forecast_outlook.get("outlook_summary", "")
    if summary:
        pdf.set_font("FreeSerif", "B", 9)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, f"Outlook Summary: {summary}",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    recommendation = forecast_outlook.get("recommendation", "")
    if recommendation:
        pdf.set_font("FreeSerif", "B", 9)
        pdf.set_fill_color(255, 235, 180)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, recommendation,
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.ln(2)


def _embed_image(pdf: EX, image_bytes: bytes, content_type: str, width: int = 120) -> bool:
    """Embed image bytes into the PDF. Returns True on success."""
    try:
        image_file = io.BytesIO(image_bytes)
        image_type = "jpg"
        if content_type.endswith("png"):
            image_type = "png"
        elif content_type.endswith("webp"):
            image_type = "webp"
        pdf.image(image_file, type=image_type, w=width)
        return True
    except Exception as exc:
        logger.warning("Failed to embed image in PDF: %s", exc)
        return False


def render_pest_session_pdf(
    payload: Dict[str, Any],
    image_source: ImageSource,
    pdf_output_path: str,
) -> PdfRenderResult:
    """
    Generate a pest session PDF report from the orchestration payload,
    writing it to `pdf_output_path`. `image_source(variant)` is called with
    "clean" and "annotated" -- the caller owns the fetch strategy and any
    further fallback chaining. Raises `PdfRenderError` if the PDF cannot be written.
    """
    pdf = EX()
    add_fonts(pdf)

    envelope = payload.get("envelope") or {}

    _time_start = envelope.get("time_start")
    if _time_start:
        try:
            _creation_dt = datetime.fromisoformat(
                _time_start.replace("Z", "+00:00")
            )
            if _creation_dt.tzinfo is None:
                _creation_dt = _creation_dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            _creation_dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    else:
        _creation_dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    pdf.creation_date = _creation_dt

    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("FreeSerif", "B", 16)
    pdf.cell(0, 10, "Pest Session Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    time_start = envelope.get("time_start")
    time_end = envelope.get("time_end")
    pdf.set_font("FreeSerif", "", 9)
    if time_start or time_end:
        pdf.cell(
            0, 6,
            f"Session Window: {format_session_window(time_start, time_end)}",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C",
        )
    else:
        pdf.cell(
            0, 6,
            f"Generated: {_creation_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C",
        )
    pdf.ln(4)

    _render_section_header(pdf, "Session Overview")
    _render_key_value(pdf, "Session ID", envelope.get("session_id", ""))
    _render_key_value(pdf, "Best Image ID", envelope.get("best_image_id", ""))
    _render_key_value(
        pdf,
        "Infestation Score",
        f"{payload.get('infestation_score', '')}",
    )
    _render_key_value(
        pdf,
        "Time Window",
        format_session_window(envelope.get("time_start"), envelope.get("time_end")),
    )

    lat = envelope.get("latitude")
    lon = envelope.get("longitude")
    site_label = payload.get("site_label")
    if site_label is not None and not isinstance(site_label, str):
        logger.warning(
            "site_label ignored: expected a string, got %s", type(site_label).__name__
        )
        site_label = None
    if lat is not None and lon is not None:
        location_value = f"{lat}, {lon}"
        if site_label:
            location_value = f"{location_value}  -  {site_label}"
        _render_key_value(pdf, "Location", location_value)
    else:
        _render_key_value(pdf, "Location", "Position unavailable for this session")

    if envelope.get("parcel_id"):
        _render_key_value(pdf, "Parcel ID", envelope.get("parcel_id", ""))

    _render_section_header(pdf, "Risk Assessment")
    risk = payload.get("risk_assessment") or {}
    _render_key_value(pdf, "Risk Level", risk.get("level", "n/a"))
    if risk.get("details"):
        _render_key_value(pdf, "Details", risk.get("details", ""))

    _render_section_header(pdf, "Pest & Crop Context")
    pest_ctx = payload.get("pest_crop_context") or {}
    if pest_ctx:
        pest_name = pest_ctx.get("pest_name", "")
        pest_scientific_name = pest_ctx.get("pest_scientific_name", "")
        pest_eppo_code = pest_ctx.get("pest_eppo_code", "")
        pest_eppo_url = pest_ctx.get("pest_eppo_url", "")
        pest_description = pest_ctx.get("pest_description", "")
        symptoms = pest_ctx.get("symptoms", "")
        crop_name = pest_ctx.get("crop_name", "")
        crop_growth_stage = pest_ctx.get("crop_growth_stage", "")

        if pest_name:
            pest_label = f"{pest_name} ({pest_scientific_name})" if pest_scientific_name else pest_name
            _render_key_value(pdf, "Pest", pest_label)
        if pest_eppo_code:
            _render_key_value(pdf, "EPPO Code", pest_eppo_code)
        if pest_eppo_url:
            _render_key_value(pdf, "EPPO URL", pest_eppo_url)
        if crop_name:
            _render_key_value(pdf, "Host Crop", crop_name)
        if crop_growth_stage:
            _render_key_value(pdf, "Growth Stage", crop_growth_stage)
        if pest_description:
            _render_key_value(pdf, "Description", pest_description)
        if symptoms:
            _render_key_value(pdf, "Symptoms", symptoms)
    else:
        unavailable_reason = payload.get("pest_crop_context_unavailable_reason")
        pdf.set_font("FreeSerif", "", 9)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            0, 5,
            unavailable_reason
            if isinstance(unavailable_reason, str) and unavailable_reason
            else "No pest/crop context provided.",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

    risk_reasoning = payload.get("risk_reasoning") or {}
    if risk_reasoning:
        _render_section_header(pdf, "Risk Reasoning")

        nn_score = risk_reasoning.get("nn_score")
        nn_verdict = risk_reasoning.get("nn_verdict", "")
        nn_note = risk_reasoning.get("nn_note", "")
        if nn_score is not None:
            _render_key_value(
                pdf,
                "NN Detection Score",
                f"{nn_score:.2f}  ->  {nn_verdict}",
            )
        if nn_note:
            _render_verdict_rows(pdf, nn_note)

        rules_list = risk_reasoning.get("pdms_rules_evaluated") or []
        if rules_list:
            pdf.set_font("FreeSerif", "B", 10)
            pdf.cell(0, 6, "PDMS Weather-Rule Evaluation:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            for model_entry in rules_list:
                model_name = model_entry.get("model", "unknown model")
                dominant = model_entry.get("dominant_level", "")
                pdf.set_font("FreeSerif", "B", 9)
                pdf.set_x(pdf.l_margin + 4)
                pdf.cell(
                    0, 5,
                    f"Model: {model_name}  |  Dominant level: {dominant.upper() if dominant else 'n/a'}",
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                )
                for rule in model_entry.get("rules", []):
                    level = str(rule.get("level", "")).upper()
                    conditions = rule.get("conditions", "")
                    hours_trig = rule.get("hours_triggered")
                    hours_total = rule.get("hours_total")
                    hours_text = (
                        f"{hours_trig}/{hours_total} h"
                        if hours_trig is not None and hours_total is not None
                        else ""
                    )
                    line = f"  [{level}]  {conditions}"
                    if hours_text:
                        line += f"  ->  fired {hours_text}"
                    pdf.set_font("FreeSerif", "", 9)
                    pdf.set_x(pdf.l_margin + 8)
                    pdf.multi_cell(0, 5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)

        weather_window = risk_reasoning.get("weather_window", "")
        if weather_window:
            _render_key_value(pdf, "Weather Window", format_window_string(weather_window))

        combined = risk_reasoning.get("combined_level", "")
        if combined:
            pdf.set_font("FreeSerif", "B", 10)
            pdf.set_fill_color(220, 235, 220)
            pdf.cell(0, 7, f"Combined Risk Assessment:  {combined.upper()}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
            pdf.ln(1)

        narrative = risk_reasoning.get("narrative", "")
        if narrative:
            pdf.set_font("FreeSerif", "", 9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, narrative, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)

    _render_section_header(pdf, "Weather Summary")
    weather = payload.get("weather_summary") or {}
    _weather_has_data = bool(weather.get("source") or weather.get("observed"))
    if weather.get("unavailable_reason") and not _weather_has_data:
        pdf.set_font("FreeSerif", "", 9)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            0, 5,
            f"Weather data unavailable for this session: {weather.get('unavailable_reason')}",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
    elif weather:
        source = str(weather.get("source", ""))
        source_note = weather.get("source_note", "")
        source_label = source
        if source_note:
            source_label = f"{source}  ({source_note})"
        _render_key_value(pdf, "Source", source_label)
        _render_key_value(pdf, "Variables", _weather_variables_in_words(weather.get("variables")))
        location = weather.get("location") or {}
        if location:
            _render_key_value(
                pdf,
                "Weather Location",
                f"{location.get('lat', '')}, {location.get('lon', '')}",
            )
        observed = weather.get("observed") or {}
        if observed:
            pdf.set_font("FreeSerif", "B", 10)
            pdf.cell(0, 6, "Observed Conditions:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            period = observed.get("period", "")
            if period:
                pdf.set_font("FreeSerif", "", 9)
                pdf.set_x(pdf.l_margin + 4)
                pdf.cell(0, 5, f"Period: {period}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            obs_fields = [
                ("temperature_2m_avg_C",         "Temp avg",   "C"),
                ("temperature_2m_max_C",         "Temp max",   "C"),
                ("temperature_2m_min_C",         "Temp min",   "C"),
                ("relative_humidity_2m_avg_pct", "Humidity avg", "%"),
                ("precipitation_mm",             "Precipitation", "mm"),
            ]
            for key, label, unit in obs_fields:
                val = observed.get(key)
                if val is not None:
                    pdf.set_font("FreeSerif", "", 9)
                    pdf.set_x(pdf.l_margin + 4)
                    pdf.cell(0, 5, f"{label}: {val} {unit}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.set_font("FreeSerif", "", 9)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, "No weather summary provided.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    forecast_outlook = payload.get("forecast_outlook")
    if forecast_outlook:
        _render_forecast_outlook(pdf, forecast_outlook)

    _render_section_header(pdf, "Session image")

    clean_bundle = image_source("clean")
    annotated_bundle = image_source("annotated")

    clean_embedded = False
    annotated_embedded = False
    image_width = 88

    if clean_bundle:
        pdf.set_font("FreeSerif", "B", 9)
        pdf.cell(0, 5, "Best image", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        img_bytes, ct = clean_bundle
        clean_embedded = _embed_image(pdf, img_bytes, ct, width=image_width)
        pdf.ln(3)

    if annotated_bundle:
        pdf.set_font("FreeSerif", "B", 9)
        pdf.cell(0, 5, "Best image with the detected regions", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        img_bytes, ct = annotated_bundle
        annotated_embedded = _embed_image(pdf, img_bytes, ct, width=image_width)
        pdf.ln(3)

    images_embedded = clean_embedded and annotated_embedded

    if not (clean_embedded or annotated_embedded):
        pdf.set_font("FreeSerif", "", 9)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            0, 5,
            "Session images unavailable; report generated without images.",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

    _render_section_header(pdf, "Inference Details")
    provenance = envelope.get("provenance") or {}
    if provenance:
        _render_key_value(
            pdf, "Detector",
            f"{provenance.get('detector_impl_id', 'n/a')} version {provenance.get('detector_version', '?')}"
            f" ({provenance.get('designation_status', 'n/a')})",
        )
        _render_key_value(pdf, "Scoring allowlist", ", ".join(provenance.get("target_class_set") or []))
        _render_key_value(pdf, "Confidence floor", f"{provenance.get('confidence_floor', 'n/a')}")
    else:
        pdf.set_font("FreeSerif", "", 9)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, "No detector provenance reported.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    details_full = (payload.get("risk_assessment") or {}).get("assessment_details_full")
    if details_full is not None and not isinstance(details_full, str):
        logger.warning(
            "assessment_details_full ignored: expected a string, got %s",
            type(details_full).__name__,
        )
        details_full = None
    fuzzy_details = (payload.get("risk_assessment") or {}).get("fuzzy_details")
    if fuzzy_details is not None and not isinstance(fuzzy_details, str):
        logger.warning(
            "fuzzy_details ignored: expected a string, got %s",
            type(fuzzy_details).__name__,
        )
        fuzzy_details = None

    if fuzzy_details:
        pdf.set_font("FreeSerif", "B", 9)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 5, "Assessment details", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if details_full:
            pdf.set_font("FreeSerif", "B", 9)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 5, "Weather rules (PDM)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("FreeSerif", "", 9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, details_full, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("FreeSerif", "B", 9)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 5, "Fuzzy model (Mamdani)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("FreeSerif", "", 9)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, fuzzy_details, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
    elif details_full:
        pdf.set_font("FreeSerif", "B", 9)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 5, "Assessment details", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("FreeSerif", "", 9)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, details_full, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    engine_provenance = (payload.get("risk_reasoning") or {}).get("engine_provenance")
    if engine_provenance and not isinstance(engine_provenance, (list, tuple)):
        logger.warning(
            "engine_provenance ignored: expected a list of lines, got %s",
            type(engine_provenance).__name__,
        )
        engine_provenance = None
    if engine_provenance:
        pdf.set_font("FreeSerif", "B", 8)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 4, "Engine provenance:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("FreeSerif", "", 8)
        for prov_line in engine_provenance:
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 4, str(prov_line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    try:
        os.makedirs(os.path.dirname(pdf_output_path), exist_ok=True)
        pdf.output(pdf_output_path)
    except Exception as exc:
        raise PdfRenderError(f"PDF generation failed: {exc}") from exc

    return PdfRenderResult(images_embedded=images_embedded)
