#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de presentación PowerPoint (.pptx) con texto justificado y sintetizado
para exposición oral (evitando saturación de lectura) del artículo:
"Análisis Comparativo de Métodos de Machine Learning para la Detección Temprana
de Cáncer Cervicouterino: Avances, Brechas y Desafíos de Explicabilidad"
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Paleta de colores académica / médica moderna
NAVY_DARK = RGBColor(15, 32, 67)       # #0F2043 - Azul marino institucional
NAVY_MED = RGBColor(28, 59, 112)       # #1C3B70 - Azul medio
TEAL_ACCENT = RGBColor(0, 150, 166)    # #0096A6 - Turquesa clínico
CORAL_ALERT = RGBColor(214, 69, 65)    # #D64541 - Coral / alerta clínica
GOLD_ACCENT = RGBColor(230, 159, 0)    # #E69F00 - Ámbar / destaque
GREEN_OK = RGBColor(39, 174, 96)       # #27AE60 - Verde éxito
BG_LIGHT = RGBColor(246, 248, 252)     # #F6F8FC - Fondo claro diapositiva
WHITE = RGBColor(255, 255, 255)
TEXT_DARK = RGBColor(30, 41, 59)       # #1E293B - Gris pizarra oscuro
TEXT_MUTED = RGBColor(90, 105, 125)    # #5A697D - Gris medio
BORDER_GRAY = RGBColor(203, 213, 225)  # #CBD5E1 - Borde suave
ROW_ALT = RGBColor(235, 241, 248)      # #EBF1F8 - Fila alterna tabla

FONT_MAIN = "Calibri"


def set_slide_background(slide, color):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title_text, category_text="ANÁLISIS COMPARATIVO DE MÉTODOS DE ML EN CÁNCER CERVICOUTERINO", slide_num=None):
    set_slide_background(slide, BG_LIGHT)

    # Franja superior decorativa
    top_bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.10)
    )
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = TEAL_ACCENT
    top_bar.line.fill.background()

    # Categoría / Subtítulo superior
    cat_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.16), Inches(11.5), Inches(0.32))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = FONT_MAIN
    p_cat.font.size = Pt(12)
    p_cat.font.bold = True
    p_cat.font.color.rgb = TEAL_ACCENT

    # Título principal de la diapositiva
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.42), Inches(12.3), Inches(0.58))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = FONT_MAIN
    p_title.font.size = Pt(25)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY_DARK

    # Línea divisoria bajo el título
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.02), Inches(12.333), Inches(0.02)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_GRAY
    line.line.fill.background()

    # Pie de página y número de diapositiva
    footer_box = slide.shapes.add_textbox(Inches(0.5), Inches(7.12), Inches(11.0), Inches(0.3))
    tf_foot = footer_box.text_frame
    tf_foot.margin_left = tf_foot.margin_top = tf_foot.margin_right = tf_foot.margin_bottom = 0
    p_foot = tf_foot.paragraphs[0]
    p_foot.text = "Detección Temprana de Cáncer Cervicouterino (CaCu) • Machine Learning y Explicabilidad (XAI)"
    p_foot.font.name = FONT_MAIN
    p_foot.font.size = Pt(11.5)
    p_foot.font.color.rgb = TEXT_MUTED

    if slide_num is not None:
        num_box = slide.shapes.add_textbox(Inches(11.8), Inches(7.10), Inches(1.033), Inches(0.32))
        tf_num = num_box.text_frame
        tf_num.margin_left = tf_num.margin_top = tf_num.margin_right = tf_num.margin_bottom = 0
        p_num = tf_num.paragraphs[0]
        p_num.alignment = PP_ALIGN.RIGHT
        p_num.text = f"{slide_num}"
        p_num.font.name = FONT_MAIN
        p_num.font.size = Pt(12)
        p_num.font.bold = True
        p_num.font.color.rgb = NAVY_MED


def add_card(
    slide, left, top, width, height, title, bullets,
    accent_color=NAVY_MED, bg_color=WHITE,
    title_size=19, body_size=16, space_after=10
):
    # Fondo de tarjeta
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.adjustments[0] = 0.025
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = BORDER_GRAY
    card.line.width = Pt(1.2)

    # Barra de acento superior en la tarjeta
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Inches(0.09))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent_color
    bar.line.fill.background()

    # Contenedor de texto
    tb = slide.shapes.add_textbox(
        left + Inches(0.24), top + Inches(0.18),
        width - Inches(0.48), height - Inches(0.28)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_title = tf.paragraphs[0]
    p_title.text = title
    p_title.font.name = FONT_MAIN
    p_title.font.size = Pt(title_size)
    p_title.font.bold = True
    p_title.font.color.rgb = accent_color
    p_title.space_after = Pt(space_after + 2)
    p_title.alignment = PP_ALIGN.LEFT

    for b in bullets:
        p = tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = 1.18
        p.alignment = PP_ALIGN.JUSTIFY
        if isinstance(b, tuple):
            bold_prefix, rest = b
            r1 = p.add_run()
            r1.text = "• " + bold_prefix + " "
            r1.font.name = FONT_MAIN
            r1.font.size = Pt(body_size)
            r1.font.bold = True
            r1.font.color.rgb = TEXT_DARK

            r2 = p.add_run()
            r2.text = rest
            r2.font.name = FONT_MAIN
            r2.font.size = Pt(body_size)
            r2.font.color.rgb = TEXT_DARK
        else:
            p.text = "• " + b
            p.font.name = FONT_MAIN
            p.font.size = Pt(body_size)
            p.font.color.rgb = TEXT_DARK


def style_table(table, col_widths, headers, rows, highlight_rows=None, font_size=14, header_size=None, justify_cols=None):
    highlight_rows = highlight_rows or {}
    justify_cols = set(justify_cols or [])
    header_size = header_size or font_size
    for idx, w in enumerate(col_widths):
        table.columns[idx].width = w

    # Encabezado
    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_DARK
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.08)
        cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.04)
        cell.margin_bottom = Inches(0.04)
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.font.name = FONT_MAIN
        p.font.size = Pt(header_size)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT

    # Filas
    for r_idx, row_data in enumerate(rows):
        bg = highlight_rows.get(r_idx, ROW_ALT if r_idx % 2 == 1 else WHITE)
        is_highlighted = r_idx in highlight_rows
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.name = FONT_MAIN
            p.font.size = Pt(font_size)
            p.font.bold = is_highlighted or (c_idx == 0)
            p.font.color.rgb = NAVY_DARK if is_highlighted else TEXT_DARK
            if c_idx in justify_cols:
                p.alignment = PP_ALIGN.JUSTIFY
            else:
                p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT


def build_presentation(output_path="Presentacion_Analisis_Comparativo_CaCu_ML.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    slide_counter = 0

    def next_slide():
        nonlocal slide_counter
        slide_counter += 1
        return prs.slides.add_slide(blank_layout), slide_counter

    # =========================================================================
    # SLIDE 1: PORTADA
    # =========================================================================
    slide1, s_num = next_slide()
    set_slide_background(slide1, NAVY_DARK)

    side_panel = slide1.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(12.8), Inches(0), Inches(0.533), Inches(7.5)
    )
    side_panel.fill.solid()
    side_panel.fill.fore_color.rgb = TEAL_ACCENT
    side_panel.line.fill.background()

    accent_line = slide1.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.95), Inches(1.6), Inches(0.08)
    )
    accent_line.fill.solid()
    accent_line.fill.fore_color.rgb = TEAL_ACCENT
    accent_line.line.fill.background()

    tag_box = slide1.shapes.add_textbox(Inches(0.8), Inches(0.50), Inches(11.2), Inches(0.4))
    tf_tag = tag_box.text_frame
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = "ARTÍCULO CIENTÍFICO • INTELIGENCIA ARTIFICIAL EN ONCOLOGÍA CLÍNICA"
    p_tag.font.name = FONT_MAIN
    p_tag.font.size = Pt(14)
    p_tag.font.bold = True
    p_tag.font.color.rgb = TEAL_ACCENT

    title_box = slide1.shapes.add_textbox(Inches(0.8), Inches(1.18), Inches(11.5), Inches(2.3))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.alignment = PP_ALIGN.JUSTIFY
    p_t.text = "Análisis Comparativo de Métodos de Machine Learning para la Detección Temprana de Cáncer Cervicouterino:"
    p_t.font.name = FONT_MAIN
    p_t.font.size = Pt(32)
    p_t.font.bold = True
    p_t.font.color.rgb = WHITE
    p_t.space_after = Pt(10)

    p_t2 = tf_t.add_paragraph()
    p_t2.text = "Avances, Brechas y Desafíos de Explicabilidad"
    p_t2.font.name = FONT_MAIN
    p_t2.font.size = Pt(28)
    p_t2.font.bold = True
    p_t2.font.color.rgb = RGBColor(125, 211, 252)

    en_box = slide1.shapes.add_textbox(Inches(0.8), Inches(3.55), Inches(11.4), Inches(0.85))
    tf_en = en_box.text_frame
    tf_en.word_wrap = True
    p_en = tf_en.paragraphs[0]
    p_en.alignment = PP_ALIGN.JUSTIFY
    p_en.text = "Comparative Analysis of Machine Learning Methods for Early Cervical Cancer Detection: Breakthroughs, Gaps, and Explainability Challenges"
    p_en.font.name = FONT_MAIN
    p_en.font.size = Pt(16.5)
    p_en.font.italic = True
    p_en.font.color.rgb = RGBColor(203, 213, 225)

    # Autores
    auth_box = slide1.shapes.add_textbox(Inches(0.8), Inches(4.42), Inches(11.4), Inches(0.55))
    tf_au = auth_box.text_frame
    tf_au.word_wrap = True
    p_au = tf_au.paragraphs[0]
    p_au.text = "Jacinto Alexander Acosta Zambrano¹  •  Claudio Meneses Villegas²,*   |   Universidad Católica del Norte (Antofagasta, Chile)"
    p_au.font.name = FONT_MAIN
    p_au.font.size = Pt(15.5)
    p_au.font.bold = True
    p_au.font.color.rgb = GOLD_ACCENT

    meta_card = slide1.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.10), Inches(11.4), Inches(1.90)
    )
    meta_card.adjustments[0] = 0.04
    meta_card.fill.solid()
    meta_card.fill.fore_color.rgb = RGBColor(23, 47, 92)
    meta_card.line.color.rgb = TEAL_ACCENT
    meta_card.line.width = Pt(1.5)

    tb_meta = slide1.shapes.add_textbox(Inches(1.05), Inches(5.22), Inches(10.9), Inches(1.65))
    tf_meta = tb_meta.text_frame
    tf_meta.word_wrap = True

    p_kw = tf_meta.paragraphs[0]
    p_kw.alignment = PP_ALIGN.JUSTIFY
    r_kw1 = p_kw.add_run()
    r_kw1.text = "Palabras clave: "
    r_kw1.font.bold = True
    r_kw1.font.size = Pt(14.5)
    r_kw1.font.color.rgb = GOLD_ACCENT
    r_kw1.font.name = FONT_MAIN

    r_kw2 = p_kw.add_run()
    r_kw2.text = "Cáncer cervicouterino • Aprendizaje automático • Detección temprana • Inteligencia artificial explicable (XAI) • Comparación de algoritmos"
    r_kw2.font.size = Pt(14.5)
    r_kw2.font.color.rgb = WHITE
    r_kw2.font.name = FONT_MAIN
    p_kw.space_after = Pt(10)

    p_ack = tf_meta.add_paragraph()
    p_ack.alignment = PP_ALIGN.JUSTIFY
    r_ack1 = p_ack.add_run()
    r_ack1.text = "Proyecto BIP 40074839-0: "
    r_ack1.font.bold = True
    r_ack1.font.size = Pt(14)
    r_ack1.font.color.rgb = TEAL_ACCENT
    r_ack1.font.name = FONT_MAIN

    r_ack2 = p_ack.add_run()
    r_ack2.text = "Formación de Capital Humano Avanzado en IA para la Revolución Digital de la Región de Antofagasta (Fondo Regional para la Productividad y el Desarrollo)."
    r_ack2.font.size = Pt(14)
    r_ack2.font.color.rgb = RGBColor(226, 232, 240)
    r_ack2.font.name = FONT_MAIN

    # =========================================================================
    # SLIDE 2: RESUMEN EJECUTIVO (PARTE 1: PROBLEMÁTICA Y HALLAZGOS)
    # =========================================================================
    slide2, s_num = next_slide()
    add_header(slide2, "Resumen Ejecutivo: Problemática y Hallazgos Clave", "1. VISIÓN GENERAL DEL ESTUDIO", s_num)

    add_card(
        slide2, Inches(0.5), Inches(1.18), Inches(6.0), Inches(5.75),
        "Problemática y Enfoque Metodológico",
        [
            ("Limitación del tamizaje clásico:", "El Papanicolaou presenta sensibilidad variable y alta dependencia de personal experto."),
            ("Marco integrador:", "Articula rendimiento predictivo en Machine Learning con tres dimensiones de XAI: transparencia, interpretabilidad y auditabilidad."),
            ("Evaluación bimodal:", "Contrasta clasificadores clásicos en datos clínicos tabulares frente a redes convolucionales (CNN) en citología.")
        ],
        accent_color=NAVY_MED, title_size=20, body_size=16.5, space_after=14
    )

    add_card(
        slide2, Inches(6.8), Inches(1.18), Inches(6.033), Inches(5.75),
        "Hallazgos Empíricos Principales",
        [
            ("Datos tabulares (UCI):", "XGBoost logra alta exactitud (93.80%), pero baja sensibilidad (12.50%) por desbalance de clases."),
            ("Citología (SIPaKMeD y RIVA):", "AlexNet lidera con 93.63% de exactitud (AUC 99.65%) en SIPaKMeD y 59.20% (AUC 88.90%) en RIVA."),
            ("Directriz clínica:", "Calibrar umbrales de decisión y combinar modelos de alto desempeño con explicabilidad post-hoc (SHAP y Grad-CAM).")
        ],
        accent_color=TEAL_ACCENT, title_size=20, body_size=16.5, space_after=14
    )

    # =========================================================================
    # SLIDE 3: RESUMEN EJECUTIVO (PARTE 2: INDICADORES CLAVE DEL ESTUDIO)
    # =========================================================================
    slide3, s_num = next_slide()
    add_header(slide3, "Indicadores Empíricos Destacados (KPIs)", "1. SÍNTESIS CUANTITATIVA", s_num)

    kpi_data = [
        ("93.80%", "Exactitud XGBoost (UCI)", "Sensibilidad de 12.50% con umbral estándar; evidencia la «Paradoja del Accuracy».", CORAL_ALERT),
        ("93.63%", "Exactitud AlexNet (SIPaKMeD)", "AUC-ROC de 99.65%, sensibilidad de 93.99% y especificidad de 98.40% en 5 clases.", GREEN_OK),
        ("59.20%", "Exactitud AlexNet (RIVA)", "AUC-ROC de 88.90% y especificidad de 93.49% en 8 categorías del Sistema Bethesda.", TEAL_ACCENT),
        ("< 5%", "Modelos con XAI Validada", "Escasa validación clínica de explicabilidad post-hoc en la literatura académica actual.", GOLD_ACCENT),
    ]
    for i, (big_num, kpi_title, kpi_sub, col) in enumerate(kpi_data):
        col_idx = i % 2
        row_idx = i // 2
        left_pos = Inches(0.5 + col_idx * 6.3)
        top_pos = Inches(1.22 + row_idx * 2.88)

        box = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, top_pos, Inches(6.033), Inches(2.65))
        box.adjustments[0] = 0.04
        box.fill.solid()
        box.fill.fore_color.rgb = WHITE
        box.line.color.rgb = col
        box.line.width = Pt(2.0)

        tb = slide3.shapes.add_textbox(left_pos + Inches(0.25), top_pos + Inches(0.20), Inches(5.533), Inches(2.25))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = big_num
        p1.font.name = FONT_MAIN
        p1.font.size = Pt(42)
        p1.font.bold = True
        p1.font.color.rgb = col
        p1.space_after = Pt(4)

        p2 = tf.add_paragraph()
        p2.text = kpi_title
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(19)
        p2.font.bold = True
        p2.font.color.rgb = NAVY_DARK
        p2.space_after = Pt(6)

        p3 = tf.add_paragraph()
        p3.alignment = PP_ALIGN.JUSTIFY
        p3.text = kpi_sub
        p3.font.name = FONT_MAIN
        p3.font.size = Pt(16)
        p3.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 4: INTRODUCCIÓN — CONTEXTO EPIDEMIOLÓGICO Y TAMIZAJE
    # =========================================================================
    slide4, s_num = next_slide()
    add_header(slide4, "Introducción: Magnitud Epidemiológica y Tamizaje", "2. INTRODUCCIÓN Y MOTIVACIÓN CLÍNICA (SECCIÓN 1)", s_num)

    add_card(
        slide4, Inches(0.5), Inches(1.18), Inches(6.0), Inches(5.75),
        "Impacto Global y Realidad en Chile",
        [
            ("Carga global (GLOBOCAN):", ">600.000 nuevos casos y ~340.000 muertes anuales; 90% en países de ingresos bajos y medios."),
            ("Impacto en Chile (DEIS):", "Incidencia de 9,1 por 100.000 mujeres y ~550 defunciones anuales."),
            ("Factor etiológico:", "Infección persistente por VPH de alto riesgo (genotipos 16 y 18)."),
            ("Ventana de oportunidad:", "Progresión lenta (10 a 20 años) que permite el diagnóstico y tratamiento temprano.")
        ],
        accent_color=NAVY_DARK, title_size=20, body_size=16, space_after=12
    )

    add_card(
        slide4, Inches(6.8), Inches(1.18), Inches(6.033), Inches(5.75),
        "Limitaciones del Papanicolaou Convencional",
        [
            ("Sensibilidad limitada:", "Rango de 51% a 95% en revisiones sistemáticas (promedio habitual de 70%–80%)."),
            ("Alta dependencia experta:", "Exige citopatólogos especializados para evaluar alteraciones morfológicas sutiles."),
            ("Sesgo humano:", "Fatiga visual, subjetividad y variabilidad interobservador incrementan los falsos negativos."),
            ("Brecha territorial:", "Déficit crítico de especialistas en zonas rurales y extremas de Chile.")
        ],
        accent_color=CORAL_ALERT, title_size=20, body_size=16, space_after=12
    )

    # =========================================================================
    # SLIDE 5: INTRODUCCIÓN — OPORTUNIDADES Y BARRERAS DE LA IA
    # =========================================================================
    slide5, s_num = next_slide()
    add_header(slide5, "Introducción: Potencial del ML y Barreras Clínicas", "2. INTRODUCCIÓN Y DESAFÍOS DE ADOPCIÓN (SECCIÓN 1)", s_num)

    add_card(
        slide5, Inches(0.5), Inches(1.18), Inches(5.9), Inches(5.75),
        "Oportunidades de la IA en Tamizaje",
        [
            ("Redes Convolucionales (CNN):", "Desempeño comparable a expertos en análisis de colposcopía y citología cervical."),
            ("Modelos tabulares clásicos:", "Random Forest, SVM y XGBoost destacan en la perfilación de factores de riesgo clínicos."),
            ("Sinergia diagnóstica:", "Automatización escalable para priorizar casos de alto riesgo y reducir tiempos de respuesta.")
        ],
        accent_color=TEAL_ACCENT, title_size=20, body_size=16.5, space_after=14
    )

    add_card(
        slide5, Inches(6.7), Inches(1.18), Inches(6.133), Inches(5.75),
        "Barreras para la Adopción Médica",
        [
            ("Efecto «Caja Negra»:", "La opacidad algorítmica frena la confianza médica; se requieren explicaciones accionables."),
            ("Domain Shift:", "Caída de rendimiento al pasar de entornos controlados a muestras clínicas heterogéneas."),
            ("Resiliencia operativa:", "Limitaciones de cómputo y conectividad; el aprendizaje federado surge como solución distribuida.")
        ],
        accent_color=GOLD_ACCENT, title_size=20, body_size=16.5, space_after=14
    )

    # =========================================================================
    # SLIDE 6: OBJETIVOS DEL ESTUDIO Y ESTRUCTURA
    # =========================================================================
    slide6, s_num = next_slide()
    add_header(slide6, "Objetivos del Estudio", "2. PROPÓSITO Y OBJETIVOS ESPECÍFICOS (SECCIÓN 1)", s_num)

    banner = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.15), Inches(12.333), Inches(1.12))
    banner.adjustments[0] = 0.06
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY_DARK
    banner.line.fill.background()

    tb_b = slide6.shapes.add_textbox(Inches(0.75), Inches(1.26), Inches(11.85), Inches(0.9))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    pb = tf_b.paragraphs[0]
    pb.alignment = PP_ALIGN.JUSTIFY
    rb1 = pb.add_run()
    rb1.text = "OBJETIVO GENERAL: "
    rb1.font.name = FONT_MAIN
    rb1.font.size = Pt(16.5)
    rb1.font.bold = True
    rb1.font.color.rgb = GOLD_ACCENT

    rb2 = pb.add_run()
    rb2.text = "Comparar sistemáticamente métodos de Machine Learning en la detección temprana de CaCu, integrando rendimiento predictivo y explicabilidad clínica (XAI)."
    rb2.font.name = FONT_MAIN
    rb2.font.size = Pt(16.5)
    rb2.font.color.rgb = WHITE

    objs = [
        ("Objetivo (i): Marco Teórico",
         [("Enfoque integrador:", "Articular fundamentos biológicos (Bethesda, OMS), algoritmos ML/DL y principios de XAI.")],
         NAVY_MED, Inches(0.5), Inches(2.48)),
        ("Objetivo (ii): Brechas del Arte",
         [("Revisión crítica:", "Detectar vacíos en validación cuantitativa de XAI y estandarización experimental.")],
         TEAL_ACCENT, Inches(6.8), Inches(2.48)),
        ("Objetivo (iii): Evaluación Empírica",
         [("Benchmark unificado:", "Comparar modelos sobre datasets públicos (UCI, Herlev, SIPaKMeD y RIVA) bajo métricas comunes.")],
         GREEN_OK, Inches(0.5), Inches(4.78)),
        ("Objetivo (iv): Auditoría XAI",
         [("Evaluación estandarizada:", "Analizar transparencia, interpretabilidad y auditabilidad mediante SHAP y Grad-CAM.")],
         CORAL_ALERT, Inches(6.8), Inches(4.78)),
    ]

    for title, bullets, col, l_pos, t_pos in objs:
        add_card(slide6, l_pos, t_pos, Inches(6.033), Inches(2.12), title, bullets, accent_color=col, title_size=18.5, body_size=16, space_after=6)

    # =========================================================================
    # SLIDE 7: MARCO TEÓRICO (PARTE 1: BIOLOGÍA Y MODELOS ML/DL)
    # =========================================================================
    slide7, s_num = next_slide()
    add_header(slide7, "Marco Teórico: Biología Cervical y Modelos ML/DL", "3. MARCO TEÓRICO (SECCIONES 2.1 Y 2.2)", s_num)

    add_card(
        slide7, Inches(0.5), Inches(1.18), Inches(6.0), Inches(5.75),
        "2.1. Historia Natural y Sistema Bethesda",
        [
            ("Evolución histológica:", "Infección persistente por VPH evoluciona hacia neoplasia intraepitelial (NIC) y cáncer invasor."),
            ("Sistema Bethesda:", "Clasifica citologías en NILM (negativo), ASC-US, LSIL (bajo grado) y HSIL (alto grado)."),
            ("Estándar de datos:", "Estructura las etiquetas de entrenamiento en repositorios como RIVA."),
            ("Estrategia OMS:", "Impulsa tecnologías diagnósticas escalables para acelerar la erradicación del CaCu.")
        ],
        accent_color=NAVY_DARK, title_size=19.5, body_size=16, space_after=11
    )

    add_card(
        slide7, Inches(6.8), Inches(1.18), Inches(6.033), Inches(5.75),
        "2.2. Algoritmos Biomédicos y Balanceo",
        [
            ("Aprendizaje clásico:", "SVM (hiperplano óptimo), Random Forest (bagging de árboles) y XGBoost (boosting regularizado)."),
            ("Transfer Learning en CNN:", "Extracción jerárquica automática con MobileNet, InceptionV3 y ResNet."),
            ("Segmentación celular:", "Arquitectura U-Net delimita núcleos y lesiones con pocas muestras."),
            ("Control de desbalance:", "Sobremuestreo sintético SMOTE mitiga el sesgo hacia la clase mayoritaria sana.")
        ],
        accent_color=TEAL_ACCENT, title_size=19.5, body_size=16, space_after=11
    )

    # =========================================================================
    # SLIDE 8: MARCO TEÓRICO (PARTE 2: FUNDAMENTOS DE XAI)
    # =========================================================================
    slide8, s_num = next_slide()
    add_header(slide8, "Marco Teórico: Inteligencia Artificial Explicable (XAI)", "3. MARCO TEÓRICO (SECCIÓN 2.3)", s_num)

    add_card(
        slide8, Inches(0.5), Inches(1.18), Inches(5.9), Inches(5.75),
        "Taxonomía de Explicabilidad Algorítmica",
        [
            ("Objetivo de XAI:", "Traducir el comportamiento interno de modelos complejos en conocimiento comprensible para el humano."),
            ("Modelos intrínsecos:", "Transparentes por diseño estructural (ej. Regresión Logística y Árboles de Decisión)."),
            ("Métodos post-hoc:", "Auditan modelos de «caja negra» ya entrenados sin alterar su arquitectura ni precisión.")
        ],
        accent_color=NAVY_MED, title_size=20, body_size=16.5, space_after=14
    )

    add_card(
        slide8, Inches(6.7), Inches(1.18), Inches(6.133), Inches(5.75),
        "Herramientas Post-Hoc y Criterio Médico",
        [
            ("SHAP:", "Cuantifica el aporte marginal de cada factor clínico mediante teoría de juegos cooperativos."),
            ("LIME:", "Aproxima localmente una predicción compleja con un modelo lineal interpretable."),
            ("Grad-CAM:", "Proyecta mapas térmicos sobre las regiones celulares que activan la red convolucional."),
            ("Requisito clínico:", "Explicaciones alineadas con la semiología médica y el razonamiento diagnóstico.")
        ],
        accent_color=GOLD_ACCENT, title_size=20, body_size=16, space_after=11
    )

    # =========================================================================
    # SLIDE 9: ESTADO DEL ARTE (SECCIONES 3.1 A 3.4)
    # =========================================================================
    slide9, s_num = next_slide()
    add_header(slide9, "Estado del Arte: Evidencia en Tabulares, Imágenes y XAI", "4. REVISIÓN DE LA LITERATURA (SECCIONES 3.1 – 3.4)", s_num)

    add_card(
        slide9, Inches(0.5), Inches(1.18), Inches(6.0), Inches(5.75),
        "3.1. Datos Tabulares y 3.3. AutoML",
        [
            ("Benchmark en UCI (858 casos):", "Literatura reporta hasta 97,99% de exactitud con Random Forest (AUC 99,74%) y 97,01% con XGBoost."),
            ("Meta-análisis recientes:", "Ramos-Casallas et al. y Kondo et al. evidencian limitaciones de diversidad y validación externa."),
            ("Hibridación AutoML + XAI:", "FSAE y H2O AutoML integrados con SHAP/LIME mejoran la interpretabilidad de factores de riesgo.")
        ],
        accent_color=NAVY_DARK, title_size=19.5, body_size=16, space_after=12
    )

    add_card(
        slide9, Inches(6.8), Inches(1.18), Inches(6.033), Inches(5.75),
        "3.2. Imágenes Citológicas y 3.4. XAI en CaCu",
        [
            ("Evolución de datasets:", "Desde células aisladas (Herlev, SIPaKMeD) hacia frotis completos anotados (RIVA, BiCervi, WSI)."),
            ("Desempeño CNN y U-Net:", "MobileNet alcanza AUC 0,97, InceptionV3 AUC 0,947 y U-Net supera Dice > 0,85."),
            ("Déficit de validación XAI:", "A pesar del uso de Grad-CAM, menos del 5% de los estudios validan sus explicaciones con médicos.")
        ],
        accent_color=TEAL_ACCENT, title_size=19.5, body_size=16, space_after=12
    )

    # =========================================================================
    # SLIDE 10: BRECHAS IDENTIFICADAS EN EL ESTADO DEL ARTE (SECCIÓN 3.5)
    # =========================================================================
    slide10, s_num = next_slide()
    add_header(slide10, "Brechas Identificadas en el Estado del Arte", "4. MOTIVACIÓN EXPERIMENTAL (SECCIÓN 3.5)", s_num)

    add_card(
        slide10, Inches(0.5), Inches(1.15), Inches(6.0), Inches(3.25),
        "BRECHA 1: Validación Cuantitativa y Clínica de XAI",
        [
            ("Carencia métrica:", "Predominio de visualizaciones cualitativas sin evaluación cuantitativa rigurosa de fidelidad."),
            ("Brecha práctica:", "Ausencia de validación directa sobre la utilidad diagnóstica real para el especialista médico.")
        ],
        accent_color=CORAL_ALERT, title_size=18.5, body_size=15.5, space_after=8
    )

    add_card(
        slide10, Inches(6.8), Inches(1.15), Inches(6.033), Inches(3.25),
        "BRECHA 2: Comparación Experimental Unificada",
        [
            ("Fragmentación metodológica:", "Estudios previos evalúan modelos con particiones, preprocesamientos y métricas dispares."),
            ("Necesidad de estandarización:", "Un protocolo unificado permite exponer sesgos críticos como la «Paradoja del Accuracy».")
        ],
        accent_color=TEAL_ACCENT, title_size=18.5, body_size=15.5, space_after=8
    )

    add_card(
        slide10, Inches(0.5), Inches(4.55), Inches(12.333), Inches(2.42),
        "Aporte del Presente Estudio frente a las Brechas",
        [
            ("1. Control de sesgos:", "Eliminación de fuga de datos (data leakage) y balanceo estricto en entrenamiento."),
            ("2. Contraste arquitectónico:", "Evaluación cruzada de 4 arquitecturas CNN en citología unicelular y frotis multicelular."),
            ("3. Doble auditoría XAI:", "Integración práctica de atribución epidemiológica (SHAP) y saliencia morfológica (Grad-CAM).")
        ],
        accent_color=NAVY_DARK, title_size=18.5, body_size=15, space_after=5
    )

    # =========================================================================
    # SLIDE 11: METODOLOGÍA — CONJUNTOS DE DATOS PÚBLICOS (TABLA 1)
    # =========================================================================
    slide11, s_num = next_slide()
    add_header(slide11, "Metodología: Conjuntos de Datos Públicos (Tabla 1)", "5. METODOLOGÍA Y FUENTES DE DATOS (SECCIÓN 4.1)", s_num)

    t1_shape = slide11.shapes.add_table(5, 5, Inches(0.5), Inches(1.18), Inches(12.333), Inches(3.95))
    t1 = t1_shape.table
    headers_t1 = ["Dataset", "Tipo", "Tamaño / Clases", "Descripción", "Acceso"]
    rows_t1 = [
        ["UCI Cervical Cancer Risk Factors", "Tabular", "858 registros, 36 atributos;\n2 clases (CaCu / No CaCu)", "Datos demográficos, conductuales y clínicos de pacientes del Hospital Universitario de Caracas", "UCI ML Repository"],
        ["Herlev", "Imágenes", "917 imágenes de células aisladas;\n7 categorías", "Imágenes de frotis de Pap con anotaciones para diferentes etapas de lesiones preneoplásicas", "MDE-Lab"],
        ["SIPaKMeD", "Imágenes", "966 parches, 4.049 células;\n5 categorías", "Imágenes de células cervicales con anotaciones de segmentación", "Público"],
        ["RIVA", "Imágenes", "959 imágenes (1024×1024 px);\n8 categorías (Bethesda)", "Frotis convencional con anotaciones de hasta 4 expertos; 26.158 anotaciones", "Nature Scientific Data"]
    ]
    style_table(
        t1,
        [Inches(2.6), Inches(1.25), Inches(2.65), Inches(4.033), Inches(1.8)],
        headers_t1,
        rows_t1,
        font_size=13.5,
        header_size=14,
        justify_cols=[3]
    )

    add_card(
        slide11, Inches(0.5), Inches(5.30), Inches(12.333), Inches(1.65),
        "Fuentes de Validación Complementaria",
        [
            ("Proyección de generalización:", "Se contemplan repositorios recientes como BiCervi y láminas histopatológicas completas (WSI) para evaluar robustez externa frente al domain shift.")
        ],
        accent_color=TEAL_ACCENT, title_size=18.5, body_size=16, space_after=6
    )

    # =========================================================================
    # SLIDE 12: METODOLOGÍA — ALGORITMOS Y PROTOCOLO EXPERIMENTAL
    # =========================================================================
    slide12, s_num = next_slide()
    add_header(slide12, "Metodología: Algoritmos y Protocolo Experimental", "5. ALGORITMOS, MÉTRICAS Y PREPROCESAMIENTO (SECCIONES 4.2 – 4.4)", s_num)

    add_card(
        slide12, Inches(0.5), Inches(1.18), Inches(6.0), Inches(5.75),
        "4.2. Algoritmos Seleccionados",
        [
            ("Regresión Logística (RL):", "Referencia lineal interpretable (max_iter=1000)."),
            ("SVM:", "Kernel radial (kernel='rbf', probability=True)."),
            ("Decision Tree (DT):", "Árbol interpretable por reglas explícitas."),
            ("Random Forest (RF):", "Ensamble robusto (n_estimators=100)."),
            ("XGBoost (XGB):", "Gradient boosting (eval_metric='logloss')."),
            ("Arquitecturas CNN:", "Transfer learning con AlexNet, MobileNet, InceptionV3 y ResNet.")
        ],
        accent_color=NAVY_DARK, title_size=19.5, body_size=15.5, space_after=8
    )

    add_card(
        slide12, Inches(6.8), Inches(1.18), Inches(6.033), Inches(5.75),
        "4.3 – 4.4. Pipeline Experimental y Métricas",
        [
            ("Prevención de Data Leakage:", "Exclusión de pruebas diagnósticas intermedias ('Hinselmann', 'Schiller', 'Citology')."),
            ("Normalización y Balanceo:", "Imputación por mediana (SimpleImputer), escalado (StandardScaler) y SMOTE solo en entrenamiento."),
            ("Validación Reproducible:", "Hold-out estratificado (15% prueba, stratify=y, random_state=42)."),
            ("Métricas Evaluadas:", "Accuracy, Sensibilidad (Recall), Especificidad, Precisión, F1-Score y AUC-ROC.")
        ],
        accent_color=TEAL_ACCENT, title_size=19.5, body_size=15.5, space_after=10
    )

    # =========================================================================
    # SLIDE 13: RESULTADOS EMPÍRICOS — DATOS TABULARES (TABLA 2a)
    # =========================================================================
    slide13, s_num = next_slide()
    add_header(slide13, "Resultados Empíricos en Datos Tabulares (Tabla 2a)", "6. RESULTADOS EXPERIMENTALES — MODALIDAD TABULAR (SECCIÓN 4.3 – 4.4)", s_num)

    t2a_shape = slide13.shapes.add_table(6, 7, Inches(0.5), Inches(1.15), Inches(12.333), Inches(2.70))
    t2a = t2a_shape.table
    headers_t2a = ["Algoritmo", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score", "AUC-ROC"]
    rows_t2a = [
        ["Regresión Logística", "79.84%", "12.50%", "84.30%", "5.00%", "7.14%", "50.93%"],
        ["SVM", "89.15%", "0.00%", "95.04%", "0.00%", "0.00%", "65.60%"],
        ["Decision Tree", "87.60%", "25.00%", "91.74%", "16.67%", "20.00%", "58.06%"],
        ["Random Forest", "92.25%", "12.50%", "97.52%", "25.00%", "16.67%", "76.45%"],
        ["XGBoost", "93.80%", "12.50%", "99.17%", "50.00%", "20.00%", "75.41%"]
    ]
    style_table(
        t2a,
        [Inches(2.833), Inches(1.58), Inches(1.58), Inches(1.58), Inches(1.58), Inches(1.58), Inches(1.60)],
        headers_t2a,
        rows_t2a,
        highlight_rows={3: RGBColor(254, 243, 199), 4: RGBColor(254, 226, 226)},
        font_size=14.5,
        header_size=15
    )

    add_card(
        slide13, Inches(0.5), Inches(4.02), Inches(12.333), Inches(2.92),
        "Hallazgos en Datos Clínicos: La «Paradoja del Accuracy»",
        [
            ("Ilusión de exactitud:", "XGBoost alcanza 93.80% de exactitud y 99.17% de especificidad, pero apenas 12.50% de sensibilidad."),
            ("Peligro de falsos negativos:", "El modelo clasifica correctamente a las pacientes sanas, pero omite a la mayoría de las pacientes con cáncer."),
            ("Sesgo hacia clase mayoritaria:", "SVM (0.00% sensibilidad) y Random Forest (12.50% sensibilidad, 25.00% precisión) priorizan la clase negativa bajo el umbral por defecto.")
        ],
        accent_color=CORAL_ALERT, title_size=18.5, body_size=15.5, space_after=6
    )

    # =========================================================================
    # SLIDE 14: CURVAS ROC COMPARATIVAS EN DATOS TABULARES (FIGURA 3)
    # =========================================================================
    slide14, s_num = next_slide()
    add_header(slide14, "Capacidad Discriminativa en Datos Tabulares (Curvas ROC)", "6. ANÁLISIS DE DISCRIMINACIÓN Y CALIBRACIÓN DE UMBRALES", s_num)

    fig3_path = "figura3_curvas_roc_comparativas.png"
    if os.path.exists(fig3_path):
        img_card = slide14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.18), Inches(6.6), Inches(5.75))
        img_card.adjustments[0] = 0.03
        img_card.fill.solid()
        img_card.fill.fore_color.rgb = WHITE
        img_card.line.color.rgb = BORDER_GRAY
        slide14.shapes.add_picture(fig3_path, Inches(0.68), Inches(1.30), width=Inches(6.24))

    add_card(
        slide14, Inches(7.35), Inches(1.18), Inches(5.483), Inches(5.75),
        "Lectura Clínica de las Curvas ROC",
        [
            ("Aprendizaje latente real:", "Random Forest (AUC 76.45%) y XGBoost (AUC 75.41%) logran separar los perfiles de riesgo muy por encima del azar (50%)."),
            ("Origen del bajo Recall:", "El umbral estándar (p = 0.50) penaliza la clase minoritaria ante el fuerte desbalance en prueba."),
            (" Solución clínica (Thresholding):", "Reducir el umbral de decisión permite incrementar drásticamente la sensibilidad asumiendo falsos positivos controlados.")
        ],
        accent_color=NAVY_MED, title_size=19.5, body_size=16, space_after=14
    )

    # =========================================================================
    # SLIDE 15: RESULTADOS EMPÍRICOS — IMÁGENES CITOLÓGICAS (TABLA 2b)
    # =========================================================================
    slide15, s_num = next_slide()
    add_header(slide15, "Resultados Empíricos en Imágenes Citológicas (Tabla 2b)", "6. RESULTADOS EXPERIMENTALES — VISIÓN POR COMPUTADORA (TRANSFER LEARNING)", s_num)

    t2b_shape = slide15.shapes.add_table(13, 8, Inches(0.5), Inches(1.15), Inches(12.333), Inches(5.82))
    t2b = t2b_shape.table
    headers_t2b = ["Algoritmo", "Dataset", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score", "AUC-ROC"]
    rows_t2b = [
        ["AlexNet", "Herlev (7 clases)", "45.26%", "45.96%", "90.30%", "43.30%", "42.42%", "80.39%"],
        ["MobileNet", "Herlev (7 clases)", "42.70%", "47.86%", "90.18%", "50.66%", "47.76%", "81.66%"],
        ["InceptionV3", "Herlev (7 clases)", "40.15%", "43.71%", "89.61%", "45.27%", "43.73%", "79.34%"],
        ["ResNet", "Herlev (7 clases)", "16.79%", "11.34%", "84.97%", "4.31%", "6.11%", "57.98%"],
        ["AlexNet ★", "SIPaKMeD (5 clases)", "93.63%", "93.99%", "98.40%", "93.79%", "93.76%", "99.65%"],
        ["MobileNet", "SIPaKMeD (5 clases)", "49.35%", "49.42%", "87.19%", "66.98%", "49.93%", "81.29%"],
        ["InceptionV3", "SIPaKMeD (5 clases)", "41.86%", "41.65%", "85.31%", "55.46%", "41.84%", "73.28%"],
        ["ResNet", "SIPaKMeD (5 clases)", "33.17%", "33.49%", "83.14%", "16.62%", "20.85%", "60.33%"],
        ["AlexNet ★", "RIVA (8 clases Bethesda)", "59.20%", "50.20%", "93.49%", "53.82%", "49.80%", "88.90%"],
        ["MobileNet", "RIVA (8 clases Bethesda)", "50.40%", "38.89%", "92.11%", "45.40%", "38.06%", "83.56%"],
        ["InceptionV3", "RIVA (8 clases Bethesda)", "46.99%", "34.85%", "91.58%", "39.06%", "33.44%", "81.16%"],
        ["ResNet", "RIVA (8 clases Bethesda)", "32.89%", "15.54%", "88.31%", "19.33%", "12.28%", "66.83%"],
    ]
    style_table(
        t2b,
        [Inches(1.75), Inches(2.55), Inches(1.32), Inches(1.35), Inches(1.35), Inches(1.32), Inches(1.32), Inches(1.373)],
        headers_t2b,
        rows_t2b,
        highlight_rows={
            4: RGBColor(209, 250, 229),  # AlexNet SIPaKMeD
            8: RGBColor(209, 250, 229),  # AlexNet RIVA
            3: RGBColor(254, 226, 226),  # ResNet Herlev
            7: RGBColor(254, 226, 226),  # ResNet SIPaKMeD
            11: RGBColor(254, 226, 226)  # ResNet RIVA
        },
        font_size=13.5,
        header_size=14
    )

    # =========================================================================
    # SLIDE 16: ANÁLISIS DE VISIÓN POR COMPUTADORA Y RECOMENDACIONES (SECCIÓN 4.4)
    # =========================================================================
    slide16, s_num = next_slide()
    add_header(slide16, "Análisis en Visión por Computadora y Recomendaciones", "6. DISCUSIÓN DE RESULTADOS EN IMÁGENES CITOLÓGICAS (SECCIÓN 4.4)", s_num)

    add_card(
        slide16, Inches(0.5), Inches(1.18), Inches(6.0), Inches(3.60),
        "1. Liderazgo de AlexNet (PyTorch)",
        [
            ("Máximo desempeño:", "Logró 93.63% de exactitud (AUC 99.65%) en SIPaKMeD, 59.20% (AUC 88.90%) en RIVA y 45.26% (AUC 80.39%) en Herlev."),
            ("Ventaja estructural:", "Su menor profundidad evita el sobreajuste y facilita la convergencia morfológica.")
        ],
        accent_color=GREEN_OK, title_size=19, body_size=15.5, space_after=10
    )

    add_card(
        slide16, Inches(6.8), Inches(1.18), Inches(6.033), Inches(3.60),
        "2. Modelos en Keras y Colapso de ResNet50",
        [
            ("Estancamiento moderado:", "MobileNet (42.70%–50.40%) e InceptionV3 (40.15%–46.99%) se ven limitados al entrenar solo el cabezal denso."),
            ("Falla en ResNet50:", "Detenida en la época 6 por Early Stopping (16.79%–33.17%); congelar capas profundas bloquea el aprendizaje celular.")
        ],
        accent_color=CORAL_ALERT, title_size=19, body_size=15.5, space_after=10
    )

    add_card(
        slide16, Inches(0.5), Inches(4.95), Inches(12.333), Inches(2.00),
        "3. Recomendaciones Técnicas Clave",
        [
            ("Calibración de umbrales (Tabular):", "Aprovechar el AUC-ROC de RF (76.45%) y XGBoost (75.41%) ajustando el corte de probabilidad para elevar la sensibilidad."),
            ("Fine-Tuning progresivo (Imágenes):", "Descongelar los bloques convolucionales superiores en MobileNet e InceptionV3 para adaptar los filtros al dominio citológico.")
        ],
        accent_color=NAVY_DARK, title_size=18.5, body_size=15.5, space_after=6
    )

    # =========================================================================
    # SLIDE 17: ANÁLISIS DE EXPLICABILIDAD DE LOS ALGORITMOS (TABLA 3)
    # =========================================================================
    slide17, s_num = next_slide()
    add_header(slide17, "Evaluación de Explicabilidad de los Algoritmos (Tabla 3)", "7. DIMENSIONES Y COMPARACIÓN CUALITATIVA DE XAI (SECCIONES 5.1 – 5.2)", s_num)

    dims = [
        ("Transparencia", "Claridad estructural e inspección directa de parámetros internos.", NAVY_MED, Inches(0.5)),
        ("Interpretabilidad", "Capacidad de explicar predicciones de forma global o local.", TEAL_ACCENT, Inches(4.66)),
        ("Auditabilidad", "Verificación por terceros y trazabilidad post-hoc (SHAP, LIME, CAM).", GOLD_ACCENT, Inches(8.82)),
    ]
    for d_title, d_desc, d_col, l_pos in dims:
        add_card(slide17, l_pos, Inches(1.15), Inches(4.01), Inches(1.65), d_title, [d_desc], accent_color=d_col, title_size=18, body_size=14.5, space_after=4)

    t3_shape = slide17.shapes.add_table(8, 5, Inches(0.5), Inches(2.95), Inches(12.333), Inches(4.02))
    t3 = t3_shape.table
    headers_t3 = ["Algoritmo", "Transparencia", "Interpretabilidad", "Auditabilidad", "Explicabilidad Global"]
    rows_t3 = [
        ["Regresión Logística", "Alta", "Alta (coeficientes)", "Alta", "Alta"],
        ["Decision Tree", "Alta", "Alta (reglas)", "Alta", "Alta"],
        ["SVM (kernel lineal)", "Media", "Media (weights)", "Media", "Media-Alta"],
        ["SVM (kernel RBF)", "Baja", "Baja (post-hoc)", "Media", "Media-Baja"],
        ["Random Forest", "Media", "Media (feature importance)", "Media", "Media"],
        ["XGBoost", "Media", "Media (feature importance, SHAP)", "Media-Alta", "Media-Alta"],
        ["CNN (MobileNet, etc.)", "Baja", "Baja-Media (Grad-CAM, LIME)", "Media", "Media-Baja"],
    ]
    style_table(
        t3,
        [Inches(2.8), Inches(1.95), Inches(3.383), Inches(2.1), Inches(2.1)],
        headers_t3,
        rows_t3,
        highlight_rows={5: RGBColor(235, 248, 255), 6: RGBColor(254, 243, 199)},
        font_size=14,
        header_size=14.5
    )

    # =========================================================================
    # SLIDE 18: AUDITABILIDAD EN DATOS TABULARES — SHAP (FIGURA 1)
    # =========================================================================
    slide18, s_num = next_slide()
    add_header(slide18, "Auditabilidad en Datos Tabulares: SHAP sobre XGBoost", "7. VALIDACIÓN ANALÍTICA POST-HOC (SECCIONES 5.3 – 5.4)", s_num)

    add_card(
        slide18, Inches(0.5), Inches(1.12), Inches(12.333), Inches(1.38),
        "Factores de Mayor Contribución (TreeSHAP) y Selección Clínica",
        [
            ("Top-5 concentra el 75.62% (log-odds):", "1) Embarazos previos (18.06%), 2) Parejas sexuales (17.00%), 3) Inicio sexual (15.67%), 4) Anticonceptivos hormonales (12.98%) y 5) Edad (11.91%) — coherente con exposición a VPH."),
            ("Criterio de selección:", "XGBoost + SHAP es prioritario para tamizaje poblacional; modelos intrínsecos (RL, Árboles) facilitan el apoyo individual inmediato.")
        ],
        accent_color=NAVY_MED, title_size=17.5, body_size=14.5, space_after=3
    )

    fig1_path = "figura1_importancia_atributos.png"
    if os.path.exists(fig1_path):
        f1_card = slide18.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(2.58), Inches(12.333), Inches(4.42))
        f1_card.adjustments[0] = 0.02
        f1_card.fill.solid()
        f1_card.fill.fore_color.rgb = WHITE
        f1_card.line.color.rgb = BORDER_GRAY
        # Proporción 2.416: alto 4.26" -> ancho 10.29" (centrado en la diapositiva)
        slide18.shapes.add_picture(fig1_path, Inches(1.52), Inches(2.66), height=Inches(4.26))

    # =========================================================================
    # SLIDE 19: AUDITABILIDAD EN IMÁGENES CITOLÓGICAS — GRAD-CAM (FIGURA 2)
    # =========================================================================
    slide19, s_num = next_slide()
    add_header(slide19, "Auditabilidad Visual en Citología: Mapas de Activación", "7. VALIDACIÓN VISUAL EN VISIÓN POR COMPUTADORA (SECCIONES 5.3 – 5.4)", s_num)

    add_card(
        slide19, Inches(0.5), Inches(1.12), Inches(12.333), Inches(1.38),
        "Verificación Morfológica en Citología (SIPaKMeD) mediante Grad-CAM",
        [
            ("Atención anatómica precisa:", "Las activaciones de la red se focalizan sobre núcleos hipercromáticos y bordes citoplasmáticos alterados, descartando artefactos de fondo o sesgos de tinción."),
            ("Respaldo al citopatólogo:", "Valida visualmente que la alta exactitud (>93%) obedece a rasgos patológicos reales, transformando la predicción en evidencia auditable.")
        ],
        accent_color=TEAL_ACCENT, title_size=17.5, body_size=14.5, space_after=3
    )

    fig2_path = "figura2_explicabilidad_visual_xai.png"
    if os.path.exists(fig2_path):
        f2_card = slide19.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(2.58), Inches(12.333), Inches(4.42))
        f2_card.adjustments[0] = 0.02
        f2_card.fill.solid()
        f2_card.fill.fore_color.rgb = WHITE
        f2_card.line.color.rgb = BORDER_GRAY
        # Proporción 2.831: ancho 12.05" -> alto 4.26" (a ancho completo)
        slide19.shapes.add_picture(fig2_path, Inches(0.64), Inches(2.66), width=Inches(12.05))

    # =========================================================================
    # SLIDE 20: CONCLUSIONES, TRABAJO FUTURO Y AGRADECIMIENTOS
    # =========================================================================
    slide20, s_num = next_slide()
    add_header(slide20, "Conclusiones, Trabajo Futuro y Agradecimientos", "8. SÍNTESIS FINAL Y PROYECCIÓN (SECCIÓN 6)", s_num)

    add_card(
        slide20, Inches(0.5), Inches(1.15), Inches(6.1), Inches(4.45),
        "Conclusiones Principales",
        [
            ("Paradoja tabular:", "XGBoost alcanza 93.80% de exactitud, pero requiere calibrar el umbral clínico para superar su 12.50% de sensibilidad."),
            ("Eficacia de CNN superficiales:", "AlexNet supera a redes profundas (ResNet) con AUC-ROC de 99.65% en SIPaKMeD y 88.90% en RIVA."),
            ("Imperativo de XAI:", "Con <5% de modelos validados clínicamente, integrar SHAP y Grad-CAM es indispensable para la adopción médica.")
        ],
        accent_color=NAVY_DARK, title_size=19, body_size=15.5, space_after=9
    )

    add_card(
        slide20, Inches(6.8), Inches(1.15), Inches(6.033), Inches(4.45),
        "Trabajo Futuro",
        [
            ("Métricas cuantitativas de XAI:", "Evaluar matemáticamente la fidelidad de las explicaciones y su impacto en la confianza médica."),
            ("Enfoque Human-in-the-Loop:", "Incorporar retroalimentación interactiva del especialista para auditar y refinar el modelo."),
            ("Aprendizaje Federado:", "Entrenamiento colaborativo distribuido para fortalecer la resiliencia operativa en salud pública.")
        ],
        accent_color=TEAL_ACCENT, title_size=19, body_size=15.5, space_after=9
    )

    ack_box = slide20.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(5.75), Inches(12.333), Inches(1.20))
    ack_box.adjustments[0] = 0.06
    ack_box.fill.solid()
    ack_box.fill.fore_color.rgb = NAVY_MED
    ack_box.line.fill.background()

    tb_ack = slide20.shapes.add_textbox(Inches(0.75), Inches(5.88), Inches(11.85), Inches(0.95))
    tf_ack = tb_ack.text_frame
    tf_ack.word_wrap = True
    pack = tf_ack.paragraphs[0]
    pack.alignment = PP_ALIGN.JUSTIFY
    rack1 = pack.add_run()
    rack1.text = "AGRADECIMIENTOS: "
    rack1.font.name = FONT_MAIN
    rack1.font.size = Pt(15)
    rack1.font.bold = True
    rack1.font.color.rgb = GOLD_ACCENT

    rack2 = pack.add_run()
    rack2.text = "Se agradece al proyecto «Formación de Capital Humano Avanzado en Inteligencia Artificial para la Revolución Digital de la Región de Antofagasta», financiado mediante el Fondo Regional para la Productividad y el Desarrollo, BIP 40074839-0."
    rack2.font.name = FONT_MAIN
    rack2.font.size = Pt(15)
    rack2.font.color.rgb = WHITE

    prs.save(output_path)
    print(f"Presentación generada exitosamente ({slide_counter} diapositivas) en: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    build_presentation()
