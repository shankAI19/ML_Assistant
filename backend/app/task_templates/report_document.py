"""Render readable report Markdown into a consistently styled Word document."""

from .paths import OUT


def add_inline(paragraph, token):
    bold = italic = False
    for child in token.children or []:
        if child.type == "strong_open":
            bold = True
        elif child.type == "strong_close":
            bold = False
        elif child.type == "em_open":
            italic = True
        elif child.type == "em_close":
            italic = False
        elif child.type in {"text", "code_inline"}:
            run = paragraph.add_run(child.content)
            run.bold, run.italic = bold, italic
            if child.type == "code_inline":
                run.font.name = "Consolas"
        elif child.type in {"softbreak", "hardbreak"}:
            paragraph.add_run("\n")


def add_table(document, tokens, usable_width):
    from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    rows, row = [], []
    for token in tokens:
        if token.type == "tr_open":
            row = []
        elif token.type == "inline":
            row.append(token)
        elif token.type == "tr_close":
            rows.append(row)
    if not rows:
        return
    count = len(rows[0])
    headers = [token.content for token in rows[0]]
    if count == 2:
        proportions = [0.42, 0.58]
    elif count == 3:
        proportions = [0.48, 0.26, 0.26]
    elif count == 4:
        proportions = (
            [0.25, 0.18, 0.17, 0.40] if headers[0] == "Partition" else [0.39, 0.18, 0.20, 0.23]
        )
    elif count == 5:
        proportions = (
            [0.27, 0.17, 0.18, 0.16, 0.22]
            if headers[0] == "Column"
            else [0.28, 0.13, 0.17, 0.20, 0.22]
        )
    else:
        proportions = [1 / count] * count
    table = document.add_table(rows=0, cols=count)
    table.alignment, table.autofit = WD_TABLE_ALIGNMENT.CENTER, False
    for column, proportion in zip(table.columns, proportions, strict=True):
        column.width = int(usable_width * proportion)
    properties = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement(f"w:{name}")
        for key, value in {"val": "single", "sz": "4", "color": "D9D9D9"}.items():
            edge.set(qn(f"w:{key}"), value)
        borders.append(edge)
    properties.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for name in ("top", "left", "bottom", "right"):
        edge = OxmlElement(f"w:{name}")
        edge.set(qn("w:w"), "85")
        edge.set(qn("w:type"), "dxa")
        margins.append(edge)
    properties.append(margins)
    for index, values in enumerate(rows):
        output = table.add_row()
        row_properties = output._tr.get_or_add_trPr()
        row_properties.append(OxmlElement("w:cantSplit"))
        if index == 0:
            row_properties.append(OxmlElement("w:tblHeader"))
        for column_index, token in enumerate(values):
            cell = output.cells[column_index]
            cell.width = int(usable_width * proportions[column_index])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            shading = OxmlElement("w:shd")
            shading.set(
                qn("w:fill"), "233D4D" if index == 0 else "F1F5F7" if index % 2 == 0 else "FFFFFF"
            )
            cell._tc.get_or_add_tcPr().append(shading)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.space_before = Pt(2)
            paragraph.paragraph_format.line_spacing = 1.1
            paragraph.paragraph_format.keep_with_next = index == 0
            if column_index > 0 and len(token.content) < 24:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(paragraph, token)
            for run in paragraph.runs:
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor.from_string("FFFFFF" if index == 0 else "000000")
                if index == 0:
                    run.bold = True
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(3)
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.line_spacing = Pt(2)
    spacer.paragraph_format.keep_with_next = True


def to_docx(markdown, plots):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor
    from markdown_it import MarkdownIt

    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.left_margin = section.right_margin = Inches(0.85)
    section.top_margin = section.bottom_margin = Inches(0.75)
    usable_width = section.page_width - section.left_margin - section.right_margin
    for name, size in [
        ("Normal", 11),
        ("Title", 23),
        ("Heading 1", 15),
        ("Heading 2", 12),
        ("Heading 3", 11),
        ("Caption", 10),
    ]:
        style = document.styles[name]
        style.font.name, style.font.size = "Calibri", Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_after = Pt(6)
    document.styles["Normal"].paragraph_format.line_spacing = 1.12
    document.styles["Title"].paragraph_format.space_after = Pt(14)
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        document.styles[name].paragraph_format.space_before = Pt(12)
        document.styles[name].paragraph_format.keep_with_next = True
    tokens = MarkdownIt("commonmark").enable("table").parse(markdown)
    images = {plot["path"].name: plot for plot in plots}
    paragraph = None
    list_style = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        kind = token.type
        if kind == "table_open":
            end = next(i for i in range(index + 1, len(tokens)) if tokens[i].type == "table_close")
            add_table(document, tokens[index + 1 : end], usable_width)
            index = end
        elif kind == "heading_open":
            level = int(token.tag[1:])
            paragraph = (
                document.add_paragraph(style="Title")
                if level == 1
                else document.add_heading(level=min(level - 1, 3))
            )
        elif kind in {"bullet_list_open", "ordered_list_open"}:
            list_style = "List Bullet" if kind == "bullet_list_open" else "List Number"
        elif kind in {"bullet_list_close", "ordered_list_close"}:
            list_style = None
        elif kind == "paragraph_open":
            paragraph = document.add_paragraph(style=list_style)
        elif kind == "inline":
            pictures = [child for child in token.children or [] if child.type == "image"]
            if pictures:
                for picture in pictures:
                    plot = images.get(picture.attrGet("src"))
                    if plot:
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        paragraph.paragraph_format.keep_with_next = True
                        shape = paragraph.add_run().add_picture(
                            str(plot["path"]), width=usable_width
                        )
                        shape._inline.docPr.set("descr", plot["caption"])
            else:
                visible_text = "".join(
                    child.content
                    for child in token.children or []
                    if child.type in {"text", "code_inline"}
                )
                if visible_text.startswith("Figure "):
                    paragraph.style = document.styles["Caption"]
                add_inline(paragraph, token)
        elif kind in {"fence", "code_block"}:
            paragraph = document.add_paragraph()
            run = paragraph.add_run(token.content.rstrip())
            run.font.name, run.font.size = "Consolas", Pt(9)
        index += 1
    document.save(OUT / "report.docx")
