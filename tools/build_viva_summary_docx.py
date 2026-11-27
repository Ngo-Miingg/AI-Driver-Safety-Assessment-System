from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = Path(__file__).resolve().parents[1] / "docs" / "Tom_Tat_Van_Dap_AI_Driver_Safety.docx"

NAVY = "0B2545"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
MUTED = "566B84"
BORDER = "B8C7D9"
HEADER_FILL = "E8EEF5"
CALLOUT_FILL = "F4F6F9"
CAUTION_FILL = "FFF7E6"
GREEN_FILL = "EAF7EF"
RED_FILL = "FDECEC"
WHITE = "FFFFFF"
BLACK = "000000"


def set_run_font(run, name="Calibri", size=None, color=None, bold=None, italic=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def style_paragraph(paragraph, before=0, after=6, line=1.25, align=None):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line
    if align is not None:
        paragraph.alignment = align


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color=BORDER, size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_row_cant_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = tr_pr.find(qn("w:cantSplit"))
    if cant_split is None:
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)


def set_fixed_width(cell, width_in):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(width_in * 1440)))
    tc_w.set(qn("w:type"), "dxa")
    cell.width = Inches(width_in)


def write_cell(cell, text, bold=False, color=BLACK, size=10.2):
    cell.text = ""
    for i, line in enumerate(str(text).split("\n")):
        paragraph = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        style_paragraph(paragraph, after=3 if i < len(str(text).split("\n")) - 1 else 0, line=1.18)
        run = paragraph.add_run(line)
        set_run_font(run, size=size, color=color, bold=bold)


def add_heading(doc, text, level=1):
    paragraph = doc.add_paragraph()
    if level == 1:
        size, color, before, after = 16, BLUE, 18, 10
    elif level == 2:
        size, color, before, after = 13, BLUE, 14, 7
    else:
        size, color, before, after = 12, DARK_BLUE, 10, 5
    style_paragraph(paragraph, before=before, after=after)
    run = paragraph.add_run(text)
    set_run_font(run, size=size, color=color, bold=True)


def add_callout(doc, title, text, fill=CALLOUT_FILL):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    set_fixed_width(cell, 6.5)
    set_cell_shading(cell, fill)
    set_cell_border(cell)
    set_cell_margins(cell, top=120, bottom=120, start=160, end=160)
    paragraph = cell.paragraphs[0]
    style_paragraph(paragraph, after=3, line=1.2)
    run = paragraph.add_run(title)
    set_run_font(run, size=11, bold=True, color=NAVY)
    paragraph = cell.add_paragraph()
    style_paragraph(paragraph, after=0, line=1.2)
    run = paragraph.add_run(text)
    set_run_font(run, size=10.5, color=BLACK)


def add_two_col_table(
    doc,
    rows,
    left_header="Trong báo cáo / số liệu gốc",
    right_header="Cách nói ngắn khi vấn đáp",
    widths=(3.05, 3.45),
):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, header in enumerate((left_header, right_header)):
        set_row_cant_split(table.rows[0])
        cell = table.rows[0].cells[i]
        set_fixed_width(cell, widths[i])
        set_cell_shading(cell, HEADER_FILL)
        set_cell_border(cell)
        set_cell_margins(cell)
        write_cell(cell, header, bold=True, color=NAVY, size=10.5)

    for left, right, fill in rows:
        row = table.add_row()
        set_row_cant_split(row)
        cells = row.cells
        for i, text in enumerate((left, right)):
            cell = cells[i]
            set_fixed_width(cell, widths[i])
            set_cell_shading(cell, fill or WHITE)
            set_cell_border(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            write_cell(cell, text)

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def add_meta_table(doc):
    rows = [
        ("Đề tài", "PHÁT HIỆN BUỒN NGỦ VÀ GIẢM CHÚ Ý CỦA TÀI XẾ BẰNG MÔ HÌNH ĐA PHƯƠNG THỨC TRONG CABIN"),
        ("Nhóm", "Ngo Van Minh, Nguyen Thi Phuong - Group 01 - Dai Nam University"),
        ("Model nghiên cứu", "mmrt_multimodal_full_v2_t4x2"),
        ("Dataset", "DBR-DMD-v1.1 CLEAN_FULLMM_5CLASS, 5 lớp, 4 modality"),
        ("Metric chính", "Validation macro-F1 = 0.7129"),
        (
            "Vai trò MDDAD",
            "Application layer: phân tích video, giám sát, log sự kiện, risk/reporting; không phải benchmark chính của model.",
        ),
    ]
    table = doc.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for label, value in rows:
        row = table.add_row()
        set_row_cant_split(row)
        cells = row.cells
        set_fixed_width(cells[0], 1.55)
        set_fixed_width(cells[1], 4.95)
        set_cell_shading(cells[0], HEADER_FILL)
        set_cell_shading(cells[1], WHITE)
        for cell in cells:
            set_cell_border(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
        write_cell(cells[0], label, bold=True, color=NAVY, size=10.3)
        write_cell(cells[1], value, size=10.3)


def add_bullet(doc, text):
    paragraph = doc.add_paragraph()
    style_paragraph(paragraph, after=4)
    paragraph.paragraph_format.left_indent = Inches(0.375)
    paragraph.paragraph_format.first_line_indent = Inches(-0.188)
    run = paragraph.add_run("• ")
    set_run_font(run, size=10.5, bold=True, color=BLUE)
    run = paragraph.add_run(text)
    set_run_font(run, size=10.5, color=BLACK)


def build():
    doc = Document()
    section = doc.sections[0]
    section.start_type = WD_SECTION_START.NEW_PAGE
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    styles["Normal"]._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    styles["Normal"].font.size = Pt(11)
    styles["Normal"].font.color.rgb = RGBColor.from_string(BLACK)

    header = section.header.paragraphs[0]
    style_paragraph(header, after=0, line=1.0)
    run = header.add_run("Tóm tắt vấn đáp - MDDAD / Multimodal Driver Safety")
    set_run_font(run, size=9, color=MUTED, bold=True)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("Page ")
    set_run_font(run, size=9, color=MUTED)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)

    paragraph = doc.add_paragraph()
    style_paragraph(paragraph, after=4, line=1.15, align=WD_ALIGN_PARAGRAPH.CENTER)
    run = paragraph.add_run("TÓM TẮT VẤN ĐÁP ĐỒ ÁN")
    set_run_font(run, size=20, bold=True, color=NAVY)

    paragraph = doc.add_paragraph()
    style_paragraph(paragraph, after=6, line=1.15, align=WD_ALIGN_PARAGRAPH.CENTER)
    run = paragraph.add_run("PHÁT HIỆN BUỒN NGỦ VÀ GIẢM CHÚ Ý CỦA TÀI XẾ BẰNG MÔ HÌNH ĐA PHƯƠNG THỨC TRONG CABIN")
    set_run_font(run, size=13, bold=True, color=BLACK)

    paragraph = doc.add_paragraph()
    style_paragraph(paragraph, after=14, line=1.15, align=WD_ALIGN_PARAGRAPH.CENTER)
    run = paragraph.add_run("Ngo Van Minh, Nguyen Thi Phuong | Group 01 | Dai Nam University | Hanoi, Vietnam")
    set_run_font(run, size=10.5, color=MUTED)

    add_meta_table(doc)
    add_callout(
        doc,
        "Câu mở đầu nên nói trong 20 giây",
        "Đề tài của em/nhóm em tập trung vào nhận diện buồn ngủ và giảm chú ý của tài xế trong cabin bằng mô hình đa phương thức. Thay vì chỉ dùng một dấu hiệu như mắt hoặc điện thoại, mô hình nghiên cứu hợp nhất body, face, audio và skeleton theo thời gian. Checkpoint tốt nhất đạt validation macro-F1 = 0.7129 trên bộ DBR-DMD-v1.1 CLEAN_FULLMM_5CLASS. Hệ thống MDDAD là tầng ứng dụng để minh họa cách đưa kết quả nhận diện vào phân tích video, giám sát và báo cáo rủi ro.",
        GREEN_FILL,
    )

    add_heading(doc, "1. Tổng quan đề tài")
    add_two_col_table(doc, [
        (
            "Báo cáo đặt bài toán giám sát tài xế trong cabin: buồn ngủ, mất tập trung, nhìn lệch hướng, dùng điện thoại.\nCách cũ thường dựa vào một kênh đầu vào hoặc ngưỡng heuristic.",
            "Mở vấn đề: hệ thống đơn kênh dễ sai khi hành vi không rõ, bị che khuất, ánh sáng kém hoặc biểu hiện qua nhiều tín hiệu cùng lúc. Vì vậy em chọn hướng đa phương thức và suy luận theo clip.",
            None,
        ),
        (
            "Đóng góp chính: baseline đa phương thức thống nhất trên 5 lớp, đọc kết quả dựa trên artifact đã xác minh, phân tích rõ vùng lỗi, tách nghiên cứu khỏi tầng ứng dụng MDDAD.",
            "Điểm mạnh khi bảo vệ: không chỉ khoe metric, mà chứng minh mình hiểu mô hình làm tốt ở đâu, yếu ở đâu, và phần nào là nghiên cứu, phần nào là demo ứng dụng.",
            None,
        ),
        (
            "Tầng nghiên cứu: mmrt_multimodal_full_v2_t4x2.\nTầng ứng dụng: MDDAD.",
            "Nói rõ: model nghiên cứu là lõi học sâu đa phương thức; MDDAD là lớp triển khai workflow để upload video, giám sát, log sự kiện, tính rủi ro và sinh báo cáo.",
            CAUTION_FILL,
        ),
    ])

    add_heading(doc, "2. Dataset và nhãn")
    add_two_col_table(doc, [
        (
            "Dataset: DBR-DMD-v1.1 CLEAN_FULLMM_5CLASS.\nTổng: 2216 mẫu.\nTrain: 1432, Validation: 370, Test: 414.",
            "Đây là bộ dữ liệu đã được làm sạch theo 5 lớp, mỗi mẫu có đủ bốn modality đồng bộ. Em dùng validation macro-F1 để chọn checkpoint vì dữ liệu mất cân bằng.",
            None,
        ),
        (
            "5 lớp: normal_driving, eye_closure, yawning, looking_away, phone_use.",
            "Cách giải thích lớp: bình thường, nhắm mắt, ngáp, nhìn lệch hướng và sử dụng điện thoại. Đây là các trạng thái gần với mục tiêu phát hiện buồn ngủ và giảm chú ý.",
            None,
        ),
        (
            "Phân bố lớp: Train normal 720, eye 80, yawn 30, away 298, phone 304.\nVal normal 181, eye 23, yawn 8, away 83, phone 75.\nTest normal 168, eye 36, yawn 16, away 91, phone 103.",
            "Điểm cần nhấn mạnh: dữ liệu lệch lớp rất rõ, đặc biệt yawning ít. Vì vậy accuracy đơn thuần có thể đánh lừa; macro-F1 công bằng hơn vì tính đều theo từng lớp.",
            None,
        ),
    ])

    add_heading(doc, "3. Xử lý dữ liệu và input contract")
    add_two_col_table(doc, [
        (
            "Mỗi mẫu là một chuỗi đa phương thức theo thời gian, không phải một ảnh tĩnh.\nBody và face nhận clip 16 frame.",
            "Câu nói gọn: model không nhận diện hành vi từ một ảnh đơn lẻ; nó dùng ngữ cảnh clip để giảm nhầm lẫn giữa các trạng thái gần nhau.",
            None,
        ),
        (
            "Audio được chuyển thành log-mel kích thước 64 x 40.",
            "Audio đóng vai trò tín hiệu phụ trợ, ví dụ có âm thanh cabin hoặc lời nói. Tuy nhiên báo cáo chưa có ablation v2 nên không claim định lượng audio giúp tăng bao nhiêu điểm.",
            CAUTION_FILL,
        ),
        (
            "Skeleton được chuẩn hóa thành tensor 16 x 50.",
            "Skeleton cung cấp động học tư thế theo thời gian. Khi vấn đáp, chỉ nên nói skeleton là một modality đầu vào của checkpoint; không nói skeleton trên giao diện demo chính là output quyết định cuối cùng.",
            CAUTION_FILL,
        ),
        (
            "Thông tin artifact đủ để xác định đúng schema đầu vào, nhưng chưa đủ để tái lập toàn bộ pipeline trích đặc trưng từ dữ liệu gốc.",
            "Nếu bị hỏi về tái lập: em có thể tái lập input contract và logic checkpoint; còn một số chi tiết trích xuất/siêu tham số cuối chưa đóng gói đầy đủ nên báo cáo giữ claim vừa phải.",
            None,
        ),
    ])

    add_heading(doc, "4. Kiến trúc mô hình")
    add_two_col_table(doc, [
        ("Body branch và face branch: MobileNetV3-Small.", "Hai nhánh ảnh dùng backbone gọn để cân bằng biểu diễn và chi phí tính toán, phù hợp dữ liệu cabin quy mô vừa.", None),
        ("Audio branch: MLP + GRU trên log-mel và scalar features.", "Audio không xử lý như detector riêng; nó được mã hóa thành đặc trưng chuỗi rồi đưa vào hợp nhất đa phương thức.", None),
        ("Skeleton branch: MLP + BiGRU.", "Skeleton được dùng như chuỗi landmark/pose theo thời gian, giúp bổ sung chuyển động tư thế và hướng đầu.", None),
        ("Fusion: gated fusion, sau đó BiGRU + attention pooling, cuối cùng classifier 5 lớp.", "Điểm học thuật chính: mô hình học cách cân nhắc từng modality bằng gated fusion, rồi dùng BiGRU + attention để ra quyết định theo thời gian.", GREEN_FILL),
        ("Không dùng video transformer lớn như SlowFast/TimeSFormer/Video Swin làm lõi chính.", "Lý do: dữ liệu 2216 mẫu chưa lớn, chi phí pipeline nằm ở 4 modality dị chất; chọn kiến trúc gọn giúp dễ phân tích lỗi và triển khai dần.", None),
    ])

    add_heading(doc, "5. Huấn luyện và checkpoint")
    add_two_col_table(doc, [
        ("Checkpoint trung tâm: mmrt_multimodal_full_v2_t4x2.\nMôi trường huấn luyện: Kaggle GPU T4x2.", "Khi giới thiệu training: đây là checkpoint tốt nhất đã xác minh từ artifact, không phải model vừa gắn tạm vào app.", None),
        ("So với v1, v2 giữ hướng mô hình nhưng siết training: DataParallel, AMP, batch size lớn hơn, sqrt inverse class weighting, label smoothing.", "Cách nói: cải tiến chính của v2 nằm ở chiến lược huấn luyện và xử lý mất cân bằng, không phải đổi toàn bộ kiến trúc.", None),
        ("Metric checkpoint: validation 5-class macro-F1.\nV2 đạt 0.7129; v1 đạt khoảng 0.696 ở epoch tốt nhất.", "Nên nói: v2 cải thiện nhẹ nhưng ổn định hơn về cách train. Mức tăng nhỏ cho thấy hệ thống có thể đã chạm giới hạn của schema dữ liệu hiện tại.", None),
        ("Báo cáo chưa xác minh test macro-F1 cho checkpoint tốt nhất; test metric chỉ công bố khi test set và artifact đủ điều kiện.", "Không nên claim: “model đạt 0.7129 trên test”. Câu đúng là “checkpoint tốt nhất đạt validation macro-F1 = 0.7129”.", RED_FILL),
    ])

    add_heading(doc, "6. Kết quả và phân tích lỗi")
    add_two_col_table(doc, [
        ("Kết quả chính: validation macro-F1 = 0.7129, accuracy = 0.765.", "Đây là con số trung tâm khi hội đồng hỏi “model tốt đến đâu”. Nhớ luôn nói kèm “trên validation”.", GREEN_FILL),
        ("Các lớp ổn định hơn: phone_use, eye_closure, yawning.", "Giải thích: các lớp này có dấu hiệu rõ như điện thoại (F1 = 1.0), mắt nhắm (F1 = 0.958) hoặc miệng/ngáp (F1 = 0.800).", None),
        ("Nguồn lỗi chính: looking_away bị nhầm mạnh thành normal_driving.", "Đây là insight quan trọng: looking_away có F1 = 0.000 do bị nhầm 81/83 mẫu sang Normal. Lỗi do ranh giới nhãn thời gian cực kỳ hẹp.", CAUTION_FILL),
        ("Báo cáo đã có Modality-Dropout Ablation trên v2.", "Nếu bị hỏi đóng góp từng modality: trả lời có thí nghiệm modality-dropout; audio quan trọng nhất (F1 giảm 96.72%), body quan trọng nhì (F1 giảm 24.11%), face/skeleton loại bỏ giúp F1 tăng nhẹ.", GREEN_FILL),
    ])

    add_heading(doc, "7. MDDAD và tầng ứng dụng")
    add_two_col_table(doc, [
        ("MDDAD có upload video analysis, webcam monitoring, event timeline, risk score, safety score, report, history/evaluation.", "Cách nói: app chứng minh hướng chuyển giao, giúp người dùng nhìn thấy phiên giám sát, sự kiện, điểm rủi ro và báo cáo thay vì chỉ có file checkpoint.", None),
        ("App hiện hành chưa phải runtime thống nhất 100% của mmrt. Realtime vẫn có MediaPipe/OpenCV rules và detector điện thoại trong một số nhánh.", "Nói thật để không bị bắt lỗi: MDDAD là application layer/demo, còn bằng chứng hiệu năng 5 lớp phải lấy từ artifact model nghiên cứu.", RED_FILL),
        ("Điểm giao nhau giữa app và baseline: tư duy xử lý theo clip/cửa sổ thời gian, sinh event, score và report.", "Câu nối hợp lý: nghiên cứu cung cấp hướng nhận diện đa phương thức; MDDAD minh họa cách biến kết quả thành workflow giám sát tài xế.", None),
    ])

    add_heading(doc, "8. Kiểm thử hệ thống")
    add_two_col_table(doc, [
        ("Functional end-to-end: login/logout, tạo driver, upload video, MediaPipe/phone detector workflow, realtime monitoring, snapshot, dashboard, history, report, evaluation CSV.", "Đây là phần nghiệm thu sản phẩm: hệ thống chạy được trọn workflow, có dữ liệu lưu lại và xem lại được.", None),
        ("Qualitative explainability: timeline, evidence snapshot, risk explanation, report có dễ hiểu không.", "Điểm bảo vệ: với hệ thống an toàn tài xế, không chỉ cần dự đoán mà còn phải giải thích được tại sao sinh cảnh báo.", None),
        ("Simple quantitative app metrics: total sessions, safety/risk score, alert level, event count, face detection rate, FPS, phone usage count.", "Nhấn mạnh: các chỉ số này đo vận hành của app, không thay thế macro-F1 của model nghiên cứu.", CAUTION_FILL),
        ("Failure cases: camera unavailable, missing model, missing upload file, no face detected, poor lighting, invalid upload.", "Nên nói: hệ thống cần xuống cấp có kiểm soát, không chỉ chạy trong điều kiện lý tưởng.", None),
    ])

    add_heading(doc, "9. Threats to validity và giới hạn")
    add_two_col_table(doc, [
        ("Construct validity: looking_away có thể chồng lấn với normal_driving khi lệch đầu/ánh mắt nhẹ.", "Cách nói: ranh giới nhãn là một nguyên nhân hợp lý của confusion, không chỉ là lỗi thuật toán.", None),
        ("Internal validity: thiếu log training đầy đủ và ablation từng modality cho v2.", "Không suy diễn quá mức vai trò audio/skeleton nếu chưa có thí nghiệm loại bỏ từng kênh.", RED_FILL),
        ("External validity: metric chính ở validation DBR-DMD-v1.1, chưa đủ để suy rộng mạnh sang cabin/camera/ánh sáng khác.", "Câu trả lời tốt: cần test-set đầy đủ, cross-subject/cross-domain và benchmark điều kiện thực tế trước khi deploy nghiêm túc.", None),
        ("Deployment validity: MDDAD chưa trùng hoàn toàn runtime mmrt.", "Phân biệt rõ: app demo khả dụng không đồng nghĩa model nghiên cứu đã được benchmark realtime hoàn chỉnh trong sản phẩm.", CAUTION_FILL),
    ])

    add_heading(doc, "10. Bản nói 3 phút")
    add_callout(
        doc,
        "Script ngắn",
        "Đề tài của em/nhóm em giải quyết bài toán phát hiện buồn ngủ và giảm chú ý của tài xế trong cabin. Thay vì chỉ dựa vào một tín hiệu như mắt, đầu hoặc điện thoại, hướng nghiên cứu xây dựng baseline đa phương thức với bốn nguồn đầu vào gồm body, face, audio và skeleton. Dữ liệu sử dụng là DBR-DMD-v1.1 CLEAN_FULLMM_5CLASS với 2216 mẫu và 5 lớp: normal_driving, eye_closure, yawning, looking_away và phone_use. Vì dữ liệu mất cân bằng, đặc biệt lớp yawning ít, em dùng macro-F1 làm metric chính.\n\nMô hình trung tâm là mmrt_multimodal_full_v2_t4x2. Body và face dùng MobileNetV3-Small, audio dùng MLP + GRU, skeleton dùng MLP + BiGRU. Các embedding được hợp nhất bằng gated fusion, sau đó đi qua BiGRU + attention pooling để phân loại 5 lớp. Checkpoint tốt nhất đạt validation macro-F1 = 0.7129. Kết quả cho thấy các lớp phone_use, eye_closure và yawning ổn định hơn, còn lỗi chính nằm ở looking_away bị dự đoán nhầm thành normal_driving do ranh giới ngữ nghĩa gần nhau.\n\nBên cạnh phần nghiên cứu, hệ thống MDDAD được xây dựng như tầng minh họa ứng dụng, gồm phân tích video, giám sát webcam, timeline sự kiện, điểm rủi ro/an toàn và báo cáo. Em tách rõ hai tầng: hiệu năng model lấy từ artifact nghiên cứu, còn MDDAD chứng minh workflow ứng dụng. Giới hạn hiện tại là chưa có test macro-F1 cuối, nhưng nhóm đã thực hiện thành công Modality-Dropout ablation cho v2 và benchmark latency cho model. Hướng tiếp theo là bổ sung test-set summary đầy đủ và đưa runtime mmrt thống nhất vào app.",
        GREEN_FILL,
    )

    add_heading(doc, "11. Câu hỏi hội đồng dễ hỏi")
    add_two_col_table(doc, [
        ("Vì sao dùng multimodal?", "Vì hành vi tài xế không luôn biểu hiện qua một kênh. Buồn ngủ, nhìn lệch, dùng điện thoại hoặc nói chuyện có thể cần kết hợp mặt, thân người, âm thanh và skeleton theo thời gian.", None),
        ("Vì sao dùng macro-F1?", "Dataset mất cân bằng lớp; macro-F1 tính đều theo từng lớp nên công bằng hơn accuracy, nhất là với lớp ít mẫu như yawning.", None),
        ("0.7129 là test hay validation?", "Validation macro-F1. Báo cáo cố ý không claim là test vì artifact tốt nhất chưa đủ điều kiện xác minh test macro-F1.", RED_FILL),
        ("Audio và skeleton đóng góp thế nào?", "Thí nghiệm modality-dropout chỉ ra audio là quan trọng nhất: khi bỏ audio, F1 sụt giảm nghiêm trọng tới 96.72% (còn 0.023). Ngược lại, bỏ face hoặc skeleton riêng lẻ làm F1 tăng nhẹ +1.07% (đạt 0.721) do nhiễu.", GREEN_FILL),
        ("MDDAD có phải chính model mmrt chạy realtime không?", "Không hoàn toàn. MDDAD là application layer minh họa workflow. Runtime hiện có các nhánh heuristic/detector để demo ổn định; benchmark model 5 lớp dựa vào artifact mmrt.", CAUTION_FILL),
        ("Điểm yếu lớn nhất của model là gì?", "Ranh giới normal_driving và looking_away. Mô hình bỏ sót hoàn toàn lớp looking_away (recall = 0) do các mẫu này bị dự đoán nhầm thành lái xe bình thường (Normal).", None),
        ("Làm gì để nâng báo cáo lên mức mạnh hơn?", "Bổ sung test-set summary đầy đủ, kiểm thử cross-subject/cross-domain và tích hợp runtime mmrt thống nhất vào MDDAD.", None),
    ])

    add_heading(doc, "12. Checklist trước khi vấn đáp")
    for item in [
        "Nhớ nói đúng tên model: mmrt_multimodal_full_v2_t4x2.",
        "Nhớ nói đúng dataset: DBR-DMD-v1.1 CLEAN_FULLMM_5CLASS, 2216 mẫu, 5 lớp, 4 modality.",
        "Nhớ nói đúng metric: validation macro-F1 = 0.7129, không nói test macro-F1.",
        "Nhấn mạnh dữ liệu mất cân bằng nên dùng macro-F1.",
        "Giải thích rõ model là video/clip-level multimodal, không phải image detector đơn lẻ.",
        "Tách nghiên cứu mmrt khỏi app MDDAD để tránh bị hỏi lẫn bằng chứng.",
        "Thừa nhận giới hạn: chưa benchmark latency/throughput realtime trên thiết bị biên thực tế.",
    ]:
        add_bullet(doc, item)

    add_callout(
        doc,
        "Câu chốt cuối buổi",
        "Kết luận quan trọng nhất của đề tài là mô hình đa phương thức theo thời gian là hướng phù hợp cho giám sát tài xế trong cabin, nhưng cần phân biệt rõ giữa kết quả nghiên cứu đã xác minh và tầng ứng dụng demo. Nhóm đã xây được baseline có bằng chứng rõ, biết vùng lỗi chính, và có nền tảng để tiếp tục hoàn thiện bằng test-set, ablation và runtime thống nhất.",
        CALLOUT_FILL,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
