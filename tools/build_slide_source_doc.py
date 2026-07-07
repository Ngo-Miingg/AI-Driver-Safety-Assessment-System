from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "slide_progress_assets"
OUT = ROOT / "docs" / "AI_Driver_Safety_Slide_Source_Brief_v2_papers.docx"


BLUE = RGBColor(31, 78, 121)
ORANGE = RGBColor(237, 125, 49)
DARK = RGBColor(32, 32, 32)
MUTED = RGBColor(96, 96, 96)
LIGHT_BLUE = "D9EAF7"
LIGHT_ORANGE = "FCE4D6"
LIGHT_GRAY = "F2F4F7"


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text: str, bold: bool = False, color: RGBColor | None = None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(9)
    run.font.name = "Calibri"
    if color:
        run.font.color.rgb = color
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_title(doc: Document, title: str, subtitle: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = BLUE
    r.font.name = "Calibri"

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(subtitle)
    r.font.size = Pt(12)
    r.font.color.rgb = MUTED
    r.font.name = "Calibri"

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Nguon noi dung cho AI tao slide tien do - cap nhat theo project hien tai")
    r.italic = True
    r.font.size = Pt(10)
    r.font.color.rgb = ORANGE


def add_h1(doc: Document, text: str) -> None:
    p = doc.add_heading(level=1)
    r = p.add_run(text)
    r.font.color.rgb = BLUE
    r.font.size = Pt(16)
    r.bold = True


def add_h2(doc: Document, text: str) -> None:
    p = doc.add_heading(level=2)
    r = p.add_run(text)
    r.font.color.rgb = BLUE
    r.font.size = Pt(13)
    r.bold = True


def add_para(doc: Document, text: str, bold_prefix: str | None = None) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.color.rgb = BLUE
        p.add_run(text[len(bold_prefix) :])
    else:
        p.add_run(text)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.add_run(item)


def add_callout(doc: Document, title: str, body: str, fill: str = LIGHT_BLUE) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    shade_cell(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    r.bold = True
    r.font.color.rgb = BLUE
    r.font.size = Pt(10)
    p = cell.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(body)
    r.font.size = Pt(9)
    doc.add_paragraph()


def add_metrics_table(doc: Document) -> None:
    rows = [
        ["Model", "Vai tro", "Train images", "Best epoch", "Precision", "Recall", "mAP50", "mAP50-95", "Size"],
        ["YOLO26n baseline", "Baseline/fallback toc do cao", "3000", "21", "0.93237", "0.86031", "0.92854", "0.76828", "5.1 MB"],
        ["RT-DETR-L compare", "Checkpoint so sanh", "3000", "14", "0.93969", "0.93286", "0.95693", "0.77543", "63.1 MB"],
        ["RT-DETR-L serious", "Model chinh duoc chon", "4500", "8", "0.93331", "0.92239", "0.95497", "0.80077", "63.1 MB"],
    ]
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            set_cell_text(cell, text, bold=(i == 0 or (i == 3 and j in {0, 7})), color=BLUE if i == 0 else DARK)
            if i == 0:
                shade_cell(cell, LIGHT_BLUE)
            elif i == 3:
                shade_cell(cell, LIGHT_ORANGE)
    doc.add_paragraph()


def add_dataset_table(doc: Document) -> None:
    rows = [
        ["Split", "Images", "Labels", "Ghi chu"],
        ["Train serious RT-DETR", "4500", "4500", "Tap train chinh cho model RT-DETR-L serious"],
        ["Validation", "326", "326", "Dung chon best epoch va theo doi overfit"],
        ["Test", "185", "185", "Dung kiem tra sau train"],
    ]
    table = doc.add_table(rows=len(rows), cols=4)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            set_cell_text(cell, text, bold=(i == 0), color=BLUE if i == 0 else DARK)
            if i == 0:
                shade_cell(cell, LIGHT_BLUE)
    doc.add_paragraph()


def add_image(doc: Document, path: Path, caption: str, width: float = 6.2) -> None:
    if not path.exists():
        add_callout(doc, "Thieu asset", f"Khong tim thay: {path}", LIGHT_ORANGE)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = MUTED


def add_file_index(doc: Document) -> None:
    files = sorted([p for p in ASSETS.rglob("*") if p.is_file()])
    rows = [["Nhom", "File", "Muc dich"]]
    for p in files:
        rel = p.relative_to(ASSETS)
        group = rel.parts[0] if rel.parts else ""
        if group == "00_contact_sheets":
            purpose = "Anh tong hop de upload tung phien cho AI tao slide"
        elif group == "01_model_comparison":
            purpose = "Bieu do so sanh metrics, thoi gian train, kich thuoc model"
        elif group == "02_training_curves":
            purpose = "Training curve theo epoch"
        elif group == "03_confusion_matrices":
            purpose = "Phan tich loi/confusion matrix"
        elif group == "04_prediction_samples":
            purpose = "Anh du doan mau co bounding box"
        elif group == "05_tables":
            purpose = "Bang metrics dang csv/json/md"
        elif group == "06_prompt":
            purpose = "Prompt tao slide tien do"
        else:
            purpose = "Tai lieu/asset bo sung"
        rows.append([group, str(rel), purpose])

    table = doc.add_table(rows=len(rows), cols=3)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            set_cell_text(cell, text, bold=(i == 0), color=BLUE if i == 0 else DARK)
            if i == 0:
                shade_cell(cell, LIGHT_GRAY)


def add_prompt_text(doc: Document) -> None:
    prompt_path = ASSETS / "06_prompt" / "PROMPT_TAO_SLIDE_TIEN_DO.md"
    if not prompt_path.exists():
        return
    add_h1(doc, "12. Noi Dung Prompt Day Du Trong Thu Muc Assets")
    add_para(
        doc,
        "Phan nay chep lai prompt hien co de AI tao slide co the doc truc tiep trong cung mot file Word. "
        "Neu prompt qua dai khi dua vao cong cu AI, hay dung phan rut gon o muc cuoi tai lieu.",
    )
    text = prompt_path.read_text(encoding="utf-8", errors="replace")
    # Keep the prompt readable in Word without creating a single huge paragraph.
    for block in text.splitlines():
        if not block.strip():
            continue
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(block[:900])
        r.font.name = "Consolas"
        r.font.size = Pt(7.5)
        if block.startswith("#"):
            r.bold = True
            r.font.color.rgb = BLUE


def add_individual_asset_images(doc: Document) -> None:
    add_h1(doc, "13. Phu Luc Hinh Anh Rieng Le")
    add_para(
        doc,
        "Phan nay chen cac hinh rieng le trong thu muc `slide_progress_assets` de AI co the dung lai khi can tach rieng bieu do "
        "hoac anh minh hoa thay vi dung contact sheet.",
    )
    groups = [
        ("01_model_comparison", "Bieu do so sanh model"),
        ("02_training_curves", "Training curves"),
        ("03_confusion_matrices", "Confusion matrices"),
        ("04_prediction_samples/yolo26n_baseline", "Prediction samples - YOLO26n baseline"),
        ("04_prediction_samples/rtdetrl_compare", "Prediction samples - RT-DETR-L"),
    ]
    for group, title in groups:
        folder = ASSETS / group
        images = sorted([p for p in folder.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not images:
            continue
        add_h2(doc, title)
        for p in images:
            rel = p.relative_to(ASSETS)
            width = 5.8 if "prediction_samples" not in group else 4.9
            add_image(doc, p, f"Asset rieng le: {rel}", width)


def add_related_work_sections(doc: Document) -> None:
    add_h1(doc, "10. Nghien Cuu Lien Quan Va Vi Tri Cua Du An")
    add_para(
        doc,
        "Phan nay dung de tao them cac slide hoc thuat ve paper lien quan. Muc tieu khong phai ke qua nhieu paper, "
        "ma la chi ra: cac huong nghien cuu da co, khoang trong khi dua vao he thong demo, va diem sang tao cua project hien tai.",
    )

    add_h2(doc, "10.1 Nhom paper dung landmark/MediaPipe cho buon ngu")
    add_bullets(
        doc,
        [
            "Nhieu nghien cuu drowsiness detection dung OpenCV/MediaPipe Face Mesh de lay moc mat, mieng va dau.",
            "Cac chi so thuong dung: EAR cho mat nham, MAR cho ngap, head pose/head tilt cho cui dau hoac mat tap trung.",
            "Uu diem: nhanh, giai thich duoc, phu hop realtime tren may ca nhan.",
            "Han che: phu thuoc anh sang, goc camera, chat luong landmark va nguong cau hinh.",
            "Du an hien tai ke thua huong nay cho realtime demo, nhung bo sung session/report/evidence/evaluation thay vi chi canh bao don le.",
        ],
    )

    add_h2(doc, "10.2 Nhom paper ve distracted driver action recognition")
    add_bullets(
        doc,
        [
            "AI City Challenge Track 3 tap trung nhan dien hanh vi lai xe mat tap trung trong video tu nhien, co start/end time va loai hanh vi.",
            "Cac repo/paper nhu M2DAR hoac multi-view action recognition thuong dung video clip, temporal model, SlowFast/Transformer/ensemble.",
            "Uu diem: gan voi bai toan action recognition that su, co tinh thoi gian va ngu canh hanh vi.",
            "Han che voi do an hien tai: pipeline train phuc tap, can dataset video da cat/ngan va gan nhan hanh vi, yeu cau GPU va thoi gian lon hon.",
            "Huong tiep theo cua du an la tien tu rule-based events sang train action model cho cac hanh vi bat thuong.",
        ],
    )

    add_h2(doc, "10.3 Nhom paper ve object detection va phone detection")
    add_bullets(
        doc,
        [
            "Phone usage co the duoc xu ly nhu object detection: phat hien dien thoai trong anh/frame, dua ra bbox va confidence.",
            "RT-DETR la huong detector transformer thoi gian thuc, giam nhu cau NMS va can bang giua toc do - do chinh xac.",
            "YOLO nho phu hop fallback/realtime, con RT-DETR-L cho ket qua mAP50-95 tot hon trong thuc nghiem hien tai.",
            "Vi phone usage la nguy co can canh bao nhanh, detector theo frame co ly do thuc te hon viec doi action event 2-3 giay moi canh bao.",
        ],
    )

    add_h2(doc, "10.4 Bang tom tat paper/huong nghien cuu lien quan")
    rows = [
        ["Nhom nghien cuu", "Cach lam chinh", "Diem manh", "Khoang trong/han che", "Lien he voi du an"],
        [
            "Drowsiness bang MediaPipe/OpenCV",
            "Face Mesh, EAR, MAR, head pose",
            "Nhanh, de giai thich, realtime",
            "Phu thuoc nguong va goc camera",
            "Dang duoc dung cho mat nham, ngap, cui dau, nhin lech",
        ],
        [
            "AI City distracted action recognition",
            "Video clip, temporal model, action localization",
            "Dung ban chat hanh vi theo thoi gian",
            "Can dataset video gan nhan va train nang",
            "La huong train tiep theo cho hanh vi bat thuong",
        ],
        [
            "RT-DETR real-time detection",
            "Transformer object detector thoi gian thuc",
            "mAP tot, end-to-end detector",
            "Model nang hon YOLO nho",
            "Duoc fine-tune lam model phone chinh",
        ],
        [
            "YOLO lightweight detection",
            "One-stage detector nhe",
            "Nhanh, de trien khai",
            "Do chinh xac thap hon RT-DETR-L serious",
            "Giu lam fallback/realtime baseline",
        ],
    ]
    table = doc.add_table(rows=len(rows), cols=5)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            set_cell_text(cell, text, bold=i == 0, color=BLUE if i == 0 else DARK)
            if i == 0:
                shade_cell(cell, LIGHT_BLUE)

    add_h1(doc, "11. Diem Noi Bat Va Sang Tao Cua Du An")
    add_callout(
        doc,
        "Diem khac biet can dua vao slide",
        "Du an khong chi trinh bay mot model detect phone. Diem manh la ghep model da train vao mot he thong DMS co realtime, upload, "
        "temporal event tracking, risk engine, evidence snapshot, bao cao va evaluation.",
        LIGHT_ORANGE,
    )
    add_bullets(
        doc,
        [
            "Huong lai ghep: MediaPipe/rule-based cho fatigue-distraction + model train/fine-tune cho phone detection.",
            "Co temporal logic: khong chi detect tung frame, ma chuyen thanh su kien co thoi gian bat dau/ket thuc va muc do.",
            "Co risk engine rieng: bien cac su kien thanh Risk Score, Safety Score va Alert Level de phu hop demo DMS.",
            "Co evidence snapshot: moi su kien co the co anh minh chung nho, giup bao cao de giai thich.",
            "Co evaluation toolkit: luu processed frames, face detection rate, event counts, export CSV.",
            "Co model comparison that: YOLO26n vs RT-DETR-L, chon model dua tren mAP50-95, toc do train va kich thuoc model.",
            "Kien truc co the mo rong: thay phone detector hoac them action model ma khong can viet lai workflow web.",
        ],
    )

    add_h1(doc, "12. Cach Noi Ve Huong Train Hanh Vi Bat Thuong Tiep Theo")
    add_para(
        doc,
        "Can noi ro voi hoi dong: phan hanh vi bat thuong hien tai dang dung OpenCV/MediaPipe va nguong cau hinh de dam bao demo on dinh. "
        "Day khong phai diem yeu neu trinh bay dung: no la baseline giai thich duoc. Huong nghien cuu tiep theo la train model hanh vi de thay the/tang cuong baseline nay.",
    )
    add_bullets(
        doc,
        [
            "Buoc 1: thu thap video cabin theo tung hanh vi: normal, eye closure, yawning, looking away, head down, phone usage, mixed-risk.",
            "Buoc 2: cat clip ngan 2-5 giay hoac gan nhan theo timestamp start/end neu lam action localization.",
            "Buoc 3: train model action recognition/video model: CNN-LSTM, SlowFast, Video Transformer hoac multimodal fusion neu co nhieu dau vao.",
            "Buoc 4: so sanh voi baseline rule-based bang Accuracy, Precision, Recall, F1, confusion matrix, inference time.",
            "Buoc 5: tich hop output action model vao Temporal Logic va Risk Engine hien co.",
        ],
    )
    add_callout(
        doc,
        "Thong diep nen noi",
        "Hien tai du an da train thanh cong phone detector va tich hop vao app. Giai doan tiep theo se mo rong train AI cho cac hanh vi bat thuong, "
        "dua tren cung kien truc da co san: input frame/video -> model -> temporal event -> risk score -> report.",
        LIGHT_BLUE,
    )


def add_sources(doc: Document) -> None:
    add_h1(doc, "15. Nguon Paper Va Tai Lieu Tham Khao De Tao Slide")
    rows = [
        ["Chu de", "Nguon", "Lien ket"],
        [
            "RT-DETR",
            "Zhao et al., DETRs Beat YOLOs on Real-time Object Detection, CVPR 2024",
            "https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf",
        ],
        [
            "RT-DETR official/project",
            "RT-DETR project page / code",
            "https://zhao-yian.github.io/RTDETR/",
        ],
        [
            "AI City Track 3",
            "2023 AI City Challenge Data and Evaluation - Naturalistic Driving Action Recognition",
            "https://www.aicitychallenge.org/2023-data-and-evaluation/",
        ],
        [
            "Distracted driver action recognition",
            "Multi View Action Recognition for Distracted Driver Behavior Localization, CVPRW 2023",
            "https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Zhou_Multi_View_Action_Recognition_for_Distracted_Driver_Behavior_Localization_CVPRW_2023_paper.pdf",
        ],
        [
            "M2DAR source",
            "PurdueDigitalTwin/M2DAR - driver action recognition in untrimmed videos",
            "https://github.com/PurdueDigitalTwin/M2DAR",
        ],
        [
            "MediaPipe drowsiness",
            "Development of a Real-time Driver's Drowsiness Detection System Using MediaPipe Face Mesh",
            "https://www.mecs-press.org/ijem/ijem-v15-n5/v15n5-4.html",
        ],
        [
            "Integrated DMS",
            "Integrated deep learning framework for driver distraction detection and driver safety assessment",
            "https://www.nature.com/articles/s41598-025-08475-4",
        ],
    ]
    table = doc.add_table(rows=len(rows), cols=3)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            set_cell_text(cell, text, bold=i == 0, color=BLUE if i == 0 else DARK)
            if i == 0:
                shade_cell(cell, LIGHT_GRAY)


def build() -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"].font.size = Pt(10.5)
    styles["Normal"].paragraph_format.space_after = Pt(6)
    styles["List Bullet"].font.name = "Calibri"
    styles["List Bullet"].font.size = Pt(10)

    add_title(
        doc,
        "AI Driver Safety Assessment System",
        "Tai lieu nguon cho AI tao slide tien do va bao ve huong train AI",
    )

    add_h1(doc, "1. Tom Tat Dung Ban Chat Du An")
    add_para(
        doc,
        "Du an xay dung mot he thong web danh gia an toan tai xe tu webcam realtime va video upload. "
        "He thong khong chi dung o muc demo model, ma da co workflow dau-cuoi: dang nhap, quan ly tai xe, "
        "phan tich video/realtime, luu session, timeline su kien, dashboard, report, settings, evaluation va export CSV."
    )
    add_callout(
        doc,
        "Thong diep chinh cho slide",
        "Tien do hien tai da co he thong hoan chinh de demo. Trong do, phan phone detection da duoc train/fine-tune va tich hop; "
        "phan hanh vi bat thuong nhu mat nham, ngap, nhin lech, cui dau hien dang dung MediaPipe/OpenCV + nguong cau hinh, "
        "va la huong train AI tiep theo.",
        LIGHT_ORANGE,
    )

    add_h1(doc, "2. Tinh Nang Da Hoan Thanh")
    add_bullets(
        doc,
        [
            "Flask web app: login/logout, demo account admin/admin123, quan ly nhieu tai xe.",
            "Dashboard voi du lieu that tu SQLite: tong session, diem an toan, su kien va danh sach session gan day.",
            "Upload video analysis: doc metadata bang OpenCV, sample frame, phan tich bang MediaPipe va phone detector.",
            "Realtime webcam monitoring: Flask-SocketIO, OpenCV webcam, MediaPipe, metrics live va low-rate JPEG snapshot.",
            "Event evidence snapshots: luu anh minh chung nho khi xac nhan su kien.",
            "Risk engine: Risk Score, Safety Score, Alert Level, fatigue trend va rule-based report.",
            "History, Session Detail, Reports, Settings/Calibration, Evaluation page va CSV export.",
            "Phone-test UI rieng: upload anh/video, chon model, output co bounding box.",
        ],
    )

    add_h1(doc, "3. Kien Truc Va Luong Xu Ly")
    add_para(
        doc,
        "Pipeline tong quat: User -> Login/Driver -> Webcam realtime hoac Video upload -> OpenCV preprocessing -> "
        "MediaPipe Face Mesh + Phone Detector -> Temporal Logic -> Risk Engine -> SQLite -> Dashboard/History/Reports/Evaluation."
    )
    add_bullets(
        doc,
        [
            "OpenCV: doc camera/video, tach frame, trich metadata, resize/snapshot.",
            "MediaPipe Face Mesh: lay landmark khuon mat de tinh EAR, MAR, huong dau, trang thai mat khuon mat.",
            "Phone Detector: so sanh YOLO26n baseline va RT-DETR-L fine-tuned cho bai toan phat hien dien thoai.",
            "Temporal Logic: bien tin hieu tung frame thanh su kien theo thoi gian, giam nhieu va giam canh bao gia.",
            "Risk Engine: tong hop su kien de tinh diem rui ro, diem an toan va cap do canh bao.",
        ],
    )

    add_h1(doc, "4. Dataset Phone Detection")
    add_para(
        doc,
        "Dataset hien tai duoc dung de train/fine-tune phan phat hien dien thoai. Day la bai toan object detection 1 class "
        "`Mobile-phone`, khong phai dataset day du cho toan bo hanh vi lai xe."
    )
    add_dataset_table(doc)
    add_bullets(
        doc,
        [
            "Nguon dataset: Roboflow Mobile-phone v7.",
            "Dinh dang nhan: YOLO detection format.",
            "Bad labels = 0 sau khi kiem tra.",
            "Tap serious RT-DETR dung 4500 anh train de can bang giua chat luong va thoi gian tren Kaggle GPU.",
        ],
    )

    add_h1(doc, "5. Huong Train AI Da Thuc Hien")
    add_para(
        doc,
        "Phan train AI tap trung vao phone detection vi day la rui ro can canh bao nhanh va co nhan bounding box ro rang. "
        "Cach lam nay tao duoc ket qua dinh luong bang Precision, Recall, mAP50, mAP50-95 va co the tich hop truc tiep vao ung dung."
    )
    add_metrics_table(doc)
    add_callout(
        doc,
        "Vi sao chon RT-DETR-L serious?",
        "RT-DETR-L serious co mAP50-95 cao nhat: 0.80077. Chi so nay khat khe hon mAP50 vi danh gia bbox o nhieu nguong IoU, "
        "nen phu hop de chon model chinh. YOLO26n co kich thuoc chi 5.1 MB, vi vay duoc giu lam baseline va fallback toc do cao.",
        LIGHT_BLUE,
    )

    add_h1(doc, "6. Giai Thich Lua Chon Model")
    add_h2(doc, "YOLO26n baseline")
    add_bullets(
        doc,
        [
            "Uu diem: nhe, nhanh, de chay realtime/fallback, kich thuoc model nho.",
            "Vai tro trong project: baseline de so sanh va phuong an fallback khi can toc do cao.",
            "Han che: mAP50-95 thap hon RT-DETR-L serious trong ket qua hien tai.",
        ],
    )
    add_h2(doc, "RT-DETR-L serious")
    add_bullets(
        doc,
        [
            "Uu diem: detector transformer-based, ket qua mAP50-95 cao nhat tren thuc nghiem hien tai.",
            "Vai tro trong project: model chinh cho phone detection tren `/phone-test` va upload analysis.",
            "Han che: model nang hon, can can nhac FPS neu dua vao realtime.",
        ],
    )

    add_h1(doc, "7. Asset Tong Hop De AI Tao Slide")
    add_para(
        doc,
        "Cac anh duoi day la contact sheet da gom nhieu bieu do/anh mau theo tung nhom. Khi AI chi cho upload mot anh moi phien, "
        "hay upload tung contact sheet nay kem prompt trong file `PROMPT_TAO_SLIDE_TIEN_DO.md`."
    )
    contact = ASSETS / "00_contact_sheets"
    add_image(doc, contact / "01_model_comparison_sheet.png", "Danh muc: Model comparison + training efficiency + deployment cost.", 6.3)
    add_image(doc, contact / "02_training_curves_sheet.png", "Danh muc: Training curves cua cac model da train/fine-tune.", 6.3)
    add_image(doc, contact / "03_confusion_matrices_sheet.png", "Danh muc: Error analysis/confusion matrix.", 6.3)
    add_image(doc, contact / "06_model_metrics_table_sheet.png", "Danh muc: Bang tong hop metrics model.", 6.3)
    add_image(doc, contact / "04_yolo26n_prediction_samples_sheet.png", "Danh muc: Prediction samples - YOLO26n baseline.", 6.3)
    add_image(doc, contact / "05_rtdetrl_prediction_samples_sheet.png", "Danh muc: Prediction samples - RT-DETR-L.", 6.3)

    add_h1(doc, "8. Tich Hop Model Vao Project")
    add_bullets(
        doc,
        [
            "`models/phone_rtdetr_l.pt`: model chinh RT-DETR-L serious fine-tuned.",
            "`models/phone_yolo26n.pt`: YOLO26n baseline/fallback.",
            "`models/phone_yolo11n.pt`: compatibility copy cho workflow cu.",
            "`models/phone_rtdetr_l_light_compare.pt`: checkpoint so sanh.",
            "`RTDETR_MODEL_PATH = models/phone_rtdetr_l.pt`.",
            "`YOLO_FINETUNED_MODEL_PATH = models/phone_yolo26n.pt`.",
            "`/phone-test`: giao dien test detect phone tren anh/video, output co bounding box.",
        ],
    )

    add_h1(doc, "9. Gioi Han Hien Tai Va Huong Train Tiep Theo")
    add_para(
        doc,
        "Hien tai cac hanh vi bat thuong nhu mat nham, ngap, nhin lech, cui dau va mat khuon mat chua duoc train thanh model rieng. "
        "He thong dang dung MediaPipe Face Mesh, OpenCV va cac nguong cau hinh nhu EAR, MAR, head pose, duration threshold."
    )
    add_bullets(
        doc,
        [
            "Diem manh hien tai: nhanh, de demo, co the giai thich, khong can dataset lon cho moi hanh vi.",
            "Gioi han: phu thuoc anh sang, goc camera, nguong cau hinh va chat luong landmark.",
            "Huong tiep theo: xay dung dataset hanh vi bat thuong co nhan clip/frame cho eye closure, yawning, looking away, head down, phone usage.",
            "Co the train action recognition model hoac multimodal/action model khi co dataset video da cat ngan va gan nhan.",
            "Tiep tuc giu risk engine va temporal logic de tich hop output model moi ma khong phai viet lai workflow.",
        ],
    )
    add_callout(
        doc,
        "Cach noi truoc hoi dong",
        "Phan phone detection da duoc train va co ket qua dinh luong. Phan drowsiness/distraction hien tai la rule-based dua tren landmark, "
        "phu hop de demo va giai thich. Buoc nghien cuu tiep theo la thay tung rule bang model hanh vi duoc train tren dataset chuyen biet.",
        LIGHT_ORANGE,
    )

    add_related_work_sections(doc)

    add_h1(doc, "13. De Xuat Cau Truc Slide Tu Tai Lieu Nay")
    rows = [
        ["Slide", "Noi dung", "Asset nen dung"],
        ["1", "Tieu de va tien do hien tai", "Khong can asset hoac dung UI screenshot"],
        ["2", "Kien truc tong quan", "So do pipeline trong prompt"],
        ["3", "Hai luong realtime va upload", "So do workflow"],
        ["4", "Dataset phone detection", "06_model_metrics_table_sheet.png"],
        ["5", "Model da train/fine-tune", "01_model_comparison_sheet.png"],
        ["6", "Ket qua so sanh model", "01_model_comparison_sheet.png"],
        ["7", "Training curves", "02_training_curves_sheet.png"],
        ["8", "Error analysis va prediction samples", "03 + 05 contact sheets"],
        ["9", "Tich hop vao project", "UI /phone-test neu co screenshot"],
        ["10", "Nghien cuu lien quan", "Bang paper/huong nghien cuu"],
        ["11", "Diem noi bat va sang tao", "Bullet summary + callout"],
        ["12", "Viec da lam va huong train tiep theo", "Bullet summary"],
    ]
    table = doc.add_table(rows=len(rows), cols=3)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            set_cell_text(cell, text, bold=i == 0, color=BLUE if i == 0 else DARK)
            if i == 0:
                shade_cell(cell, LIGHT_BLUE)

    add_h1(doc, "14. Danh Muc Day Du Cac File Asset")
    add_file_index(doc)
    add_sources(doc)

    add_prompt_text(doc)
    add_individual_asset_images(doc)

    add_h1(doc, "16. Prompt Rut Gon Cho AI Tao Slide")
    add_para(
        doc,
        "Hay tao bo slide tien do 8-10 trang cho du an AI Driver Safety Assessment System bang tieng Viet, nen trang, "
        "mau xanh duong/cam, phong cach hoc thuat. Noi dung phai dung voi tien do hien tai: he thong Flask/SQLite da co realtime webcam, "
        "upload analysis, dashboard/history/report/settings/evaluation; phone detector da duoc train/fine-tune va tich hop; "
        "RT-DETR-L serious la model chinh vi mAP50-95 cao nhat 0.80077; YOLO26n la baseline/fallback nhe 5.1 MB. "
        "Khong noi da train model buon ngu/ngap/head pose; cac hanh vi nay hien dung MediaPipe/OpenCV + rule/threshold va la huong train tiep theo. "
        "Khong noi realtime stream full video; realtime chi gui metrics va low-rate JPEG snapshot. Hay chen cac bieu do/anh trong tai lieu nay vao slide phu hop.",
    )

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
