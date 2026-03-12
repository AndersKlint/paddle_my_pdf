import fitz
import time

CJK_FONT_PATH = "/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc"

# Method 1
page_pdfs = []
for i in range(10):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), f"hello world {i}", fontname="Noto", fontfile=CJK_FONT_PATH)
    path = f"/home/anders/git/paddle_my_pdf/p_{i}.pdf"
    doc.save(path)
    page_pdfs.append(path)

merged = fitz.open()
for p in page_pdfs:
    with fitz.open(p) as pg:
        merged.insert_pdf(pg)
print("Parallel generated font count sum:", sum([len(merged.get_page_fonts(i)) for i in range(10)]))
merged.save("/home/anders/git/paddle_my_pdf/merged_parallel.pdf", garbage=4)

# Method 2
single = fitz.open()
for i in range(10):
    page = single.new_page()
    page.insert_text((100, 100), f"hello world {i}", fontname="Noto", fontfile=CJK_FONT_PATH)
single.save("/home/anders/git/paddle_my_pdf/merged_single.pdf", garbage=4)
print("Single generated font count sum:", sum([len(single.get_page_fonts(i)) for i in range(10)]))
