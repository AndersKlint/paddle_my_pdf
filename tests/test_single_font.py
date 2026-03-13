import fitz

CJK_FONT_PATH = "/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc"

# Method: create pages, then insert text in main thread
doc = fitz.open()
for i in range(10):
    page = doc.new_page()

# Insert text
for i in range(10):
    page = doc[i]
    page.insert_text((100, 100), f"hello world {i}", fontname="Noto", fontfile=CJK_FONT_PATH)

doc.save("test_single_font.pdf", garbage=4)
print("Single embedded font count sum:", sum([len(doc.get_page_fonts(i)) for i in range(10)]))
