import fitz

doc = fitz.open()
page = doc.new_page()
# Try using "cjk" built-in language font
try:
    page.insert_text((100, 100), "你好，世界", fontname="cjk")
    print("Success with built-in cjk!")
except Exception as e:
    print("Failed with built-in:", e)
doc.save("test_cjk.pdf")
