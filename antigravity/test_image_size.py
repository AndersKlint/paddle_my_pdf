import fitz
import tempfile
import cv2
import os
import numpy as np

src_doc = fitz.open("test_pdf/koyu_chapter1_test_small.pdf")
page = src_doc[0]
zoom = 250 / 72.0
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
if pix.n == 4:
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
elif pix.n == 3:
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
else:
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

fd, tmp_img_path = tempfile.mkstemp(suffix=".jpg")
os.close(fd)
cv2.imwrite(tmp_img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

jpeg_size = os.path.getsize(tmp_img_path)
print(f"JPEG size on disk: {jpeg_size / 1024:.0f} KB")

# Test PDF save size
dst_doc = fitz.open()
new_page = dst_doc.new_page(width=page.rect.width, height=page.rect.height)
new_page.insert_image(new_page.rect, filename=tmp_img_path)
dst_doc.save("test_pdf/test_out.pdf")
dst_doc.close()
pdf_size = os.path.getsize("test_pdf/test_out.pdf")
print(f"PDF size: {pdf_size / 1024:.0f} KB")

# If it's still large, what if we use fitz's native convert to jpeg?
pil_jpg_size = len(pix.tobytes("jpeg"))
print(f"Fitz native JPEG size: {pil_jpg_size / 1024:.0f} KB")

os.remove(tmp_img_path)
