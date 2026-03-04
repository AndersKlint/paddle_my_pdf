import fitz
import numpy as np
import cv2
import os

# Create a dummy image
img = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
cv2.imwrite("dummy.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

doc = fitz.open()
page = doc.new_page(width=1000, height=1000)
page.insert_image(page.rect, filename="dummy.jpg")

print("Saved with deflate_images=False")
doc.save("test1.pdf", deflate_images=False)
print("File size:", os.path.getsize("test1.pdf"))

print("Saved with deflate_images=True")
doc.save("test2.pdf", deflate_images=True)
print("File size:", os.path.getsize("test2.pdf"))

os.remove("dummy.jpg")
os.remove("test1.pdf")
os.remove("test2.pdf")

