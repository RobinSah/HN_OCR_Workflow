# pdf_batcher.py
# Script to batch-convert images into PDFs (approximately 300 images per PDF).

import os
from PIL import Image

# 1. Define input and output directories
images_dir = ".../minneapolis_1910_pages/"
output_pdf_dir = ".../minneapolis_1910_pdfs/"

# 2. Create the output directory if it doesn't exist
os.makedirs(output_pdf_dir, exist_ok=True)

# 3. Gather all JPEG image filenames from the images directory, sorted in order
image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")])
total_images = len(image_files)
chunk_size = 200  # number of images per PDF file

# 4. Loop through the images in chunks of 300
for start_idx in range(0, total_images, chunk_size):
    batch_files = image_files[start_idx : start_idx + chunk_size]
    if not batch_files:
        continue  # skip if batch is empty (should not happen unless no images)

    # 5. Open the first image in the batch and convert it to RGB mode
    first_image_path = os.path.join(images_dir, batch_files[0])
    first_image = Image.open(first_image_path).convert("RGB")

    # Open and convert the rest of the images in the batch
    additional_images = []
    for img_file in batch_files[1:]:
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        additional_images.append(img)

    # 6. Determine the output PDF file name for this batch (e.g., batch_1.pdf, batch_2.pdf, ...)
    batch_num = start_idx // chunk_size + 1  # batch numbering starts at 1
    pdf_filename = f"minneapolis_1910_batch_{batch_num}.pdf"
    pdf_path = os.path.join(output_pdf_dir, pdf_filename)

    # 7. Save the images as a single PDF file 
    # (Pillow uses the first image and appends the rest when save_all=True and append_images are provided)
    first_image.save(pdf_path, format="PDF", save_all=True, append_images=additional_images)

    # 8. Close image files to free memory
    first_image.close()
    for img in additional_images:
        img.close()

    # 9. Log the result for this batch
    print(f"Created PDF: {pdf_filename} with {len(batch_files)} pages.")

# End of pdf_batcher.py
