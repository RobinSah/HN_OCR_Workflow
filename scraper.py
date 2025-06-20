import os
import time
import requests

# Base URL for the 1910 Minneapolis City Directory (ImageTileRenderer endpoint).
# The placeholder {} will be replaced with the page number.
base_url = (
    "https://box2.nmtvault.com/Hennepin2/servlet/ImageTileRenderer"
    "?doc_id=7083e412-1de2-42fe-b070-7f82e5c869a4%2Fmnmhcl00%2F20130429%2F00000018"
    "&pg_seq={}"  # page number will be formatted here
    "&query1_modifier=AND&query1_field=DATE_PUBLISHED_MILLIS"
    "&query1_min=-2208988800000&query1_max=-599702400000&search_doc="
    "&scale=0.5811577752553916&rotation=0"
)

# Output directory for images
output_dir = "minneapolis_1910_pages"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist:contentReference[oaicite:4]{index=4}

page_num = 1  # starting page number
while True:
    url = base_url.format(page_num)
    try:
        response = requests.get(url)
    except Exception as e:
        print(f"Request for page {page_num} failed: {e}")
        break

    if response.status_code != 200:
        # Stop if we hit a page that doesn't exist (non-200 status)
        print(f"No more pages (HTTP {response.status_code}). Stopping at page {page_num - 1}.")
        break

    # Save the image content to a local file
    filename = f"page_{page_num:04d}.jpg"  # zero-pad page number to 4 digits:contentReference[oaicite:5]{index=5}
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "wb") as f:  # open file in binary write mode
        f.write(response.content)     # write the image data to file:contentReference[oaicite:6]{index=6}

    print(f"Downloaded page {page_num:04d} -> {filename}")
    page_num += 1  # move to the next page
    time.sleep(1)  # polite delay of 1 second between requests:contentReference[oaicite:7]{index=7}
