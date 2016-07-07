from PIL import Image
import sys, os

if sys.argv[1] == '--help':
    print("Average Images")
    print("--------------------------------------------------------------")
    print("`python average_image.py [source directory] [output filename]`")
    print("source directory: Directory of images to be averaged")
    print("output filename: Path for averaged image")
    print("--------------------------------------------------------------")

source_dir = sys.argv[1]
output_file = sys.argv[2]
out = None
files = os.listdir(source_dir)
total = len(files)

for i, img_name in enumerate(files):
    img_path = os.path.join(source_dir, img_name)
    img = Image.open(img_path)
    if out == None:
        out = img
        continue

    out = Image.blend(out, img, 1.0/i)

out.save(output_file)
    
