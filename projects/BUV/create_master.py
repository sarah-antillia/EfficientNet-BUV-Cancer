#
# create_master.py
# 2023/04/10 Copy right (C) Antillia.com Toshiyuki Arai

import os
import glob
from   PIL import Image
import shutil
import traceback

def create_master(input_dir, category, output_dir):
  categorized_input_dir_category  = os.path.join(input_dir, category)
  sub_dirs = os.listdir(categorized_input_dir_category)

  categorized_output_dir_category = os.path.join(output_dir, category)
  if not os.path.exists(categorized_output_dir_category):
    os.makedirs(categorized_output_dir_category)

  for sub_dir in sub_dirs:
    full_sub_dir = os.path.join(categorized_input_dir_category, sub_dir)
    pattern = full_sub_dir + "/*.png"
    print("=== pattern {}".format(pattern))

    files = glob.glob(pattern)
    #input("HIT")
    for file in files:
      img      = Image.open(file)
      img      = img.resize((360, 360))
      basename = os.path.basename(file)
      output_filename = sub_dir + "_" + basename
      output_filepath = os.path.join(categorized_output_dir_category, output_filename)
      print("=== Copyed to {}".format(output_filepath))
      img.save(output_filepath)
      #shutil.copy2(file, output_filepath)
      

if __name__ == "__main__":
  input_dir = "./rawframes"
  categories = ["benign", "malignant"]
  output_dir = "./Miccai_2022_BUV_Dataset_master_360x360"
  try:
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    for category in categories:
      create_master(input_dir, category, output_dir)

  except:
    traceback.print_exc()