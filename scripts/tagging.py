# a program to tag the data
import tkinter as tk
from PIL import Image

import os
current_image_path = ""
def save_tag_to_file(tag):
    global label_path
    with open(label_path, 'w') as txt_file:
        txt_file.write(tag)
    return f"标签已保存到 {label_path} 文件中"

def refresh():
    global current_image_path
    image = Image.open(current_image_path)
    return image




if __name__ == '__main__':
    current_image_path = ""
    src_dir = 'datasets/ir_dataset/untagged_images'
    dest_dir = 'datasets/ir_dataset/images'
    label_dir = 'datasets/ir_dataset/labels'
    image_list = os.listdir(src_dir)
    root = tk.Tk()
    image_tk_label = tk.Label(root)
    for image_basename in image_list:
        current_image_path = os.path.join(src_dir, image_basename)
        image = Image.open(current_image_path)
        image_tk_label.configure(image=image)
        
        

#     with gr.Blocks() as iface:
#         image_box = gr.Image()
#         image_box.
#         gr.Number(value=10, label="Food Count")
#         status_box = gr.Textbox()
#         def eat(food):
#             if food > 0:
#                 return food - 1, "full"
#             else:
#                 return 0, "hungry"
#         gr.Button("EAT").click(
#             fn=eat,
#             inputs=food_box,
#             #根据返回值改变输入组件和输出组件
#             outputs=[food_box, status_box]
#         )
#     iface.launch()

#     
#         with open(image_path,'r') as f:
#             image = Image.open(image_path)
#             # disp it on webui
#             iface.show(image)
#             label_path = os.path.join(label_dir, image_basename.replace('.jpg','.txt'))
#             label = iface.get_text()
#             # save label
            




