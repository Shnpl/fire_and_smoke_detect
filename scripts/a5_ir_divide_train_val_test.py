import os

root_dir = 'datasets/ir_dataset'
# # make dirs
# os.mkdir(os.path.join(root_dir,'train'))
# os.mkdir(os.path.join(root_dir,'val'))
# os.mkdir(os.path.join(root_dir,'test'))
# os.mkdir(os.path.join(root_dir,'train','images'))
# os.mkdir(os.path.join(root_dir,'train','labels'))
# os.mkdir(os.path.join(root_dir,'val','images'))
# os.mkdir(os.path.join(root_dir,'val','labels'))
# os.mkdir(os.path.join(root_dir,'test','images'))
# os.mkdir(os.path.join(root_dir,'test','labels'))

total_list = os.listdir(os.path.join(root_dir,'labels'))
items = []
for file in total_list:
    item = {}
    item_path = os.path.join(root_dir,'labels',file)
    with open(item_path,'r') as f:
        label = f.read()
        item['label'] = label
        item['path'] = item_path
        items.append(item)
rgb_items = [item for item in items if item['label'] == '0']
print(f"RGB num:{len(rgb_items)}")
ir_red_items = [item for item in items if item['label'] == '1']
print(f"IR red num:{len(ir_red_items)}")
ir_green_items = [item for item in items if item['label'] == '2']
print(f"IR green num:{len(ir_green_items)}")
ir_grey_items = [item for item in items if item['label'] == '3']
print(f"IR grey num:{len(ir_grey_items)}")

# divide in a ratio of 7:2:1
rgb_num = len(rgb_items)
train_rgb_items = rgb_items[:int(rgb_num*0.7)].copy()
rgb_items = rgb_items[int(rgb_num*0.7):]
val_rgb_items = rgb_items[:int(rgb_num*0.2)].copy()
rgb_items = rgb_items[int(rgb_num*0.2):]
test_rgb_items = rgb_items.copy()

ir_red_num = len(ir_red_items)
train_ir_red_items = ir_red_items[:int(ir_red_num*0.7)].copy()
ir_red_items = ir_red_items[int(ir_red_num*0.7):]
val_ir_red_items = ir_red_items[:int(ir_red_num*0.2)].copy()
ir_red_items = ir_red_items[int(ir_red_num*0.2):]
test_ir_red_items = ir_red_items.copy()

ir_green_num = len(ir_green_items)
train_ir_green_items = ir_green_items[:int(ir_green_num*0.7)].copy()
ir_green_items = ir_green_items[int(ir_green_num*0.7):]
val_ir_green_items = ir_green_items[:int(ir_green_num*0.2)].copy()
ir_green_items = ir_green_items[int(ir_green_num*0.2):]
test_ir_green_items = ir_green_items.copy()

ir_grey_num = len(ir_grey_items)
train_ir_grey_items = ir_grey_items[:int(ir_grey_num*0.7)].copy()
ir_grey_items = ir_grey_items[int(ir_grey_num*0.7):]
val_ir_grey_items = ir_grey_items[:int(ir_grey_num*0.2)].copy()
ir_grey_items = ir_grey_items[int(ir_grey_num*0.2):]
test_ir_grey_items = ir_grey_items.copy()

print(f"train_rgb_num:{len(train_rgb_items)}")
print(f"val_rgb_num:{len(val_rgb_items)}")
print(f"test_rgb_num:{len(test_rgb_items)}")

print(f"train_ir_red_num:{len(train_ir_red_items)}")
print(f"val_ir_red_num:{len(val_ir_red_items)}")
print(f"test_ir_red_num:{len(test_ir_red_items)}")

print(f"train_ir_green_num:{len(train_ir_green_items)}")
print(f"val_ir_green_num:{len(val_ir_green_items)}")
print(f"test_ir_green_num:{len(test_ir_green_items)}")

print(f"train_ir_grey_num:{len(train_ir_grey_items)}")
print(f"val_ir_grey_num:{len(val_ir_grey_items)}")
print(f"test_ir_grey_num:{len(test_ir_grey_items)}")

train_items = train_rgb_items + train_ir_red_items + train_ir_green_items + train_ir_grey_items

val_items = val_rgb_items + val_ir_red_items + val_ir_green_items + val_ir_grey_items

test_items = test_rgb_items + test_ir_red_items + test_ir_green_items + test_ir_grey_items

print(f"train_num:{len(train_items)}")
print(f"val_num:{len(val_items)}")
print(f"test_num:{len(test_items)}")
print(train_items[0])
# move images and labels
for item in train_items:
    image_path = item['path'].replace('labels','images').replace('.txt','.jpg')
    os.rename(item['path'],os.path.join(root_dir,'train','labels',item['path'].split('/')[-1]))
    os.rename(image_path,os.path.join(root_dir,'train','images',image_path.split('/')[-1]))

for item in val_items:
    image_path = item['path'].replace('labels','images').replace('.txt','.jpg')
    os.rename(item['path'],os.path.join(root_dir,'val','labels',item['path'].split('/')[-1]))
    os.rename(image_path,os.path.join(root_dir,'val','images',image_path.split('/')[-1]))

for item in test_items:
    image_path = item['path'].replace('labels','images').replace('.txt','.jpg')
    os.rename(item['path'],os.path.join(root_dir,'test','labels',item['path'].split('/')[-1]))
    os.rename(image_path,os.path.join(root_dir,'test','images',image_path.split('/')[-1]))

