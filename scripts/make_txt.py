import os
file_dir = 'datasets/fire_and_smoke_detect/Fog/val/images'
label_dir = file_dir.replace('images', 'labels')
file_list = os.listdir(file_dir)
file_list.sort()
filtered_list = []
max_num = 0
min_num = 512
for file in file_list:
    path = os.path.join(label_dir, file)
    label_path = path.replace('.jpg', '.txt')
    #print(label_path,path)
    if os.path.exists(label_path):
        with open(label_path, 'r') as label_file:
            labels = label_file.readlines()
            num = len(labels)
            if num > max_num:
                max_num = num
            if num < min_num:
                min_num = num
            print(labels)
            for label in labels:
                divided_label = label.split(' ')
                if len(divided_label) != 5:
                    print('corrupted')
                    break
            # if not empty file
            if len(labels) > 0:
                filtered_list.append(file)
print('no corruption')
print(max_num,min_num)
print(filtered_list[0:10])
print(len(filtered_list))
print(len(file_list))
with open('val.txt', 'w') as file:
    for file_name in filtered_list:
        path = os.path.join(file_dir, file_name)
        file.write(path + '\n')
        

