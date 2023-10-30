import os

root_path = 'datasets/fire_and_smoke_detect/val/labels'
files = os.listdir(root_path)
for file in files:
    label_path = os.path.join(root_path, file)
    with open(label_path,'r') as f:
        for line in f:
            line = line.replace('\n','').split(' ')
            if len(line) != 5:
                print(label_path)
                print(line)
                print('---------------------')
                continue
            # name= int(line[0])
            # x_center = float(line[1])
            # y_center = float(line[2])
            # width = float(line[3])
            # height = float(line[4])        

            # xmin = x_center - width / 2.0
            # ymin = y_center - height / 2.0
            # xmax = x_center + width / 2.0
            # ymax = y_center + height / 2.0

            # if xmin == xmax or ymin == ymax:
            #     print(label_path)
            #     print(line)
            #     print(xmin, ymin, xmax, ymax)
            #     print('---------------------')