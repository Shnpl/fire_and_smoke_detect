import os
root_path = 'datasets/fire_and_smoke_detect/Fog'
print(f'---root path:{root_path}---')
files = os.listdir(os.path.join(root_path,'labels'))


train_num = int(len(files)*0.7)
val_num = int(len(files)*0.2)
test_num = len(files)-train_num-val_num
train_list = []
val_list = []
test_list = []

for file in files[:train_num]:
    train_list.append(file)
files = files[train_num:]
for file in files[:val_num]:
    val_list.append(file)
files = files[val_num:]
test_list = files

# make dirs
for subdir in ['train','val','test']:
    os.mkdir(os.path.join(root_path,subdir))
    os.mkdir(os.path.join(root_path,subdir,'images'))
    os.mkdir(os.path.join(root_path,subdir,'labels'))
for file in train_list: 
    os.rename(os.path.join(root_path,'labels',file),os.path.join(root_path,'train','labels',file))
    file = file.replace('.txt','.jpg')
    os.rename(os.path.join(root_path,'images',file),os.path.join(root_path,'train','images',file))
for file in val_list:
    os.rename(os.path.join(root_path,'labels',file),os.path.join(root_path,'val','labels',file))
    file = file.replace('.txt','.jpg')
    os.rename(os.path.join(root_path,'images',file),os.path.join(root_path,'val','images',file))
for file in test_list:
    os.rename(os.path.join(root_path,'labels',file),os.path.join(root_path,'test','labels',file))
    file = file.replace('.txt','.jpg')
    os.rename(os.path.join(root_path,'images',file),os.path.join(root_path,'test','images',file))