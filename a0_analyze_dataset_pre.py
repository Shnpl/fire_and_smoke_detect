import os
import tqdm
  
root_path = 'datasets/fire_and_smoke_detect/'
    
files = os.listdir(os.path.join(root_path,'Fog','labels'))
    #fog_files = os.listdir(os.path.join(root_path,'Fog','labels'))
i = 3155
for file in tqdm.tqdm(files):
    file = file.split('.')[0]
    new_name = f'{i:06d}'
    i += 1
    os.rename(os.path.join(root_path,'Fog','labels',file+'.txt'), os.path.join(root_path,'Fog','labels',new_name+'.txt'))
    os.rename(os.path.join(root_path,'Fog','images',file+'.jpg'), os.path.join(root_path,'Fog','images',new_name+'.jpg'))
    # compare if they overlap
    # overlap = set(fire_files).intersection(set(fog_files))
    # print('overlap:', overlap)\n
 