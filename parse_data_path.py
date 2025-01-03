
import os
import os.path as osp
from sklearn.model_selection import train_test_split
# data_dir ='/scratch/zhiqiu/yfcc_dynamic_10/dynamic_300/images'


# def list_all_files(rootdir):
#     _files = []
#     list_file = os.listdir(rootdir)
#     for i in range(0,len(list_file)):
#         path = os.path.join(rootdir,list_file[i])

#         if os.path.isdir(path):
#             _files.extend(list_all_files(path))
#         if os.path.isfile(path):
#              _files.append(path)
#     return _files

def list_all_files(args,rootdir):

    train_list,test_list,all_list = [],[],[]
    bucket_list = os.listdir(rootdir)
    # bucket_list=list(filter(lambda a: 'bucket_' in a,bucket_list))
    if('0' in bucket_list):
        bucket_list.remove('0') # skip bucket 0, since it's for pretrain feature
    classes_list=  os.listdir(osp.join(rootdir,bucket_list[0]))
    if('clear25d' in args.split and 'BACKGROUND' in classes_list):
        classes_list.remove('BACKGROUND') # skip bucket 0, since it's for pretrain feature
    for bucket in bucket_list:
        for classes in classes_list:
      
            image_list=os.listdir(osp.join(rootdir,bucket,classes))
            
            image_list=list(map(lambda a: osp.join(osp.join(rootdir,bucket,classes,a)), image_list))
            image_list=image_list[:args.num_instance_each_class] # if background class have more image, we use only part of it
 
            try:
                assert len(image_list)==args.num_instance_each_class
            except:
                import pdb;pdb.set_trace()
                print('a')
            train_subset,test_subset=train_test_split(image_list,test_size=args.test_split, random_state=args.random_seed)
            train_list.extend(train_subset)
            test_list.extend(test_subset)
            all_list.extend(image_list)
    import random
    random.seed(args.random_seed)
    random.shuffle(train_list)
    random.shuffle(test_list)
    random.shuffle(all_list)
    return train_list,test_list,all_list

def parse_data_path(args):

    class_list=args.class_list.split()
    # if available, use pre-split train/test, else, auto split the data_folder_path
    if(args.data_test_path !='' and args.data_train_path!=''):
        _, _,train_list= list_all_files(args,args.data_train_path)

        train_datasize=args.num_instance_each_class
        args.num_instance_each_class=args.num_instance_each_class_test

        _, _,test_list= list_all_files(args,args.data_test_path)
        all_list=train_list+test_list
        args.num_instance_each_class=train_datasize
    else:
        data_dir = args.data_folder_path
        print('parse data from {}'.format(data_dir))
        train_list, test_list,all_list= list_all_files(args,data_dir)
    os.makedirs("../{}/data_cache/".format(args.split),exist_ok=True)
    for stage in ['train','test','all']:
        if(stage=='train'):
            image_list=train_list
        elif(stage=='test'):
            image_list=test_list
        else:
            image_list=all_list
        # folder need to be like */folder/timestamp/class/image.png
        with open('../{}/data_cache/data_{}_path.txt'.format(args.split,stage) , 'w') as file:
            file.write("file class_index timestamp")
            for item in image_list:
                name_list=item.split('/')
                classes=name_list[-2]
                if classes not in class_list:
                    continue
                class_index=class_list.index(classes)
                timestamp=name_list[-3]
                # timestamp=name_list[-3].split('_')[-1] # since name is bucket_x
                file.write("\n")
                file.write(item+ " "+str(class_index)+" "+str(timestamp))
        print('{} parse path finish!'.format(stage))
