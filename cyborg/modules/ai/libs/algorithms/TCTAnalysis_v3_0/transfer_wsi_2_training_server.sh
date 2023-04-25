#!/bin/bash
read -p "Please input test organization name: " name
echo "The organization name is $name"

s_dir="/d/new_test_img_20X/$name/*"
d_dir="/home/data/kww/datasets/wsi_classify/"$name"_untrained_20X/"
#d_dir="/home/pym/workstation/DB/shanzhongyaofushuyiyuan_untrained_20X/"

#scp -r $s_dir "pym@192.168.111.3:$d_dir"
scp -r $s_dir "kww@192.168.111.3:$d_dir"