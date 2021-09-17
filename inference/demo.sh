CUDA_VISIBLE_DEVICES=3 \
python inference_acc.py \
--loadmodel '../pretrained/deeplabv3plus-xception-vocNov14_20-51-38_epoch-89.pth' \
--img_list /mnt/date/zhaofuwei/Projects/2D-Human-Parsing/demo_imgs/img_list.txt \
--output_dir /mnt/date/zhaofuwei/Projects/2D-Human-Parsing/parsing_result