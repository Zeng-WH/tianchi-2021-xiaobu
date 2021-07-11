for ((i=8;i<10;i++));
do

python run_model_aug4.py \
--do_train \
--bert_model /home1/wlw2020/head_motion/abu/beyond_output/pre_train_roberta_n_gram \
--output_dir /home1/wlw2020/head_motion/abu/beyond_output/roberta_100_3_epoch/$i \
--data_dir  /home1/wlw2020/head_motion/abu/beyond_baseline/10-fold/$i \
--adv_type  fgm

done