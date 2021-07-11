for ((i=4;i<6;i++));
do

python run_model_aug2.py \
--do_train \
--bert_model /home/ypd-19-2/abu/beyond_output/pre_train_roberta_n_gram_mask/90epoch \
--output_dir /home/ypd-19-2/abu/beyond_output/roberta_100_3_epoch/$i \
--data_dir  /home/ypd-19-2/abu/beyond_baseline/10-fold/$i \
--adv_type  fgm

done