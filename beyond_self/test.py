import numpy as np

'''
probs_a = np.load('test_a_3_30.npy')
resu = np.mean(probs_a, axis=0)
probs_b = np.load('test_b_3_30.npy')
resu1 = np.mean(probs_b, axis=0)
'''
'''
resu2 = np.load('test_a_4_7_bert.npy')
resu3 = np.load('test_b_4_7_bert.npy')
resu2 = np.mean(resu2, axis=0)
resu3 = np.mean(resu3, axis=0)
resu4 = np.load('test_a_4_4_roberta.npy')
resu4 = np.mean(resu4, axis=0)
resu5 = np.load('test_b_4_4_roberta.npy')
resu5 = np.mean(resu5, axis=0)
'''

'''
probs_roberta_a = np.load('test_a_3_31_roberta_test.npy')
resu4 = np.mean(probs_roberta_a, axis=0)
probs_roberta_b = np.load('test_b_3_31_roberta_test.npy')
resu5 = np.mean(probs_roberta_b, axis=0)
'''
'''
probs_nezha_a = np.load('/home/ypd-19-2/abu/NeZha_Chinese_PyTorch-main/model/test_a_3_31_nezha.npy')
resu4 = np.mean(probs_nezha_a, axis=0)
probs_nezha_b = np.load('/home/ypd-19-2/abu/NeZha_Chinese_PyTorch-main/model/test_b_3_31_nezha.npy')
resu5 = np.mean(probs_nezha_b, axis=0)
'''

'''
resu6 = np.load('/home/ypd-19-2/abu/NeZha_Chinese_PyTorch-main/model/test_a_4_3_nezha.npy')
resu6 = np.mean(resu6, axis=0)
resu7 = np.load('/home/ypd-19-2/abu/NeZha_Chinese_PyTorch-main/model/test_b_4_3_nezha.npy')
resu7 = np.mean(resu7, axis=0)

resu8 = np.load('/home/ypd-19-2/abu/NeZha_Chinese_PyTorch-main/model/test_a_4_7_nezha.npy')
resu8 = np.mean(resu8, axis=0)

resu9 = np.load('/home/ypd-19-2/abu/NeZha_Chinese_PyTorch-main/model/test_b_4_7_nezha.npy')
resu9 = np.mean(resu9, axis = 0)

resu10 = np.load('test_a_4_8_roberta.npy')
resu10 = np.mean(resu10, axis = 0)
resu11 = np.load('test_b_4_8_roberta.npy')
resu11 = np.mean(resu11, axis = 0)
'''

resu1 = np.load('/home/ypd-19-2/abu/NeZha_Chinese_PyTorch-main/model/test_B_a_4_8_nezha.npy')
resu1 = np.mean(resu1, axis=0)
resu2 = np.load('/home/ypd-19-2/abu/NeZha_Chinese_PyTorch-main/model/test_B_b_4_8_nezha.npy')
resu2 = np.mean(resu2, axis=0)
resu3 = np.load('test_B_a_4_8_bert.npy')
resu3 = np.mean(resu3, axis=0)
resu4 = np.load('test_B_b_4_8_bert.npy')
resu4 = np.mean(resu4, axis=0)
resu5 = np.load('test_B_a_4_8_roberta.npy')
resu5 = np.mean(resu5, axis=0)
resu6 = np.load('test_B_b_4_8_roberta.npy')
resu6 = np.mean(resu6, axis=0)

resutest = np.load('/home/ypd-19-2/abu/tianchi_project/user_data/tmp_data/test_B_b_nezha.npy')
resutest = np.mean(resutest, axis=0)

resu_return = (resu1 + resu2 + resu3 + resu4 + resu5 + resu6) / 6



with open('result.tsv', 'w') as w:
    for prob in resu_return:
        w.write(str(prob))
        w.write('\n')
print('bupt')