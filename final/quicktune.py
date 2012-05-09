#!/usr/bin/python

import os

#testdir = './img/phptune/'
#testdir = './img/nettune/'
testdir = './img/recaptchatune/'
caprequired = False

def get_res(img):
    print 'running ' + str(img)
    #cmd = './charTest -s 0 -e 62 -p 1 -l lib/php/ '
    #cmd = './charTest -s 0 -e 26 -p 2 -l lib/net/ '
    cmd = './charTest -s 0 -e 52 -p 1 -l lib/recaptcha/ '
    cmd += testdir + img
    f = os.popen(cmd)
    for l in f.readlines():
        if 'ms' in l :
            print l.rstrip()
        lastline = l.rstrip()
    return lastline



def score_bw(res, cor):
    resp = res[::-1]
    corp = cor[::-1]
    s= score_fw(resp, corp)
    return s

def score_fw(res, cor):
    right = 0
    wrong = 0
    for c in cor:
        try:
            loc = res.index(c)
       #     print '[found ' + c + '@' + str(loc) + ']'
            res = res[loc+1:]
            right += 1
        except:
            # c not in res
            wrong += 1
    s = (float(right) / float(right+wrong))
    return s

def score(res,cor):
    fw = score_fw(res,cor)
    bw = score_bw(res,cor)
    return max(fw, bw)

print 'launching...'


tot_sum = 0.0
tot_right = 0
tot_cnt = 0
for correct in os.listdir(testdir):
    res = get_res(correct)
    correct = correct.split('.')[0]
    if not caprequired:
        correct = correct.lower()
        res = res.lower()
    s = score(res, correct)
    print 'score: ',
    print("%.2f" % s),
    print '\t' + str(res) + '\n'
    tot_sum += s
    if s == 1.0:
        tot_right += 1
    tot_cnt += 1

print '\n\nChar Accuracy: ' + str(tot_sum / float(tot_cnt))
print 'Word Accuracy: ' + str(float(tot_right) / float(tot_cnt))


