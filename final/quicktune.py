#!/usr/bin/python

import os, operator

#testdir = './img/phptune/'
#testdir = './img/nettune/'
testdir = './img/recaptchatune_proc3/'
#testdir = './img/nocheattest_proc3/'
caprequired = False
canmissone = False

missed_letters = dict()
for l in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
    missed_letters[l] = 0

def get_res(img):
    print 'running ' + str(img)
    #cmd = './charTest -s 0 -e 62 -p 1 -l lib/php/ '
    #cmd = './charTest -s 0 -e 26 -p 2 -l lib/net/ '
    cmd = './charTest -s 0 -e 52 -p 0 -l lib/recaptcha_v3_proc3/ '
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
            missed_letters[c] = missed_letters[c] + 1
    if canmissone and wrong > 0:
        wrong -= 1
        right += 1
    s = (float(right) / float(right+wrong))
    return s

def score(res,cor):
    fw = score_fw(res,cor)
    bw = score_bw(res,cor)
    return max(fw, bw)

print 'launching...'


char_right = 0.0
word_right = 0
char_cnt = 0
word_cnt = 0
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
    char_right += s * len(correct)
    if s == 1.0:
        word_right += 1
    char_cnt += len(correct)
    word_cnt += 1

print 'missed letters'
missed_letters = sorted(missed_letters.iteritems(), key=operator.itemgetter(1))
print missed_letters

print '\n\nChar Accuracy: ' + str(char_right / float(char_cnt))
print 'Word Accuracy: ' + str(float(word_right) / float(word_cnt))


