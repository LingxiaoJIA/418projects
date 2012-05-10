#!/usr/bin/python

import re, os, operator

#testdir = './img/phptune/'
#testdir = './img/nettune/'
testdir = './img/recaptchatune_proc3/'
#testdir = './img/nocheattest_proc3/'
caprequired = False
canmissone = True

missed_letters = dict()
for l in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
    missed_letters[l] = 0

def get_res(img):
    print 'running ' + str(img)
    #cmd = './charTest -s 0 -e 62 -p 1 -l lib/php/ '
    #cmd = './charTest -s 0 -e 26 -p 2 -l lib/net/ '
    cmd = './charTest -s 0 -e 52 -p 3 -l lib/recaptcha_v4_proc3/ '
    cmd += testdir + img
    f = os.popen(cmd)
    for l in f.readlines():
        if 'ms' in l :
            print l.rstrip()
        lastline = l.rstrip()
    lastline = re.sub(r'\s', '', lastline)
    return lastline



def score_bw(res, cor):
    resp = res[::-1]
    corp = cor[::-1]
    s= score_fw(resp, corp)
    return s

def score_fw(res, cor):
    right = 0
    missing = 0
    extra = 0
    total = len(res)
    for c in cor:
        try:
            loc = res.index(c)
       #     print '[found ' + c + '@' + str(loc) + ']'
            extra += len(res[0:loc])
            res = res[loc+1:]
            right += 1
        except:
            # c not in res
            missing += 1
            missed_letters[c] = missed_letters[c] + 1
    extra -= missing
    s = (float(right) / total)
    if canmissone and extra <= 0 and missing <= 1:
        return (s,True)
    else:
        return (s,False)

def score(res,cor):
    fw = score_fw(res,cor)
    bw = score_bw(res,cor)
    cor = fw[1] or bw[1]
    return (max(fw[0], bw[0]), cor)

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
    print("%.2f" % s[0]),
    print s[1],
    print '\t' + str(res) + '\n'
    char_right += s[0] * len(correct)
    if s[1]:
        word_right += 1
    char_cnt += len(correct)
    word_cnt += 1

print 'missed letters'
missed_letters = sorted(missed_letters.iteritems(), key=operator.itemgetter(1))
print missed_letters

print '\n\nChar Accuracy: ' + str(char_right / float(char_cnt))
print 'Word Accuracy: ' + str(float(word_right) / float(word_cnt))


