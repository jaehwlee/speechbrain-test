import os
import csv

# 현재 스크립트는 대문자, 인퍼런스 결과는 소문자기 때문에 wer계산 불가
f = open('jhlee_librivox-test-clean.csv', 'r')
rdr = csv.reader(f)
lines = []

for line in rdr:
    line[2] = line[2].upper()
    lines.append(line)

f = open('jhlee_upper.csv', 'w', newline='')
wr = csv.writer(f)
wr.writerows(lines)

f.close()
