# задание
TASK = "0,07;  0,052; 0,084;  0,098;  0,079;  0,054;  0,12;  0,09;   0,074; 0,06;  0,082;  0,104;  0,086;  0,065;  0,036;  0,036;  0,087; 0,036;  0,091;  0,045;  0,062;  0,073;  0,094;  0,056;  0,083; 0,115; 0,08; 0,108; 0,068; 0,085"
#TASK = "0,178; 0,134; 0,202; 0,25; 0,205; 0,147; 0,299; 0,232; 0,192; 0,16;  0,209;  0,258; 0,215; 0,165; 0,117; 0,117; 0,23; 0,193; 0,246; 0,132; 0,173; 0,18; 0,236; 0,158; 0,198; 0,294; 0,196; 0,275; 0,166;  0,213"
TASK = "1,147; 1,211; 1,088; 1,025; 1,143; 1,2; 0,998; 1,077; 1,145; 1,190; 1,181; 1,136; 1,052; 1,202; 1,108; 1,101; 1,118; 1,025; 1,144; 1,102; 1,122;1,029; 1,092; 1,155; 1,277; 1,123; 1,086; 1,139; 1,081; 1,237"


import matplotlib.pyplot as plt
import numpy as np
from asciitable import AsciiTable
from collections import Counter
from fractions import Fraction
from decimal import Decimal
from math import sqrt

TASK = TASK.replace(',', '.')
raw_data = list(map(Decimal, TASK.split(';')))

cnt = Counter()
for value in raw_data:
    cnt[value] += 1

data = []
for i in raw_data:
    if i not in data:
        data.append(i)

# дискретный вариационный ряд

freq = [cnt[i] for i in data]

data_str = data[:]
freq_str = freq[:]

data_str.insert(0, 'xi')
freq_str.insert(0, 'mi')

print("Дискретный вариационный ряд")
discrete_table = AsciiTable([data_str, freq_str], header=False, separateLines=True)
print(discrete_table)

# дискретный вариационный ряд относительных частот (статистическое распределение)
w_freq = list(map(Fraction, freq))
w_freq = [i / len(raw_data) for i in w_freq]
w_freq_str = w_freq[:]
w_freq_str.insert(0, 'wi')

print("Дискретный вариационный ряд относительных частот (статистическое распределение)")
discrete_table = AsciiTable([data_str, w_freq_str], header=False, separateLines=True)
print(discrete_table)

x_min = min(raw_data)
x_max = max(raw_data)
h = (x_max - x_min) / 5
print("x_min =", x_min)
print("x_max =", x_max)
print("h = (x_max - x_min) / 5 =", h)
print("a1 = x_min =", x_min)
print()

intervals = []
a = x_min
for i in range(5):
    b = a + h
    intervals.append((a, b))
    a = b
intervals_str = []
for i in intervals:
    a, b = i
    a_str = str(a).rstrip('0')
    b_str = str(b).rstrip('0')
    intervals_str.append("{} - {}".format(a_str, b_str))

intervals_freq = [0] * 5
for i in raw_data:
    for interval_index, interval in enumerate(intervals):
        a, b = interval
        if a < i <= b:
            intervals_freq[interval_index] += 1
index_x_min = data.index(x_min)
intervals_freq[0] += freq[index_x_min] # посчитать x_min как принадлежащий 1 интервалу

print("Интервалы")
intervals_table = [["Интервалы", "mi"]]
for i in range(5):
    intervals_table.append([intervals_str[i], intervals_freq[i]])
intervals_table = AsciiTable(intervals_table, header=True, separateLines=False)
print(intervals_table)

i_discrete_data = [(a + b) / 2 for a, b in intervals]
i_discrete_freq = intervals_freq[:]

i_discrete_data_str = ["xi"] + i_discrete_data[:]
i_discrete_freq_str = ["mi"] + i_discrete_freq[:]

print("Дискретный интервальный ряд")
i_discrete_table = AsciiTable([i_discrete_data_str, i_discrete_freq_str], header=False, separateLines=True)
print(i_discrete_table)

print("[Plot-1]\n")
plt.plot(i_discrete_data, i_discrete_freq, 'b.-')
plt.xlabel("xi")
plt.ylabel("mi")
plt.grid()
plt.show()

# дискретный вариационный ряд относительных частот (статистическое распределение)
w_i_discrete_freq = list(map(Fraction, i_discrete_freq))
w_i_discrete_freq = [i / len(raw_data) for i in w_i_discrete_freq]
dec_w_i_discrete_freq = [Decimal(wi.numerator) / Decimal(wi.denominator) for wi in w_i_discrete_freq]
w_i_discrete_freq_str = []
round_dec_w_i_discrete_freq = []
for i in range(5):
    wi = w_i_discrete_freq[i]
    dec_wi = round(dec_w_i_discrete_freq[i], 2)
    round_dec_w_i_discrete_freq.append(dec_wi)
    w_i_discrete_freq_str.append("{} ~ {}".format(wi, dec_wi))
w_i_discrete_freq_str.insert(0, 'wi')

print("Дискретный интервальный ряд относительных частот")
w_i_discrete_table = AsciiTable([i_discrete_data_str, w_i_discrete_freq_str], header=False, separateLines=True)
print(w_i_discrete_table)

print("[Plot-2]\n")
plt.plot(i_discrete_data, w_i_discrete_freq)
plt.xlabel("xi")
plt.ylabel("wi")
plt.grid()
plt.show()

i_discrete_hist = [round(float(w_i_discrete_freq[i] / Fraction(h)), 3) for i in range(5)]

print("Интервалы гистограммы")
hist_intervals_table = [["Интервалы", "wi", "wi/h"]]
for i in range(5):
    hist_intervals_table.append([intervals_str[i], w_i_discrete_freq_str[i + 1], i_discrete_hist[i]])
hist_intervals_table = AsciiTable(hist_intervals_table, header=True, separateLines=False)
print(hist_intervals_table)

print("[Hist-1]\n")
bins = [a for a, b in intervals] + [intervals[-1][-1]]
hist_weights = [float(h) for x in raw_data]
plt.hist(x=list(map(float, raw_data)), bins=list(map(float, bins)), weights=hist_weights, histtype='stepfilled')
plt.ylabel("mi/h")
plt.grid()
plt.show()

print("[Hist-2]\n")
bins = [a for a, b in intervals] + [intervals[-1][-1]]
plt.hist(x=list(map(float, raw_data)), bins=list(map(float, bins)), histtype='stepfilled', normed=True)
plt.ylabel("wi/h")
plt.grid()
plt.show()

""" for discrete
w_cum_i_discrete_freq = []
w_cum = 0
for i in range(5):
    w_cum_i_discrete_freq.append(w_cum + w_i_discrete_freq[i])
    w_cum += w_i_discrete_freq[i]
dec_w_cum_i_discrete_freq = [Decimal(wi.numerator) / Decimal(wi.denominator) for wi in w_cum_i_discrete_freq]
w_cum_i_discrete_freq_str = ["wi нак"]
for i, wi_cum in enumerate(w_cum_i_discrete_freq):
    w_cum_i_discrete_freq_str.append("{} ~ {}".format(wi_cum, round(dec_w_cum_i_discrete_freq[i], 2)))
print("Кумулятивная кривая для дискретного ряда")
w_cum_i_discrete_table = AsciiTable([i_discrete_data_str, w_cum_i_discrete_freq_str], header=False, separateLines=True)
print(w_cum_i_discrete_table)

print("[Plot-2]\n")
plt.plot(i_discrete_data, w_cum_i_discrete_freq)
plt.plot(i_discrete_data, w_cum_i_discrete_freq, 'ro')
plt.xlabel("xi")
plt.ylabel("wi нак")
plt.grid()
plt.show()
"""
w_cum_i_discrete_freq = []
w_cum = 0
for i in range(5):
    w_cum_i_discrete_freq.append(w_cum + w_i_discrete_freq[i])
    w_cum += w_i_discrete_freq[i]
dec_w_cum_i_discrete_freq = [Decimal(wi.numerator) / Decimal(wi.denominator) for wi in w_cum_i_discrete_freq]
w_cum_i_discrete_freq_str = ["wi нак"]
for i, wi_cum in enumerate(w_cum_i_discrete_freq):
    w_cum_i_discrete_freq_str.append("{} ~ {}".format(wi_cum, round(dec_w_cum_i_discrete_freq[i], 2)))

print("Кумулятивная кривая для дискретного ряда")
w_cum_i_discrete_table = AsciiTable([i_discrete_data_str, w_cum_i_discrete_freq_str], header=False, separateLines=True)
print(w_cum_i_discrete_table)

print("[Plot-3]\n")
plt.plot(i_discrete_data, w_cum_i_discrete_freq)
plt.plot(i_discrete_data, w_cum_i_discrete_freq, 'ro')
plt.xlabel("xi")
plt.ylabel("wi нак")
plt.grid()
plt.show()

w_cum_i_discrete_freq = []
w_cum = 0
for i in range(5):
    w_cum_i_discrete_freq.append(w_cum + w_i_discrete_freq[i])
    w_cum += w_i_discrete_freq[i]
dec_w_cum_i_discrete_freq = [Decimal(wi.numerator) / Decimal(wi.denominator) for wi in w_cum_i_discrete_freq]
interval_cum_table = [["Интервалы", "wi нак"]]
for i, interval in enumerate(intervals):
    a, b = interval
    wi_cum_str = "{} ~ {}".format(w_cum_i_discrete_freq[i], round(dec_w_cum_i_discrete_freq[i], 2))
    interval_cum_table.append(["{} - {}".format(a, b), wi_cum_str])
print("Кумулятивная кривая для интервального ряда")
w_cum_i_discrete_table = AsciiTable(interval_cum_table, header=True, separateLines=False)
print(w_cum_i_discrete_table)

w_cum_i_discrete_freq = [0] + w_cum_i_discrete_freq

print("[Plot-4]\n")
plt.plot(bins, w_cum_i_discrete_freq)
plt.plot(bins, w_cum_i_discrete_freq, 'ro')
plt.xlabel("xi")
plt.ylabel("wi нак")
plt.grid()
plt.show()

print("*************** Характеристики ***************\n")

print("Средняя выборочная")
print("x- = sum[i = 1..k](xi * mi) / n")
sum_str = []
for i in range(len(data)):
    sum_str.append(str(data[i]) + "*" + str(freq[i]))
print("x- = (", " + ".join(sum_str), ") /", len(raw_data))
x_avg = round(sum(raw_data) / len(raw_data), 3)
print("x- =", x_avg)
print()

print("Исправленная выборочная дисперсия")
print("S*^2 = sum[i = 1..k](mi * (xi - x-)^2) / (n - 1)")
sum_str = []
for i in range(len(data)):
    sum_str.append(str(freq[i]) + "*(" + str(data[i]) + "-" + str(x_avg) + ")^2")
print("S*^2 = (", " + ".join(sum_str), ") /", len(raw_data) - 1)
S2 = sum((freq[i] * (data[i] - x_avg) ** 2 for i in range(len(data)))) / (len(raw_data) - 1)
print("S*^2 =", S2)
print()

print("Среднее квадратичное отклонение")
print("S* = sqrt( S*^2 )")
S = round(sqrt(S2), 6)
print("S* = sqrt(", S2, ") =", S)
print()

print("Размах вариации")
print("R = x_max - x_min")
R = x_max - x_min
print("R =", x_max, "-", x_min, "=", R)
print()

print("Коэффициент вариации")
print("V = (S* / x-) * 100%")
V = round(S / float(x_avg) * 100, 6)
print("V =", "(", S, "/", x_avg, ") * 100% =", str(V) + "%")
print()

