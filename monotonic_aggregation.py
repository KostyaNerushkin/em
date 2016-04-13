from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice

frame = pd.read_csv("quotes.csv");
# print(frame)
# print(frame.get_chunk(5))

fx_series = frame.ix[:, ['closeBid']]


class Series(object):
	def __init__(self, series, span):
		self.series = series
		self.span = span
		self.padding()

	def padding(self):
		zero = [ 0 ]
		self.series = zero * self.span + self.series + zero * self.span

	def __iter__(self):
		return self.window()

	def __getitem__(self, item):
		return self.series[item]

	def window(self):
		it = iter(self.series)
		result = tuple(islice(it, self.span))
		if len(result) == self.span:
			yield result
		for elem in it:
			result = result[1:] + (elem,)
			yield result


class MonotonicAggregationTree(object):
	def __init__(self, series, span):
		self.span = span
		self.series = Series(series, span)

	def runs(self):
		extrema = []
		result = []
		series0, series1 = Series(self.series[:-1], self.span), Series(self.series[1:], self.span)
		for window0, window1 in zip(series0, series1):
			extrema.append(np.array(window1) - np.array(window0))
		for i, j in zip(extrema[:-1], extrema[1:]):
			print(i, j)
			if i * j  <= 0:
				result.append(i)
		return result


def monotonic_aggregation(data):
	series = data[:-1]
	series_1 = data[1:]
	extremes = []
	hull_min = []
	hull_max = []
	skeleton = []
	window_length = 10

	for k in range(0, len(series) - window_length):
		count = 0 + k
		for one, two in zip(series[0 + k: window_length + k], series_1[0 + k: window_length + k]):
			extremes.append((two - one, count))
			count += 1
		# print(extremes)
		skeleton_max = []
		skeleton_min = []
		for i, j in zip(extremes[:-1], extremes[1:]):
			if i[0] * j[0] <= 0 and i[0] > 0:
				skeleton_max.append(i[1])
			elif i[0] * j[0] <= 0 and i[0] < 0:
				skeleton_min.append(i[1])
		# print('&')
		# print(skeleton_max)
		# print(skeleton_min)
		# print('&')
		if skeleton_min != [] and skeleton_max != []:
			hull_max.append(skeleton_max)
			hull_min.append(skeleton_min)
			# print('Hull:', skeleton_max, skeleton_min)
		extremes = []
	max_X = []
	index_max_X = []
	min_X = []
	index_min_X = []
	# print(hull_max,hull_min)
	for index in hull_max:
		max_X.append(series[1 + index[0]])
		index_max_X.append(1 + index[0])
	for index in hull_min:
		min_X.append(series[1 + index[0]])
		index_min_X.append(1 + index[0])

	return (index_min_X, min_X), (index_max_X, max_X)


nsteps = 100
draws = np.random.randn(nsteps)
walk = draws.cumsum()
(index_min_X, min_X), (index_max_X, max_X) = monotonic_aggregation(walk)
# TODO: monotonic aggregation back min/max hull without indexing
# (index_min_X2, min_X2), (index_max_X2, max_X2) = monotonic_aggregation(max_X)
print('#' * 50)
tree = MonotonicAggregationTree(walk.tolist(), span=10)
print(tree.runs())
print(len(max_X), len(walk))
exit(0)
print('#' * 50)
# print(max_X)
# print(min_X)
plt.plot(range(len(walk)), walk, 'b')
# plt.plot(index_max_X, max_X, 'g')
print(len(max_X), len(walk))
plt.plot(index_min_X, min_X, 'r')
# plt.plot([index for index in index_max_X2], max_X2, 'y')
# plt.plot(index_min_X2, min_X2, 'c')
plt.show()
