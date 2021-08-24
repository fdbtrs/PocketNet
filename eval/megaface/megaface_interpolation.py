# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 09:16:50 2018

@author: zhaoy
"""

import numpy as np
from numpy import log10, polyfit, polyval


def linear_interp(x_t, x1, x2, y1, y2):
    assert(x2 > x1)
    assert(x_t >= x1)
    assert(x_t <= x2)

    dx12 = float(x2 - x1)
    f1 = (x_t - x1) / dx12
    f2 = (x2 - x_t) / dx12

    y_t = y1 * f2 + y2 * f1

    # print f1, f2
    # print y_t

    return y_t


def linear_interp_logx(x_t, x1, x2, y1, y2):
    assert(x2 > x1)
    assert(x_t >= x1)
    assert(x_t <= x2)

    if x1 <= 0 or x2 <=0 :
        # x1 = 1e-20
        _x = x_t
        _x1 = x1
        _x2 = x2
    else:
        _x = log10(x_t)
        _x1 = log10(x1)
        _x2 = log10(x2)

    dx12 = float(_x2 - _x1)
    f1 = (_x - _x1) / dx12
    f2 = (_x2 - _x) / dx12

    y_t = y1 * f2 + y2 * f1

    # print f1, f2
    # print y_t

    return y_t


def nearest_neighbor_interp(x_t, xs, ys):
    assert(len(xs)==len(ys))
    xa = np.array(xs)
    ya = np.array(ys)

    inds = np.argsort(xa)
    xa = xa[inds]
    ya = ya[inds]

    nn = len(xa)
    for i in range(nn):
        if xa[i] > x_t:
            break

    if i>0 and i<nn-1:
        if xa[i] - x_t > x_t - xa[i-1]:
            return ya[i-1]
        else:
            return ya[i]
    else:
        return ya[i]


def np_polyfit_interp(x_t, xs, ys, deg):
    assert(len(xs)==len(ys))
    xa = np.array(xs)
    ya = np.array(ys)

    inds = np.argsort(xa)
    xa = xa[inds]
    ya = ya[inds]

    assert(x_t >= xa[0] and x_t <= xa[-1])

    p = polyfit(xa, ya, deg)
    y_t = polyval(p, x_t)

    return y_t


def np_polyfit_interp_logx(x_t, xs, ys, deg):
    assert(len(xs)==len(ys))
    xa = np.array(xs)
    ya = np.array(ys)

    inds = np.argsort(xa)
    xa = xa[inds]
    ya = ya[inds]

    assert(x_t >= xa[0] and x_t <= xa[-1])

    xa = log10(xa)

    p = polyfit(xa, ya, deg)
    y_t = polyval(p, log10(x_t))

    return y_t


if __name__ == '__main__':
    x_t = 1e-6

    # xs,ys are from facenet ROC data
    xs = [
        6.700000199089118e-07,
        9.400000067216752e-07,
        1.259999976355175e-06,
        1.740000016070553e-06
    ]

    ys = [
        0.8436543941497803,
        0.8544462919235229,
        0.8647304773330688,
        0.8747673034667969
    ]

    print ('\n===> interpolating for x_t=', x_t)

    y_t = nearest_neighbor_interp(x_t, xs, ys)
    print ('\n---> NN interpolation: y = %g' % (y_t))

    y_t = linear_interp(x_t, xs[1], xs[2], ys[1], ys[2])
    print ('\n---> Regular lineary interpolation: y_t = ', y_t)

    y_t = linear_interp_logx(x_t, xs[1], xs[2], ys[1], ys[2])
    print ('\n---> Lineary interpolation using log10(x_t): y_t = ', y_t)

    poly_degrees = [1, 2, 3]

    for deg in poly_degrees:
        y_t = np_polyfit_interp(x_t, xs, ys, deg)
        print ('\n---> Numpy polyfit (deg=%d) interpolation: y = %g' % (deg, y_t))

        y_t = np_polyfit_interp_logx(x_t, xs, ys, deg)
        print ('\n---> Numpy polyfit (deg=%d, logx) interpolation: y = %g' % (deg, y_t))