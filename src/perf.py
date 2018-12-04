# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Tue Dec  4 18:29:33 2018

@author: MerelyLogical
"""

from time import perf_counter

# ------------------------------------------------------------------------------
# Performance
# ------------------------------------------------------------------------------
def start():
    return [perf_counter()]

def lap(event, records):
    'stopwatch'
    t = perf_counter()
    print ('[Timer] {0} took {1:.2f}s, total time {2:.2f}s'\
           .format(event, t-records[-1], t-records[0]))
    records.append(t)
    return None