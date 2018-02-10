#!/usr/bin/python

import sys

def fibo(n):
    x = 0
    y = 1
    while x < n:
        print "la sucesion de fibonacci es: " + str(x)
        z = x
        x = y
        y = z + y

def main(): 
    n = 100
    fibo(n)

if  __name__ == "__main__":
    sys.exit(main())
