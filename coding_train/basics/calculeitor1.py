#!/usr/bin/python

import sys

def ejercicio1(x):
    y = 6*(x**2) + 3*x + 2
    print "el resultado es: " + str(y)

def main():
	x = 2
	ejercicio1(x)

if __name__ == "__main__":
    sys.exit(main())
