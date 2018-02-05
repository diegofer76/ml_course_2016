#!/usr/bin/python

import sys

def ejercicio1(x):
    y = 6*(x**2) + 3*x + 2
    print "el resultado es: " + str(y)

def ejercicio2(x):
    y = x**2
    print "y( " + str(x) + " )" + " = " + str(y)

def ejercicio3(n):
    for i in range(1,n):
        x = i**2 + 1
        print "la respuesta " + str(i) + " es: " + str(x)

def ejercicio4(n):
    i = 1
    while i < n:
        x = i**2 + 1
        print "la respuesta " + str(i) + " es: " + str(x)
        i = i + 1

def main():
    print "-----------ejercicio1------------"
    x = 2
    ejercicio1(x)

    print "-----------ejercicio2------------"
    x = 5
    ejercicio2(x)

    print "-----------ejercicio3------------"
    n = 31
    ejercicio3(n)

    print "-----------ejercicio4------------"
    n = 31
    ejercicio4(n)


if  __name__ == "__main__":
    sys.exit(main())
