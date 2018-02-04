#!/usr/bin/python

import sys

def ejercicio1(x):
    y = 6*(x**2) + 3*x + 2
    print "el resultado es: " + str(y)

def ejercicio2(x):
    y = x**2
    print "y( " + str(x) + " )" + " = " + str(y)

def ejercicio3(n):
    x = n**2 + 1
    print "la respuesta " + str(n) + " es: " + str(x)

def main():
    print "-----------ejercicio1------------"
    x = 2
    ejercicio1(x)

    print "-----------ejercicio2------------"
    x = 5
    ejercicio2(x)
    
    print "-----------ejercicio3------------"
    for n in range(1,31):
        ejercicio3(n)


if __name__ == "__main__":
    sys.exit(main())
