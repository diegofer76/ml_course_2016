#!/usr/bin/python

# Los Numeros o Secuencia de Fibonacci
# ====================================
# 
# Los numero de fibonacci son una secuencia de numeros que tienen la siguiente propiedad:
#       
#   F_n = F_(n-1) + F_(n-2)
#
#   Con la excepcion de que para F_0 = 0 y para F_1 = 1
#
# Osea la secuencia es:
# 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, .... (asi hasta perder el conocimiento...)     
#


# Este desafio consiste en implementar una funcion que imprima 
# los 'n' primeros elementos de la secuencia de fibonacci.
#


def fibonacci(n):
    print "secuencia fibonacci:"
    # implementar aqui


def main():
    num = 7
    fibonacci(num)

if __name__ == "main":
    main()
