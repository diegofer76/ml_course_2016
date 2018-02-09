#!/usr/bin/python
import sys

# La linea que esta aqui arriba NO es un comentario, 
# siempre debe ir en un archivo ejecutable de python

# este archivo se llama exemplo.py
# lo ejecutas:
# ./exemplo.py 
# (tambien se puede ejecutar usando >> python exemplo.py , pero mucha letra para digitar.)
# si da error es por problemas de permisos del archivo, entonces haz esto:
# chmod 755 exemplo.py 
# (este comando chmod es solo necesario ejecutarlo una vez.)



# esta funcion debe ser implementada
# estos comentarios deben ser borrados.

def fibonacci(n):
    x = 0
    y = 1
    while x < n:
        print "la sucesion de fibonacci es: " + str(x)
        z = x
        x = y
        y = z + y

def main():
    n = 100
    fibonacci(n)

if  __name__ == "__main__":
    sys.exit(main())
