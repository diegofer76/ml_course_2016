#!/usr/bin/python

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

def funcion_que_vas_a_implementar():
	print "implementame papu! : " + x

def main():
	x = 2
	funcion_que_vas_a_implementar(x)

if __name__ == "__main__":
    sys.exit(main())