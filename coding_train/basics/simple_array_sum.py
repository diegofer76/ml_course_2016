#!/usr/bin/python


# Arrays
# ======
#
# Un array es una estructura de datos parecida com uma lista.
# En python un array vacio puede ser declarado de la siguiente forma:
#   nuevo_array = []    
#
# O un array com elementos:
#   nuevo_array_filled = [1, 2, 3, 4]
#
# Es posible saber el tamanho del array asi:
#   tamanho = len(nuevo_array_filled)
#  
# Agregamos un nuevo elemento (a modo de ejemplo agregamos un entero = 6) asi:
#   nuevo_array_filled.append(6)
#
# Podemos accesar un determinado elemento asi:
#   elemento = nuevo_array_filled[0]
# En este caso la variable 'elemento' tendra el valor = 1

# Un array puede ser recorrida facilmente usando un loop 'for', asi:
#   for i in nuevo_array_filled:



#implementar aqui la suma de los elementos
def simple_array_sum(arr):
    sum_value = 0
    # aqui abajo implementa la suma.
    # Hint: Usa un for para recorrer el array 'arr'
    for n in arr:
        sum_value = sum_value + n
    return sum_value


def main():
    # Lee los elementos del array
    #arr = map(int, raw_input().strip().split(' '))
    arr = [1, 2, 3, 4, 6]
    simple_array_sum(arr)
    # Llama la funcion que hace la suma
    result = simple_array_sum(arr)
    print "la suma total es: " + str(result)


if  __name__ == "__main__":
    main()
