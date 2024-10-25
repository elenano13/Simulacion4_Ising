import numpy as np 
import matplotlib.pyplot as plt 
import numba 
from numba import njit 
from scipy.ndimage import convolve

N = 50 

def inicializar_spins(N):
    init_random = np.random.random((N, N))
    spinsini = np.zeros((N, N))
    spinsini[init_random >= 0.75] = 1
    spinsini[init_random < 0.75] = -1
    return spinsini

spinsini = inicializar_spins(N)

# Visualizar configuración inicial 
plt.imshow(spinsini, cmap='summer') 
plt.title('Configuración inicial') 
plt.colorbar() 
plt.show()

def energia(array_spins):
    #Se utiliza wrap para simular las condiciones periódicas del lattice
    kern = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    arr = -array_spins * convolve(array_spins, kern, mode='wrap')
    return arr.sum()

@njit
def metropolis(spin_arr, times, beta, energia):
    spin_arr = spin_arr.copy()
    spinsnetos = np.zeros(times-1)
    energianeta = np.zeros(times-1)
    
    for t in range(0, times-1):
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)
        spin_i = spin_arr[x, y]  
        spin_f = spin_i * -1  

        E_i = 0
        E_f = 0
        
        # Calculate energy based on neighboring spins
        E_i += -spin_i * spin_arr[(x-1) % N, y]
        E_f += -spin_f * spin_arr[(x-1) % N, y]
        E_i += -spin_i * spin_arr[(x+1) % N, y]
        E_f += -spin_f * spin_arr[(x+1) % N, y]
        E_i += -spin_i * spin_arr[x, (y-1) % N]
        E_f += -spin_f * spin_arr[x, (y-1) % N]
        E_i += -spin_i * spin_arr[x, (y+1) % N]
        E_f += -spin_f * spin_arr[x, (y+1) % N]

        dE = E_f - E_i
        if (dE > 0) and (np.random.random() < np.exp(-beta * dE)):
            spin_arr[x, y] = spin_f
            energia += dE
        elif dE <= 0:
            spin_arr[x, y] = spin_f
            energia += dE

        spinsnetos[t] = spin_arr.sum()  
        energianeta[t] = energia  

    return spinsnetos, energianeta, spin_arr

energia_inicial = energia(spinsini)


temperaturas = np.linspace(0.1, 50.0, 20)
energiaspromedio = []
magnetizacionespromedio = []
susceptibilidades = []
capacidadescalorificas = []
spinsfinal = []

for T in temperaturas:
    beta = 1 / T  #J/kB = 1
    spins, energias, spinsfin = metropolis(spinsini, 1000000, beta, energia_inicial)
    
    
    spinsfinal.append(spinsfin.copy())

    # Después del equilibrio se calculan las cantidades termodinánimas
    pasos_equi = int(0.4 * len(spins))
    magnetizacion_promedio = np.mean(spins[pasos_equi:])
    energia_promedio = np.mean(energias[pasos_equi:])
    energiaspromedio.append(energia_promedio)
    magnetizacionespromedio.append(magnetizacion_promedio)

    # Para calcular susceptibilidades y capacidades calorídicas
    magnetizacion_cuadrado = np.mean(spins[pasos_equi:] ** 2)
    energia_cuadrado = np.mean(energias[pasos_equi:] ** 2)
    
    susceptibilidad = (magnetizacion_cuadrado - magnetizacion_promedio ** 2) * beta
    capacidad_calo = (energia_cuadrado - energia_promedio ** 2) * beta ** 2

    susceptibilidades.append(susceptibilidad)
    capacidadescalorificas.append(capacidad_calo)

# Energía vs Temp
plt.figure()
plt.plot(temperaturas, energiaspromedio, marker='o')
plt.title('Energía promedio vs Temperatura')
plt.xlabel('Temperatura')
plt.ylabel('Energía promedio')
plt.grid()
plt.show()

# Mag vs Temp
plt.figure()
plt.plot(temperaturas, magnetizacionespromedio, marker='o', color='orange')
plt.title('Magnetización promedio vs Temperatura')
plt.xlabel('Temperatura')
plt.ylabel('Magnetización promedio')
plt.grid()
plt.show()

# Susceptibilidad vs Temp
plt.figure()
plt.plot(temperaturas, susceptibilidades, marker='o', color='green')
plt.title('Susceptibilidad vs Temperatura')
plt.xlabel('Temperatura')
plt.ylabel('Susceptibilidad')
plt.grid()
plt.show()

# CC vs Temp
plt.figure()
plt.plot(temperaturas, capacidadescalorificas, marker='o', color='red')
plt.title('Capacidad Calorífica vs Temperatura')
plt.xlabel('Temperatura')
plt.ylabel('Capacidad Calorífica')
plt.grid()
plt.show()

# Coniguración final
plt.figure()
plt.imshow(spinsfinal[-1], cmap='summer')  

# Porcentajes de spins 1 y -1
numerodespins = N * N
positivos = np.sum(spinsfinal[-1] == 1)
negativos = np.sum(spinsfinal[-1] == -1)
porcentajepositivo = (positivos / numerodespins) * 100
porcentajenegativo = (negativos / numerodespins) * 100


plt.title(f'Configuración final a T = {temperaturas[-1]:.2f}\n'
          f'Porcentaje de spins +1: {porcentajepositivo:.2f}%, '
          f'Porcentaje de spins -1: {porcentajenegativo:.2f}%')
plt.colorbar()
plt.show()

print("Resultados de equilibrio a diferentes temperaturas:")
for T, E, M, sus, C in zip(temperaturas, energiaspromedio, magnetizacionespromedio, susceptibilidades, capacidadescalorificas):
    print(f"T: {T:.2f}, Energía: {E:.4f}, Magnetización: {M:.4f}, Susceptibilidad: {sus:.4f}, Capacidad Calorífica: {C:.4f}")
